# DeepEP 与当前 KunServe 跨实例通信：原理和 bs80 预期

最后更新：2026-06-01

本文只做原理和预期分析，不包含 DeepEP 代码实现。

## 1. 当前 KunServe 通信方案

当前 GLOBAL/BALLOON 模式使用 `CrossReplicaStandardDispatcher`。

一层 MoE 的跨实例通信是 lane 级 dense collective：

```text
每层、每 lane：
  1. local hidden/topk/topk_weight 写入 persistent staging buffer
  2. lane all-gather：同 lane 两个 replica 交换完整 graph_bucket_m rows
  3. remap：global expert id -> 本 rank local compact expert id，不在本卡则 -1
  4. expert kernel：每个 rank 对 union tokens 只算自己保留的 experts
  5. lane reduce-scatter：把 union partial reduce 后切回本 replica token slice
  6. replica 内 TP all-reduce：合并不同 lane 的 partial
```

修复后的状态：

```text
backing buffer capacity = max_capture_m
NCCL tensor view length = graph_bucket_m
```

因此已经不再为小 bucket 发送最大 capture rows。但是它仍然是 dense lane exchange：

| 成本 | 是否已解决 | 说明 |
|---|---|---|
| max_capture_m padding | 已解决 | view 缩到当前 graph bucket |
| graph bucket padding | 未解决 | raw bs=73 replay bucket=80 仍有 padding |
| 无效 expert slot | 未解决 | topk 中不属于本卡的 expert 仍进入 union 元数据和专家前处理 |
| 每层 collective 固定启动 | 未解决 | 80B 每层都有 dispatch + combine |
| replica lockstep | 未解决 | collective 入口要求两端步调一致 |

## 2. DeepEP 的底层思路

DeepEP 面向 MoE token dispatch/combine，而不是通用 dense all-gather。

典型路径可以抽象成：

```text
router topk
  -> 统计 token/expert 目的地 counts
  -> GPU-side prefix sum / permutation
  -> 只把需要远端 expert 的 token payload 发给对应 rank
  -> local expert compute
  -> 按原 token/topk slot combine 回原 rank
```

它和当前 KunServe 的差别：

| 维度 | 当前 KunServe dense lane collective | DeepEP routed exchange |
|---|---|---|
| 发送粒度 | graph_bucket_m rows 的 dense hidden/topk | token/expert slot |
| padding | graph bucket padding 仍进 collective | 可跳过无效/padded slot |
| expert 无效项 | remap 为 -1 后由 expert runner 跳过 | dispatch 阶段就不发给无关 rank |
| 通信语义 | all-gather + reduce-scatter | dispatch + combine |
| kernel/通信组织 | 通用 NCCL collective | MoE 专用 permutation/communication/combine |
| 同步面 | 每层两个 lane collective | 仍需同步，但可减少固定 dense 通信和无效 payload |

DeepEP 快的核心不是“完全没有同步”，而是：

1. 不把所有 token dense 复制给所有同 lane rank。
2. 不传 padded/zero rows 和无效 expert slot。
3. 把 MoE dispatch/combine 的 metadata、permutation、communication 路径做成专用 GPU 流水。
4. 更接近原生 EP 的语义，而不是用 all-gather 模拟 EP。

## 3. 当前 bs80 数据下 DeepEP 的预期

本次严格 run：

```text
/workspace/verl/outputs/prof80b_bs80_bucketfix_20260601_152327
```

stage 均值：

| 贡献项 | mean ms/step |
|---|---:|
| fixed cross-replica | 2.890 |
| lockstep wait | 1.524 |
| attention delta | 3.284 |
| expert delta | -0.791 |
| real iter delta | 8.956 |

DeepEP 主要作用在 `fixed cross-replica`，其次可能降低一部分 lockstep：

| 假设 | per-step 改善 | 说明 |
|---|---:|---|
| 保守 | 1.4-1.8ms | fixed cross-replica 降 50%-60% |
| 中性 | 2.0-2.5ms | fixed 降 60%-70%，lockstep 少量下降 |
| 乐观 | 3.0ms 左右 | fixed 大幅下降且 lockstep 同时改善 |

当前 KunServe global replay 常见 `iter_ms` 约 33-38ms。如果每步少 2-3ms，decode wall-clock 约下降 6%-9%。

端到端粗估：

| 指标 | 当前 bs80 KunServe | DeepEP 后合理目标 |
|---|---:|---:|
| e2e mean | 453s | 425-435s |
| ideal/effective decode mean | 约 426s | 400-410s |
| e2e p50 | 438s | 410-425s |

这个估计只覆盖 MoE communication。它不包含：

1. attention KV 分布优化；
2. scheduler/retract 策略变化；
3. 输出长度变化；
4. 更高 batch size 下的通信/专家负载变化。

## 4. DeepEP 实现时需要保持的语义

替换当前 dispatcher 时必须保持以下语义：

1. attention/KV 仍归属原 replica，不能把请求迁移给对端 attention。
2. dispatch 输出必须等价于当前 union token 语义：所有需要本 rank expert 的 token 都能被本 rank 计算。
3. combine 输出必须回到原 replica、原 token 顺序，shape 与 local MLP 输出一致。
4. padding/dummy/phantom rows 不能影响 logits。
5. 先结束的 idle replica 仍需 keepalive，直到对端也结束或者 global mode restore。
6. CUDA graph replay 下所有 buffer 指针必须稳定；DeepEP 的 workspace 也要在 capture 前固定。

## 5. 为什么 DeepEP 不能直接解决 attention delta

attention delta 来自实际 live KV tokens 的分布差异。DeepEP 只替换 MoE dispatch/combine，不改变 attention 需要看的实际 KV length。

所以 DeepEP 后仍可能看到：

```text
KunServe bs 与 baseline bs 相同，但 KunServe total_kv_tokens 更高 -> attention 更慢
```

这个问题需要 KV-matched 分析或调度层控制，而不是 MoE communication library 本身解决。
