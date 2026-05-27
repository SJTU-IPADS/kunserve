# Cross Replica Standard-Like Dispatcher 方案综述

更新时间：2026-05-27

本文描述当前 KunServe 跨 replica MoE dispatcher 方案。旧版中关于 DeepEP 作为主路径、GLOBAL graph 不可用、Phase E 尚未实现、Phase G 默认开启等内容已经删除。

## 1. 当前结论

当前主线：

- 两个 SGLang replica，每个 `TP=2`。
- BALLOON 后每张卡保留 32 个 experts。
- 不启用 Phase G。
- 不用 DeepEP。
- dispatcher 使用 Standard-like static path。
- 跨 replica 通信使用 KunServe PyNccl registered collective。
- CUDA graph 使用 `fixed_padded`，在 `commit_balloon` 后 capture。
- Phase E keepalive + cached negotiate 保障最后一个请求和 asymmetric drain。

## 2. 固定例子：TP=2 双 replica，128 experts

全局 4 rank：

| global rank | replica | TP lane | 保留 physical experts |
| --- | --- | --- | --- |
| 0 | r0 | 0 | 0..31 |
| 1 | r0 | 1 | 64..95 |
| 2 | r1 | 0 | 32..63 |
| 3 | r1 | 1 | 96..127 |

为什么这样分：

- 每个 replica 原本在 `TP=2/EP=2` 下，本地每个 TP lane 负责半个 expert 空间。
- BALLOON 后每个 lane 保留自己原空间的一半，另一半由另一个 replica 同 lane 补齐。
- lane 0 合起来覆盖 0..63。
- lane 1 合起来覆盖 64..127。

## 3. 一层 forward 数据流

以 decode step 为例：

1. Attention 和 `o_proj` 在本 replica 内执行。
2. `layer_communicator.prepare_mlp` 触发 SGLang 原生 local TP all-reduce。
3. 此时每个 TP lane 都有本 replica 所有请求的 hidden state。
4. KunServe dispatcher 在同 lane 跨 replica 交换 hidden/topk。
5. 每个 rank 根据 physical expert id 过滤出本 rank 保留的 experts。
6. Triton MoE runner 计算本 rank compact local expert。
7. lane 内 combine。
8. 只保留本 replica 请求对应的输出。
9. 返回 FusedMoE 外层后，本 replica local TP all-reduce 合并 lane 0/1 partial 输出。

关键点：另一个 replica 只帮忙算 expert，不拥有这些请求的 KV cache，也不会把这些请求送入自己的 attention。

## 4. dispatcher 数学

输入：

```text
hidden_states: [local_tokens, hidden]
topk_ids:      [local_tokens, top_k]
topk_weights:  [local_tokens, top_k]
```

每个 lane group 内 all-gather：

```text
global_hidden = concat(hidden_from_r0_lane, hidden_from_r1_lane)
global_topk   = concat(topk_from_r0_lane, topk_from_r1_lane)
```

本 rank 按 retained physical expert set 过滤：

```text
if expert_id in retained_set:
    local_expert_id = compact_mapping[expert_id]
else:
    local_expert_id = -1
```

MoE runner 只计算 `local_expert_id >= 0` 的项。combine 后 lane group 聚合 partial result，再 slice 回本 replica 本地 token 范围。

当前 combine 为 all-reduce + slice。它数学上等价于 reduce-scatter，但通信量更大；这是为了先稳定 GLOBAL graph 正确性。后续可以恢复 graph-safe reduce-scatter。

## 5. 为什么 lane 通信必要

用户之前提出的思路是正确的：attention 后本实例每张卡已有本实例完整 hidden state，因此跨 replica 只需要同 lane 通信。

如果改成 4 rank 全互联 all-gather：

- 会多交换不需要的 lane 数据；
- 会打破 “lane 0 只负责 0..63，lane 1 只负责 64..127” 的 expert 空间划分；
- local TP all-reduce 的语义也会变得不清晰。

因此当前 dispatcher 保持 lane group：

- lane 0：r0 rank0 与 r1 rank2。
- lane 1：r0 rank1 与 r1 rank3。

## 6. CUDA graph 约束

GLOBAL graph 要求：

- 每个 graph bucket 的 tensor shape 固定。
- 每个 lane rank collective 顺序一致。
- 每个 lane rank collective shape 一致。
- graph capture 后 pointer 拓扑稳定。

因此当前需要：

- post-commit capture，不能 pre-commit capture。
- `fixed_padded` bucket。
- padding 行写 dummy KV slot。
- Phase E negotiate 让所有 rank 同意本 step graph bucket。
- idle keepalive 让已 drain replica 继续参与 collective。

## 7. Phase E 与 dispatcher 的关系

Phase E 不改变 dispatcher 数学。它只负责每一步进入 dispatcher 前的 shape/participation 协议。

每个 step 决定：

```text
max_bs
min_bs
any_force_eager
graph_bs_override
```

情况：

- 所有 replica batch 一样：直接 replay 对应 bucket。
- busy/busy 但 bucket 不同：所有 rank replay `max_bs` bucket，小 batch padding。
- idle/busy：idle rank 构造 `ForwardMode.IDLE` keepalive，target bs 为 `max_bs`。
- 任一 rank 是 EXTEND/mixed：所有 rank force eager。
- 所有 rank idle：停止 keepalive，不 run batch。

## 8. cached negotiate 策略

每步 negotiate 太慢，因此当前用短窗口 cache。

默认：

```bash
KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=16
```

refresh step：

- collective 交换 local padded bs 和 force-eager flag；
- 更新 cache；
- 决定是否 keepalive、override graph bucket 或 eager。

cache step：

- 复用 cached max/min/force-eager；
- 小 batch pad 到 cached max bucket；
- 如果本地增长超过 cached max，立即 reset cache；
- 如果本地出现 prefill，推迟到下一次 refresh。

这保留了 “batch 变化需要重新 negotiate” 的安全性，但承认一个事实：peer shrink 在没有通信时不可见，所以只能在 refresh step 观察。这个 tradeoff 的收益是大幅降低 scheduler gap。

## 9. 2026-05-27 release cleanup

`ab_20260527_020500` 暴露了一个收尾问题：

- 所有真实请求结束。
- Phase E cache 仍在窗口末尾复用 `cached_max_bs=1`。
- overlap `result_queue` 留下一个 IDLE keepalive result。
- `release_memory_occupation` 先于下一次 pop/process 执行。
- SGLang 原生 `_is_no_request()` 看到 result queue 非空，断言失败。

修复：

- release 断言前 drain overlap result queue。
- stop keepalive。
- reset Phase E cache。
- 清理 per-step force eager 和 graph bucket override。
- 继续使用原生断言保护真实 running request。

这不是放宽正确性检查，而是把 KunServe 自己产生的 harmless IDLE residue 在 release 前收干净。

## 10. 当前状态

已落地：

- 正确 expert layout。
- lane subgroup。
- KunServe PyNccl registered collective。
- post-commit GLOBAL graph capture。
- fixed padded replay。
- dummy KV slot 和 phantom req_pool row。
- Phase E keepalive。
- mismatched decode 用 graph bucket override，不再直接 eager。
- cached negotiate。
- release memory cleanup。

仍需优化：

- all-reduce + slice 改成 graph-safe reduce-scatter。
- 继续细分 model.forward 内部 timing。
- 减少 Python scheduler gap。
- 对比原生单实例 TP=4 的 attention/MoE/TP all-reduce 各阶段耗时。
- restore path 不是当前主线。

## 11. 正确性 checklist

每次改 dispatcher 或 Phase E 后检查：

- BALLOON 后 prompt answer streaming 仍 coherent。
- `cuda_graph_replay_selected` 中 raw bs 与 graph bs 符合预期。
- `phase_e_negotiated` 的 `source=cache` 与 `source=collective` 比例合理。
- asymmetric drain 时出现 keepalive，但最终不会 hang。
- rollout 结束 release 不再触发 `_is_no_request` assertion。
- experts 32..63 和 96..127 没有被错误映射到 0..31。
