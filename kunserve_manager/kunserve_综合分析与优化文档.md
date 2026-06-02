# KunServe 综合分析与优化文档

最后更新：2026-06-01

本文是 KunServe 当前实现、性能结论和下一步优化计划的主文档。旧版中关于“lockstep 等待是最大瓶颈”“attention 因 KV cache 容量更大而变慢”等表述已删除；当前以 80B TP=4x4 的严格 Kineto + batch timing 结果为准。

## 1. 当前目标

KunServe 在 verl rollout + SGLang 后端中做 MoE expert balloon：

1. baseline：每个 SGLang 实例保持原生单实例 TP=4，expert 和 KV 都本地。
2. KunServe balloon：两个 SGLang 实例保持各自 attention/KV，本地释放一部分 expert physical pages 给 KV cache；被释放的 expert 由对端实例远程补齐。
3. 目标：降低 queue wait 和 retract，允许同样 GPU_MEMORY_UTILIZATION 下容纳更多长输出请求。

当前验证配置主要是 80B：

```text
模型：Qwen3_Next_80B_A3B_Thinking
baseline：两个独立 SGLang TP=4 实例
KunServe：两个 TP=4 实例进入 balloon 后共享 expert
本次严格 run：TRAIN_BATCH_SIZE=160，即每实例约 80 请求
```

## 2. 80B Balloon 数据面

以两个 TP=4 实例、128 experts 为例：

| rank | replica | lane/tp_rank | balloon 后保留 expert |
|---:|---:|---:|---|
| 0 | 0 | 0 | 本 lane 前半 |
| 1 | 0 | 1 | 本 lane 前半 |
| 2 | 0 | 2 | 本 lane 前半 |
| 3 | 0 | 3 | 本 lane 前半 |
| 4 | 1 | 0 | 本 lane 后半 |
| 5 | 1 | 1 | 本 lane 后半 |
| 6 | 1 | 2 | 本 lane 后半 |
| 7 | 1 | 3 | 本 lane 后半 |

每层 decode 的顺序：

```text
本 replica attention
  -> o_proj 后本实例 TP all-reduce
  -> prepare_mlp/router/topk
  -> KunServe lane dispatch all-gather
  -> expert id remap
  -> 本 rank 只算自己保留的 expert
  -> KunServe lane combine reduce-scatter
  -> 本实例 TP all-reduce 合并 lane partial
  -> 下一层
```

注意：attention/KV 不跨 replica 迁移。跨实例只发生在 MoE dispatch/combine。

## 3. 当前 GLOBAL CUDA Graph 策略

使用 `KUNSERVE_CAPTURE_POLICY=fixed_padded`：

1. balloon commit 后再 capture global graph，避免 pre-capture 指针和 VMM donor/KV remap 状态不一致。
2. decode 时按 bucket pad，例如 raw bs=73 replay graph bucket=80。
3. padding 行使用 dummy KV slot 和 phantom request，不参与真实输出。
4. static dispatcher 使用 max-capacity persistent buffers，保证 CUDA graph storage 地址稳定。

这次已实现的关键修复：

```text
旧：static dispatch/combine 的 NCCL view 长度 = max_capture_m
新：static dispatch/combine 的 NCCL view 长度 = 当前 graph_bucket_m
```

代码位置：

```text
/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py
```

含义：

| 项 | 旧行为 | 新行为 |
|---|---|---|
| backing buffer | `max_capture_m` 分配 | 不变 |
| NCCL 输入输出 tensor view | 始终最大 | 当前 graph bucket |
| 小 bucket 是否发送最大 padding | 是 | 否 |
| CUDA graph 地址稳定性 | 稳定 | 仍稳定，因为 view 指向同一 storage 起点 |

本次还补了 per-bucket log：每个 `graph_bucket_m` 第一次 capture 时会记录 dispatch/combine shape，避免日志只显示最大 bucket。

## 4. bs80 实测结果

输出目录：

```text
/workspace/verl/outputs/prof80b_bs80_bucketfix_20260601_152327
```

配置：

```text
TRAIN_BATCH_SIZE=160
GPU_MEMORY_UTILIZATION=0.55
SGLANG_MEM_FRACTION_STATIC=0.55
MAX_RUNNING_REQUESTS=160
KUNSERVE_PHASE_G=0
KUNSERVE_CAPTURE_POLICY=fixed_padded
KUNSERVE_MOE_A2A_BACKEND=none
KUNSERVE_MOE_RUNNER_BACKEND=triton
PROFILE_NUM_STEPS=40
```

端到端结果：

| 指标 | KunServe | baseline |
|---|---:|---:|
| requests | 160 | 160 |
| queue mean | 8.0s | 224.3s |
| queue p50 | 4.5ms | 200.3s |
| queue p99 | 185.0s | 600.9s |
| e2e mean | 453.2s | 496.1s |
| e2e p50 | 438.4s | 458.1s |
| e2e p99 | 717.1s | 909.1s |

stage 对照均值：

| 贡献项 | mean ms/step | 当前解释 |
|---|---:|---|
| `attention_delta` | 3.284 | profile 到的 KunServe bucket KV 更高；不是容量本身导致同长度更慢 |
| `fixed_cross_replica` | 2.890 | KunServe 架构额外成本，来自 dispatch/combine/remap |
| `lockstep_wait` | 1.524 | 存在尖峰，但均值不是第一项 |
| `expert_delta` | -0.791 | expert 平均抵消部分开销 |
| `real_iter_delta` | 8.956 | batch timing 非 profiled 真实每步中位差 |

这次优化是否有效：

| 指标 | 修复前 bs96 | 修复后 bs80 |
|---|---:|---:|
| `normal_interinst` | 约 6.725ms | 约 3.105ms |
| 平均 fixed cross-replica | 约 6.617ms | 约 2.890ms |

结论：`max_capture_m -> graph_bucket_m` 这部分确定性浪费已经解决。剩余开销不再主要是“所有小 bucket 都按最大 M 发送”，而是：

1. 每层仍有跨实例 MoE collective 的固定启动和传输成本。
2. 部分 bucket 仍有 replica lockstep wait。
3. KunServe profile 到的 live KV 分布比 baseline 更长，导致 attention 时间更高。

## 5. Attention Delta 的正确边界

不能说“KV cache 容量更大，所以计算同样长度 KV 更难”。正确边界是：

```text
attention 时间取决于实际参与 attention 的 live KV tokens / sequence length，
不取决于预留 capacity。
```

如果同一个请求在 KunServe 和 baseline 中有相同 prompt length、相同 generated length、同一层同一 batch 形态，那么 attention kernel 不应该因为 KunServe 的 KV cache capacity 更大而天然更慢。

本次看到的 attention delta 来自测量分布：

| KunServe bs | KunServe kvK | baseline bs | baseline kvK | attention delta |
|---:|---:|---:|---:|---:|
| 62 | 492 | 62 | 187 | +4.742ms |
| 55 | 499 | 53 | 186 | +4.963ms |
| 38 | 459 | 39 | 186 | +3.375ms |

这说明 KunServe balloon 后保留了更多长上下文请求；baseline 同 bs bucket 可能来自较早或不同队列阶段。下一步如果要判定 attention 是否需要优化，应做 KV-matched join 或 KV slope regression。

## 6. 当前瓶颈判断

按“观察到的 wall-clock 差值”看，bs80 这轮最大的正项是 attention delta。

按“KunServe 架构带来的可优化额外成本”看，优先级是：

1. 跨实例 MoE dispatch/combine。修复 max bucket padding 后仍有约 2.9ms fixed cross-replica 和约 1.5ms lockstep wait。
2. lockstep wait。需要降低两个 replica 在 collective 入口的步调差，尤其长尾 bucket。
3. attention delta。先做 KV 控制分析，避免优化错方向。
4. expert。当前不是瓶颈。

旧结论修正：

| 旧说法 | 当前修正 |
|---|---|
| lockstep 是最大瓶颈 | lockstep 存在但不是稳定最大项；修复 padding 后均值约 1.5ms |
| attention 是因为 KV capacity 大而慢 | 错。是 actual live KV tokens 分布不同 |
| 只要砍掉 negotiate 就能接近 TP=4 | 错。negotiate 已经不是主项，跨实例 MoE collective 和 KV 分布仍在 |
| small bucket 固定发送 max_capture_m 是主要浪费 | 这次已经修复，且实测下降明显 |

## 7. DeepEP 预期

DeepEP 不在本次代码实现范围内；这里只写预期。

当前 KunServe dense lane communication 的特点：

```text
每层每 lane：
  dispatch all-gather hidden/topk/topk_weight
  remap
  expert
  combine reduce-scatter
```

DeepEP 的目标是把 dense lane exchange 改成 expert/token routed exchange：

1. 只发真正需要远端 expert 的 token/slot。
2. 避免 padded/zero/无效 expert slot 进入跨实例通信。
3. 使用面向 MoE dispatch/combine 的 GPU-side permutation、count、scatter/gather。
4. 减少每层固定 collective 启动开销和同步面。

基于 bs80 当前数据的保守估计：

| 项 | 当前 | DeepEP 可能减少 |
|---|---:|---:|
| fixed cross-replica | 2.89ms/step | 50%-70%，约 1.4-2.0ms/step |
| lockstep wait | 1.52ms/step | 取决于实现，可能额外减少 0.3-0.8ms/step |
| attention delta | 3.28ms/step | DeepEP 不直接解决 |

如果 global decode 平均 step 从 33-38ms 降低约 2-3ms，当前 bs80 KunServe 的 ideal/e2e 可能下降约 20-40s。粗略区间：

```text
当前 KunServe e2e mean：453s
仅 DeepEP comm 改善后的合理目标：约 425-435s
更乐观、且 lockstep 同时下降：约 415-425s
```

这个估计不包含 attention KV-matched 优化，也不假设输出长度变化。

DeepEP 细节见：

```text
/workspace/sglang/kunserve_manager/deepep_vs_kunserve通信原理与bs80预期.md
```

## 8. 复现和分析链路

性能分析方法和当前结果见：

```text
/workspace/verl/data/kunserve_性能分析链路.md
```

bs96 旧实验命令记录见：

```text
/workspace/verl/data/prof80b_bs96_20260601_134532_复现实验命令.md
```

本次 bs80 的主要产物：

```text
/workspace/verl/outputs/prof80b_bs80_bucketfix_20260601_152327/prof_kunserve/stages_kunserve.csv
/workspace/verl/outputs/prof80b_bs80_bucketfix_20260601_152327/prof_baseline/stages_baseline.csv
/workspace/verl/outputs/prof80b_bs80_bucketfix_20260601_152327/stage_compare_strict.csv
/workspace/verl/outputs/prof80b_bs80_bucketfix_20260601_152327/kunserve/sglang_batch_timing.jsonl
/workspace/verl/outputs/prof80b_bs80_bucketfix_20260601_152327/baseline/sglang_batch_timing.jsonl
```

## 9. 下一步计划

1. 做 KV-matched attention 分析：按 `total_kv_tokens` 或 `avg_kv_per_req` join baseline/KunServe。
2. 为 DeepEP 版本准备最小可替换接口：dispatch input、expert-local layout、combine output 语义要与当前 `CrossReplicaStandardDispatcher` 对齐。
3. 继续记录 per-bucket `graph_bucket_m`，确认后续小 bucket 不回退到 max-capacity view。
4. 在 DeepEP 之前，不再优先做 Phase G 或 pre-capture；当前 post-commit capture 速度可以接受，瓶颈不在 capture 本身。
