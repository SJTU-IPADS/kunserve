# Lane Combine 与本地 TP All-Reduce 融合准备文档

更新时间：2026-05-27

## 目标

当前 KunServe GLOBAL + fixed_padded CUDA graph 的主要瓶颈在 MoE combine 通信链路。最新 timing 显示，balloon 后每次 sampled replay 的 48 层累计约为：

- `graph_kunserve_dispatch_static_all_gather`: 4.5 ms
- `graph_kunserve_combine_static_lane_all_reduce`: 3.9 ms
- `graph_qwen3_moe_mlp_all_reduce`: 1.3 ms

本轮已经先把 static combine 的 lane all-reduce 改成更窄的 lane reduce-scatter。更激进的下一步是研究是否可以把 `dispatcher.combine` 里的 lane combine 和 Qwen3 MoE 后面的本地 TP all-reduce 融合成一个语义等价的 collective，进一步接近 SGLang 原生单实例 TP=4 的通信形态。

## 当前代码路径

关键代码：

- `sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`
  - `_dispatch_static`: fixed-padded graph capture 时走 lane `all_gather_into_tensor`。
  - `_combine_static`: fixed-padded graph capture 时做 lane combine。
- `sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
  - `FusedMoE.forward_impl`: `dispatch -> run_moe_core -> combine`。
  - 如果 `reduce_results` 为真，随后调用 `tensor_model_parallel_all_reduce`。
- `sglang/python/sglang/srt/models/qwen3_moe.py`
  - `Qwen3MoeSparseMoeBlock.forward_normal`: `self.experts(...)` 后，如果 TP>1 且未使用 reduce-scatter，会调用 `tensor_model_parallel_all_reduce`。
- `sglang/python/sglang/srt/distributed/communication_op.py`
  - `tensor_model_parallel_all_reduce` 最终走 `get_tp_group().all_reduce`，即 SGLang 原生 TP group registered collective。

## 例子：2 个实例，每个实例 TP=2，expert=128

设两个 replica：`R0`、`R1`。每个 replica 内两个 TP/EP lane：`L0`、`L1`。全局 rank 逻辑可以看作：

- `R0L0`: experts 0-31
- `R1L0`: experts 32-63
- `R0L1`: experts 64-95
- `R1L1`: experts 96-127

每个 replica 的 attention 后，`prepare_mlp` 已经做了本地 TP all-reduce，所以 `R0L0` 和 `R0L1` 都有 R0 当前 decode batch 的完整 hidden states，`R1L0` 和 `R1L1` 都有 R1 的完整 hidden states。

GLOBAL MoE 当前 static path：

1. Dispatch:
   - `L0` lane group: `R0L0 <-> R1L0` 做 all-gather，得到 `[R0 tokens, R1 tokens]`。
   - `L1` lane group: `R0L1 <-> R1L1` 做 all-gather，得到同样 union tokens。
   - 每个 rank 根据本 rank 持有的 expert mapping remap topk ids，只计算自己 expert 的贡献。

2. Expert core:
   - `R0L0` 只产出 experts 0-31 对 union tokens 的 partial。
   - `R1L0` 只产出 experts 32-63 对 union tokens 的 partial。
   - `R0L1` 只产出 experts 64-95 对 union tokens 的 partial。
   - `R1L1` 只产出 experts 96-127 对 union tokens 的 partial。

3. Combine:
   - lane combine 先在同 lane 的 replica 间求和并按 replica token chunk 返回。
   - 本地 TP all-reduce 再在同 replica 的 `L0/L1` 间求和，让每张卡都得到该 replica 自己 tokens 的完整 MoE 输出。

数学上，对 replica `r` 的 token `t`，目标输出是：

```text
Y[r, t] = sum_{lane in {0,1}} sum_{replica_shard in {0,1}} partial[replica_shard, lane, r, t]
```

当前两阶段通信是：

```text
P[lane, r, t] = lane_reduce_scatter(partial[:, lane, :, :])[r, t]
Y[r, t]       = local_tp_all_reduce(P[:, r, t])
```

## 本轮低风险优化

`_combine_static` 不再默认对 `[num_replicas * M, H]` 做 lane all-reduce 再 slice，而是默认使用：

```text
lane_group.reduce_scatter_tensor(local_slice[M, H], hidden_states[num_replicas*M, H])
```

这样每个 rank 只接收自己 replica 的 chunk，避免 lane all-reduce 把 peer replica 的 chunk 也复制回来。

保留回退：

```bash
KUNSERVE_STATIC_COMBINE_MODE=all_reduce
```

该回退只用于验证或规避特定 NCCL/driver 上 registered reduce-scatter replay 的问题。

同时，static CUDA graph path 现在会检查 group 是否提供 KunServe PyNccl/registered collective。如果 static path 发现会落到 raw `torch.distributed`，直接报错，而不是在 graph capture 中悄悄记录不可靠的 collective。

因为 static combine 开始使用 registered reduce-scatter，本轮也同步把 graph communicator preheat 扩展到 `reduce_scatter_tensor`。否则第一次 reduce-scatter 可能在 CUDA graph capture 内触发 NCCL lazy init，风险和之前 all-gather/all-reduce 未预热时相同。

## 为什么不能简单用一个 NCCL primitive 替代两阶段

理想语义是“对每个 replica chunk 做跨所有 expert shard 的 reduce，然后把结果交付给该 replica 内的每个 TP rank”。这不是标准 NCCL reduce-scatter 的原生语义：

- lane reduce-scatter 能把 `R0` chunk 给 `R0L0`，`R1` chunk 给 `R1L0`，但不会同时复制给同 replica 的另一个 lane。
- local TP all-reduce 能把 `L0/L1` partial 合并，但它发生在 lane reduce-scatter 之后。
- global all-reduce over `[num_replicas*M, H]` 然后 slice 是语义等价的单 collective，但会把所有 replica chunk 复制到所有 rank，带宽比当前 lane reduce-scatter 更差。

所以“一个标准 NCCL collective 完成全部语义”不现实。可行方向是实现一个自定义 registered collective 或把两个 collective 包成一个 graph-safe op。

## 可行实现路径

### 路径 A：registered composite op

新增一个 custom op，例如：

```text
kunserve_lane_reduce_scatter_then_tp_all_reduce(output, union_partial, lane_group, tp_group)
```

内部依次调用：

1. lane group registered reduce-scatter，输出 `[M, H]` scratch。
2. local TP group registered all-reduce，把 `L0/L1` partial 合成完整 `[M, H]`。

优点：

- 语义风险最低。
- 复用现有 KunServe PyNccl lane group 和 SGLang TP group。
- 对 Python 层和 CUDA graph 节点数有改善，容易加 timing 和回退。

限制：

- NCCL 层仍然是两个 collective，主要省 host/custom-op 组织成本，不会从根上减少通信轮数。
- 需要确认 custom op 内连续调用两个不同 group 的 registered collective 在 graph capture/replay 中稳定。

### 路径 B：自定义 P2P/reduce kernel

每个 rank 对 union partial 按目标 replica chunk 分块，把 chunk 发送到目标 replica 的两个 TP ranks，并在目标侧完成 reduce。

优点：

- 理论通信量最优：只传目标 replica chunk，不传无用 chunk。
- 可以把 lane combine 和 TP 合并成一次面向目标 replica 的 reduce+multicast。

限制：

- PyNccl 当前暴露的是 all-reduce/all-gather/reduce-scatter，并没有现成 send/recv/all-to-all registered surface。
- 需要新增底层 NCCL send/recv 或自定义 CUDA/NCCL 扩展，开发量明显高于路径 A。
- correctness 和 capture/replay 稳定性需要单独验证。

### 路径 C：全局 all-reduce 单 collective

把 combine 改成：

```text
global_all_reduce(union_partial[num_replicas*M, H])
slice local replica chunk
```

优点：

- 实现简单。
- 单 collective，语义直接。

缺点：

- 会把所有 replica 的 chunk 都送到所有 rank，带宽浪费明显。
- 对当前 2 replica 情况，基本回到 Phase D 风格，不符合优化目标。

该路径只适合作为 correctness 对照，不建议作为性能方案。

## 推荐路线

1. 先跑本轮 `lane reduce_scatter` static combine，确认输出正确，并比较：
   - `graph_kunserve_combine_static_lane_reduce_scatter`
   - `graph_kunserve_combine_static_lane_all_reduce`
   - `graph_qwen3_moe_mlp_all_reduce`

2. 如果 reduce-scatter 正确且收益稳定，下一步做路径 A：
   - 在 `parallel_state.py` 增加 registered composite custom op。
   - 在 `KunServePyNcclGroup` 或 dispatcher 中提供 composite 调用入口。
   - 在 `CrossReplicaStandardDispatcher._combine_static` 返回已经完成 TP all-reduce 的结果，并给上层一个标记，避免 `Qwen3MoeSparseMoeBlock.forward_normal` 或 `FusedMoE.forward_impl` 再做一次 TP all-reduce。

3. 路径 A 跑通后，再评估路径 B 是否值得做。只有当 composite op 的收益不够，且 timing 仍显示 combine 通信占主要比例时，才值得实现 P2P/reduce 级别的新 collective。

## 需要特别避免的错误

- 不能在 dispatcher combine 里做完 TP all-reduce 后，上层仍然再调用 `tensor_model_parallel_all_reduce`。这会双倍求和，之前乱码/数值爆炸类问题就容易从这里出现。
- 不能让 static CUDA graph path fallback 到 raw `torch.distributed`。
- 不能让某个 rank 单独进入 Phase E negotiate。所有 Phase E negotiation 必须是 lockstep collective；“只有本地 batch 变化时 negotiate”如果没有全局事件广播，会死锁。
- 不能把 top-k 变成 top-1，也不能改变 routing 权重 dtype/scale 语义。
