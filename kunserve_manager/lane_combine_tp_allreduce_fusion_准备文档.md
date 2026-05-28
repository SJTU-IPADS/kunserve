# Global MoE 通信链路优化准备文档（路线 A 落地 + 路线 B 方案）

更新时间：2026-05-28

> 历史名：`Lane Combine 与本地 TP All-Reduce 融合准备文档`。路线 A（lane combine + TP all-reduce 复合 op）已落地，结论是 CUDA graph 下无可见正收益但代码作为对照基线保留；本次文档把范围扩展到整条 GLOBAL MoE 通信链路（dispatch all-gather + combine + TP all-reduce），并提出路线 B 的三种子方案。

## 一、当前状态总览（2026-05-28）

| 阶段 | 状态 | 关键产物 |
|---|---|---|
| Phase D（fixed-padded 静态 buffer + all-gather） | ✅ Implemented | `_dispatch_static`、`_buf_gathered_*` |
| Phase F（lane subgroup + reduce-scatter combine） | ✅ Implemented | `_combine_static` + `lane_group.reduce_scatter_tensor` |
| **路线 A**（registered composite op：lane reduce-scatter + TP all-reduce） | ✅ Implemented，env-gated，默认关 | `kunserve_lane_reduce_scatter_then_tp_all_reduce` |
| 路线 B（自定义 P2P + on-device reduce） | 📝 准备阶段 | 本文档第六节 |
| Phase E（idle keepalive） | ⏳ Not done | 后续工作 |

## 二、最新瓶颈拆解（post-balloon 单次 sampled replay，48 层 MoE）

数据来源：`/workspace/verl/outputs/ab_20260528_041612/kunserve/kunserve_forward_timing.jsonl`（融合 env off，干净基线）。

| stage | 单层均值 | 单次 replay 累计 | 占 cuda_graph_replay_launch 比例 |
|---|---|---|---|
| `graph_kunserve_dispatch_static_all_gather` | 0.090 ms | ~4.3 ms | 38% |
| `graph_kunserve_combine_static_lane_reduce_scatter` | 0.065 ms | ~3.1 ms | 27% |
| `graph_qwen3_moe_mlp_all_reduce` (local TP) | 0.025 ms | ~1.2 ms | 11% |
| 其余 dispatch_static_pad/remap/slice | <0.025 ms each | ~1.0 ms | ~9% |
| **小计：MoE 跨实例通信** | — | **~8.6 ms** | **~75%** |
| `cuda_graph_replay_launch` 总 | 11.453 ms | 11.4 ms | 100% |

**结论：dispatch 已经反超 combine 成为最大通信成本**。如果路线 B 只动 combine 最多省 ~1.5 ms（按减半估算），合计仅 ~13%；同时改 dispatch 才能把 MoE 通信压到 ~50% 以下。

## 三、例子：2 实例 × TP=EP=2，expert=128

- `R0L0`: experts 0–31
- `R1L0`: experts 32–63
- `R0L1`: experts 64–95
- `R1L1`: experts 96–127

attention 后 `prepare_mlp` 已经做了本地 TP all-reduce，所以每个 replica 内 `L0`/`L1` 都拥有该 replica 完整的 hidden states。

GLOBAL MoE 当前 static path（Phase D + F）：

1. **Dispatch**（lane all-gather）
   - `L0` lane subgroup（`R0L0` ↔ `R1L0`）all-gather hidden + topk → 每个 rank 拿到 `[union_M, H]`（union = R0 tokens ∪ R1 tokens）。
   - `L1` lane subgroup 同样。每个 rank 用本地 expert mapping remap topk ids，只算自己 shard 的 expert。
2. **Expert core**：每个 rank 对 union tokens 计算自己持有 expert 的 partial。
3. **Combine**（lane reduce-scatter + local TP all-reduce）
   - lane reduce-scatter：跨 replica shard 求和，并按 replica chunk 切给本 rank → `[M, H]`。
   - local TP all-reduce：在同 replica 的 `L0`/`L1` 间求和（在 `Qwen3MoeSparseMoeBlock.forward_normal` 里）。

数学上对 replica `r` 的 token `t`：

```text
Y[r, t] = sum_{lane∈{0,1}} sum_{replica_shard∈{0,1}} partial[replica_shard, lane, r, t]
       = local_tp_all_reduce_over_lanes( lane_reduce_scatter( partial[:, lane, :, :] ) )
```

## 四、路线 A 落地总结

### 4.1 实现位置

- `python/sglang/srt/distributed/parallel_state.py`
  - 新增 `@register_custom_op` 的 `kunserve_lane_reduce_scatter_then_tp_all_reduce(output, input, lane_group_name, tp_group_name)`，内部按序调用 `lane_group._reduce_scatter_tensor(output, input)` 然后 `tp_group._all_reduce_in_place(output)`。
- `python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`
  - 新增 mode `reduce_scatter_tp_all_reduce`；
  - `set_static_tp_allreduce_fusion_enabled(bool)` 在 `Qwen3MoeSparseMoeBlock.forward_normal` 进入 experts 前由调用方按层开关；
  - 命中条件：mode 是 `reduce_scatter_tp_all_reduce` **且** runtime gate 打开 **且** `local_tp_group` 是 SGLang GroupCoordinator（含 `_all_reduce_in_place`），否则报错而不是降级到 raw torch.distributed；
  - 命中后在返回 result 上打 `_kunserve_tp_allreduce_done=True`，并跳过下游 TP all-reduce。
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
  - 在 slice/contiguous 后把 `_kunserve_tp_allreduce_done` 标志传递回去。
- `python/sglang/srt/models/qwen3_moe.py`
  - 调 `set_static_tp_allreduce_fusion_enabled(allow_kunserve_tp_allreduce_fusion)`；
  - `_kunserve_tp_allreduce_done=True` 时跳过 `tensor_model_parallel_all_reduce`，避免双 reduce。
- `python/sglang/srt/model_executor/model_runner.py`
  - 把 Phase F 的 local TP group 与 NCCL preheat 都改为传 GroupCoordinator（之前传的是 `device_group`），以匹配 composite op 对 `_all_reduce_in_place` 的需求。

### 4.2 触发方式（env gate）

```bash
# 写法 1：直接指定 mode
KUNSERVE_STATIC_COMBINE_MODE=reduce_scatter_tp_all_reduce

# 写法 2：在 reduce_scatter 模式上额外开启 fusion
KUNSERVE_STATIC_COMBINE_MODE=reduce_scatter   # 默认值
KUNSERVE_STATIC_COMBINE_TP_ALLREDUCE_FUSION=1
```

不设这两个变量 → 沿用 Phase F 的 `reduce_scatter` 模式（041612 验证幂等）。

### 4.3 实测结果

横向对比（每层均值；post-balloon）：

| run | combine 模式 | combine/层 | qwen3_mlp_all_reduce/层 | combine + tp_ar 合计/层 | cuda_graph_replay_launch |
|---|---|---|---|---|---|
| ab_20260527_060711（Phase F 早期） | static_lane_all_reduce | 0.082 ms | 0.025 ms | 0.107 ms | 11.295 ms |
| ab_20260527_071331（Phase F lane reduce_scatter） | static_lane_reduce_scatter | 0.065 ms | 0.025 ms | 0.090 ms | 11.297 ms |
| ab_20260528_041612（路线 A 代码在 + env off） | static_lane_reduce_scatter | 0.065 ms | 0.025 ms | 0.090 ms | 11.453 ms |
| ab_20260527_104214（路线 A env on） | static_lane_rs_tp_ar | 0.110 ms | 0 (跳过) | 0.110 ms | 22.174 ms |

观察：

1. 路线 A 命中后 `qwen3_moe_mlp_all_reduce` 调用次数确实下降（detail log 显示 `mode=reduce_scatter_tp_all_reduce phase_f=True`，timing CSV 上 fusion 路径 n=34752、qwen3_mlp_all_reduce n=25728，两者互斥分布在静态/非静态路径）。
2. 单层 wall-clock：融合后 0.110 ms ≈ 融合前 0.090 ms，**没有正收益甚至略差**。
3. 104214 上 `cuda_graph_replay_launch` 翻倍（11.3 → 22.2 ms）经 041612 对照确认与融合无关，怀疑是 internal timing 注入 + post-balloon 比例差异，需要单独追查。

### 4.4 为什么路线 A 无可见收益

- 路线 A **不减少 NCCL collective 数**：仍然是 `ncclReduceScatter` + `ncclAllReduce` 两条 op，只是包在一个 `register_custom_op` 里。
- CUDA graph 已经把 host launch 顺序记录下来，replay 阶段两条 op 仍然依次启动，没有 wire-level 合并。
- 路线 A 节省的只是**捕获时**的 host 侧组织成本，replay 路径上 ~ 0。
- 路线 A 把 0.025 ms 的 TP all-reduce 时间从 qwen3 那层挪进 combine scope 里，所以 combine stage 看似变贵，但合计基本持平（差异 ~0.02 ms 多半是 custom op dispatch 自身开销）。

### 4.5 保留还是 revert

**保留路线 A**：

- 代码完全 env-gated，默认关，041612 实测幂等无回归。
- 当成"两 collective 顺序绑定"的对照基线，路线 B 实现后可以直接比较"两条 NCCL 紧挨着" vs "一个真正的 fused 内核"的差。
- 不需要 git reset；commit 时建议消息标注"路线 A 落地，默认关，作为路线 B 对照"。

## 五、为什么单条标准 NCCL 不能等价两阶段

理想语义："对每个 replica chunk 做跨所有 expert shard 的 reduce，并把结果交付给该 replica 内的每个 TP rank"。这不属于任何一条 NCCL primitive：

- lane reduce-scatter 能把 `R0` chunk 给 `R0L0`、`R1` chunk 给 `R1L0`，但**不会同时复制给同 replica 的另一个 lane**。
- local TP all-reduce 能把 `L0/L1` partial 合并，但发生在 lane reduce-scatter 之后。
- global all-reduce over `[num_replicas*M, H]` 然后 slice 是语义等价的单 collective，但带宽更差（每张卡都收完整 union）。

dispatch 侧同理：

- 现在的 lane all-gather 给每个 rank 都送来 union（`[2M, H]`），但 expert shard 的实际"工作负载"只覆盖 union 中由本 rank 持有 expert 的子集 tokens。
- 通信意义上拉来的 `(1-1/E_local_ratio)*union*H` 数据其实只用于 topk routing 决策，并未参与到 expert kernel 输入侧。

→ 真正减带宽的路只能是一条**自定义复合 collective**，把 dispatch 和 combine 各自重塑成"每个 rank 只收/发自己实际要用的那部分"。

## 六、路线 B：方案分析

### 6.1 目标语义

新设计同时优化 dispatch 和 combine，整体形如：

```text
dispatch_lane_p2p_gather(local_hidden[M,H], topk[M,K]) ->
    union_hidden[union_M, H], union_topk[union_M, K]      # 每个 rank 只拿自己 expert 真正命中的 tokens

experts(union_hidden, union_topk) -> union_partial[union_M, H]

combine_p2p_reduce_to_replica(union_partial[union_M,H]) ->
    Y[M, H]                                                # 已经合并完成 lane 求和 + replica 内 TP 求和
```

理想带宽（相对当前）：

- dispatch：从 `2M*H * lane_world (=2)` 降到 `~M*H * (1 + skip_ratio)`，预期 30%–50% 减半。
- combine：从 `2M*H * lane_world (=2) → reduce_scatter→ M*H → local_tp_ar → 2M*H` 降到一次 `~M*H` 量级传输，预期 50%+ 减半。

### 6.2 子方案 B1：sgl-kernel AOT（NCCL send/recv + on-device reduce kernel）

- **核心**：在 `sgl-kernel` 里新增 AOT CUDA + NCCL kernel，自己管 `ncclSend`/`ncclRecv`/`ncclGroupStart`/`ncclGroupEnd`，并在接收侧用一个轻量的 SM-side reduce kernel 把多个 partial 求和。
- **可行性**：`device_communicators/pynccl_wrapper.py` 已经暴露 `ncclSend`/`ncclRecv`/`ncclGroupStart`/`ncclGroupEnd`（line 491–553）。`PyNcclCommunicator.send/recv`（line 317、334）也已经在 Python 层挂好。
- **graph 安全性**：NCCL `ncclGroupStart/End` 与 `ncclSend/ncclRecv` 在 capture stream 上是 graph-safe 的（NCCL ≥ 2.18 + cudaGraph capture 支持已经成熟）。但需要在 KunServePyNccl 上加一个 registered surface（参考 `_reduce_scatter_tensor` 的写法）。
- **工程量**：中。需要的工作：
  1. `sgl-kernel/csrc/kunserve_moe_comm/` 新建一个 .cu + .h + bind.cpp 实现 `kunserve_p2p_reduce_to_replica` SM-side reduce kernel（输入：union_partial、目标 chunk metadata；输出：本 rank 应得的 chunk）。
  2. `device_communicators/pynccl_wrapper.py` 已有的 send/recv 上面包一层 graph-safe registered op。
  3. `kunserve_pynccl.py` 加 `_send_recv_grouped(...)` registered method，作为 `register_custom_op` 的目标。
  4. `parallel_state.py` 注册 `kunserve_dispatch_p2p` 和 `kunserve_combine_p2p_reduce` 两个复合 op。
  5. `kunserve_standard.py` 在 `_dispatch_static` / `_combine_static` 增加 mode `p2p_lane` / `p2p_replica_reduce`，env gate `KUNSERVE_ROUTE_B_DISPATCH`、`KUNSERVE_ROUTE_B_COMBINE`，默认关。
- **预期收益**：dispatch + combine 合计减半 ≈ 省 ~3.5–4.5 ms 单次 replay，把 MoE 通信占比从 75% 压到 ~40%。
- **风险**：
  - SM-side reduce kernel correctness 需要单元测试 + 与现行 lane reduce-scatter 数值 bit-diff 对照。
  - NCCL send/recv 与 cudaGraph capture 在某些 NCCL 版本/驱动组合上有过坑（如 ncclGroup 内非对称 send/recv 计数 → hang），preheat 必须覆盖 group + send/recv pair。
  - 需要新增 AOT 编译依赖，CI/构建链路要确认。

### 6.3 子方案 B2：纯 Python 上 PyNccl send/recv（无 sgl-kernel 改动）

- **核心**：不写新 CUDA kernel，直接在 `kunserve_standard.py` 用 `PyNcclCommunicator.send/recv` + `group_start/group_end` 实现 P2P 路径，目标 chunk 的 reduce 用 PyTorch 原生 `+=`/`scatter_add_` 完成。
- **可行性**：API 全部就绪；不需要碰 sgl-kernel。
- **graph 安全性**：需要把 send/recv 也封装成 `register_custom_op`（参考 `kunserve_lane_reduce_scatter_then_tp_all_reduce` 的实现）。
- **工程量**：低-中。可作为 B1 之前的 prototype。
- **预期收益**：
  - dispatch 端通信量减少与 B1 一致；
  - combine 端 reduce 步骤改用 cuBLAS/torch elementwise，可能比 SM-side reduce 慢一点（多一次显存往返），但仍然比 reduce_scatter + all_reduce 两阶段少一条 NCCL collective。
- **风险**：
  - PyTorch `+=` 在 capture 内必须落到具体 stream 上，否则 stream 选择错可能 capture 失败。
  - PyNccl send/recv 的 graph 注册路径之前没用过（只用过 all_reduce/all_gather/reduce_scatter），先做 preheat 探针验证。
- **建议定位**：先用 B2 跑通正确性 + 量化收益，**再决定是否升级到 B1 的 SM-side reduce kernel**。

### 6.4 子方案 B3：CUDA IPC + 直接 SM-side P2P+reduce（custom_allreduce 风格）

- **核心**：完全绕开 NCCL，参考 `sgl-kernel/csrc/allreduce/custom_all_reduce.{cu,cuh}` 的做法，开 CUDA IPC handle，远端 partial 直接 mapped 到本卡地址空间，本卡用 SM 加载远端数据并 in-register 求和写回目标 chunk。
- **可行性**：单机 NVLink 拓扑下成熟，custom_allreduce 已经在 sglang 里用过；但 KunServe 的两个 replica 在 verl 进程模型里是**分属两个 process group 的 4 张卡**，需要先确认它们共享同一 CUDA context root 或者能够走 `cudaIpcOpenMemHandle`。
- **工程量**：高。优势是理论延迟最低（一次 SM-loaded reduce），劣势是与 verl 的 process 模型耦合很深。
- **风险**：
  - CUDA IPC 在容器/不同 cgroup namespace 下可能拿不到 handle，需要在部署侧确认。
  - 与 `ExpandableVmmTensor` 的 VMM 段交互复杂：partial 缓冲必须落在可 IPC export 的物理段上，跟 balloon 的 VMM 段管理冲突风险高。
  - **不建议作为第一优先**，仅在 B1 收益不达预期时再评估。

### 6.5 路线 B 子方案对比

| 维度 | B1 (sgl-kernel AOT) | B2 (PyNccl send/recv) | B3 (CUDA IPC) |
|---|---|---|---|
| 工程量 | 中 | 低-中 | 高 |
| 数值风险 | 中（自写 reduce kernel） | 低（用 torch op） | 高（指针/对齐） |
| 通信轮数 | 1 次 ncclGroup（send/recv 内嵌） | 1 次 ncclGroup | 0 NCCL，纯 SM load |
| reduce 位置 | SM-side fused | torch elementwise | SM-side in-register |
| graph 兼容性 | 已知支持 NCCL group capture | 需先验证 send/recv graph 注册 | 已被 custom_allreduce 验证 |
| 预期 dispatch+combine 节省 | ~50% | ~30%–45% | ~50%–60% |
| 失败回退 | env off → 走路线 A 或 Phase F | env off → 走路线 A 或 Phase F | 同上 |

## 七、推荐路线

1. **阶段 1（1–2 天）**：路线 A 当前修改打 commit，标注"默认关，作为对照"。补 `KUNSERVE_GRAPH_INTERNAL_TIMING=0` 的对照 run，确认 `cuda_graph_replay_launch` 11.3 ms 基线稳定，排除 timing 注入带来的扰动。
2. **阶段 2（3–5 天，B2 prototype）**：
   - 在 `kunserve_standard.py` 加 `KUNSERVE_ROUTE_B_DISPATCH=1` / `KUNSERVE_ROUTE_B_COMBINE=1` 两个开关，分别独立验证；
   - 用 `PyNcclCommunicator.send/recv` 写出 P2P 版本的 `_dispatch_static_p2p` / `_combine_static_p2p`；
   - `parallel_state.py` 注册 `kunserve_dispatch_p2p_lane` / `kunserve_combine_p2p_replica_reduce` 两个 custom op；
   - 增加 preheat：在 graph capture 之前做一次 send/recv pair 对所有 lane group 预热（必须！否则首次 capture 中 NCCL lazy init 会破坏 graph）。
   - 正确性验证：与当前 reduce_scatter 路径做 streaming dump tail 对照，分别跑 5K + 25K decode（参考 [[finding_fp32_lane_reduce]] 的发现，注意 bf16 漂移阈值）。
   - 收益验证：直接看 `graph_kunserve_dispatch_static_*` 和 `graph_kunserve_combine_static_*` 是否实质下降；最终看 `cuda_graph_replay_launch`。
3. **阶段 3（视 B2 收益决定）**：
   - 若 B2 收益≥ 30% 且 numerics 稳定 → 进 B1（把 reduce 步骤换成 sgl-kernel SM kernel）压最后一刀；
   - 若 B2 收益不足 30% → 直接停在 B2 或考虑 B3。
4. **阶段 4**：路线 B 稳定后，把 Phase E（idle keepalive）做掉，避免两 replica workload 不对称时的 capture/replay 不一致。

## 八、必须遵守的红线

- **不能在 dispatcher combine 里做完 TP all-reduce 后上层仍调用 `tensor_model_parallel_all_reduce`**——路线 A 已经踩过这条线，路线 B 实现时必须沿用 `_kunserve_tp_allreduce_done` 标志机制（或等价信号）。
- **不能让 static CUDA graph path fallback 到 raw `torch.distributed`**——必须要么走 KunServePyNccl 注册路径，要么 hard fail。
- **任何新 NCCL 通信模式必须 preheat**：路线 B 增加 send/recv 后，`ModelRunner.preheat_for_graph_capture` 的 preheat_groups 要加 `("kunserve_p2p", lane_group)`，preheat 内容包含 send/recv pair。preheat 完成后必须有 `[KUNSERVE-MS] NCCL communicator preheat done` 日志。
- **Phase E negotiate 必须是 lockstep collective**——路线 B 不能引入仅本地 batch 变化时触发的 negotiate，否则两 replica 不对称时死锁。
- **不能把 top-k 改成 top-1，也不能改 routing 权重 dtype/scale**——只能改通信方式，不能改 routing 语义。
- **Static buffer 指针稳定性**：路线 B 新增 `_buf_p2p_*` 这类 buffer 时，禁止在 `dispatch`/`combine`/`_*_static` 中 reassign；只能在 `__init__` 或 `_allocate_static_buffers` 中分配，运行期用 `copy_`/`zero_`/`fill_` in-place 更新。
- **Phase F fallback**：所有路线 B 新 mode 都必须有 `else` 分支回退到 Phase F lane reduce_scatter，env off / 条件不满足时不允许 hard fail（除非是设计意图，如检测到错误 group 类型）。

## 九、参考代码位置

- `python/sglang/srt/distributed/parallel_state.py`：`@register_custom_op` 模式样板（`kunserve_lane_reduce_scatter_then_tp_all_reduce`）。
- `python/sglang/srt/distributed/kunserve_pynccl.py`：`_reduce_scatter_tensor`、`_all_reduce_in_place`、`preheat_for_graph_capture` 是要照搬的模板。
- `python/sglang/srt/distributed/device_communicators/pynccl.py`：`PyNcclCommunicator.send/recv/group_start/group_end`（line 317–386）。
- `python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py`：`ncclSend/ncclRecv/ncclGroupStart/ncclGroupEnd`（line 491–553）。
- `sgl-kernel/csrc/allreduce/custom_all_reduce.{cu,cuh}`：B3 / B1 的 SM-side reduce 模板。
- `python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`：`_dispatch_static` / `_combine_static` 是新模式接入点。
- `python/sglang/srt/model_executor/model_runner.py`：`preheat_for_graph_capture` 的 preheat_groups 是新通信模式必走的钩子。
