# MoE Comm/Compute Overlap 准备文档

更新时间：2026-05-28

## 一、目标与背景

### 1.1 已有结论

- 路线 A（lane RS + TP AR composite op）已落地、env-gated 默认关、对 wall-clock 无可见正收益（CUDA graph 已抹平 host 启动开销）。
- 路线 B（P2P / SM-side reduce）暂不实现，原因：当前 2-stage `lane RS + TP AR` 在 2 replica × TP=2 拓扑上已经 wire-optimal，B 路线只能省 ~1 NCCL launch 的 overhead，且 wire 实际更差。
- 与单实例 TP=4 的差距：**KunServe 多出来的 `dispatch lane all-gather (4.3 ms) + combine lane reduce-scatter (3.1 ms) = 7.4 ms / replay`** 是 cross-replica EP 的内禀代价，**结构上不能省**，只能想办法让它和 expert kernel 时间 overlap。

### 1.2 本路线目标

不改变通信总量，**通过把 dispatch/combine 这些 NCCL collective 安排到 side stream 上，让它们与 expert Triton kernel（7.2 ms / replay）部分重叠**，在不增加 wire 也不动模型语义的前提下压缩 decode replay GPU 时间。

### 1.3 现状测量（来自 `ab_20260528_041612`，post-balloon decode）

| 项 | 单层均值 | × 48 层 | 占 22.6 ms / replay |
|---|---|---|---|
| `graph_kunserve_dispatch_static_all_gather` | 0.090 ms | **4.3 ms** | 19% |
| `graph_fused_moe_core` (expert Triton kernel) | 0.149 ms | **7.2 ms** | 32% |
| `graph_kunserve_combine_static_lane_reduce_scatter` | 0.065 ms | **3.1 ms** | 14% |
| `graph_qwen3_moe_mlp_all_reduce` (TP AR) | 0.025 ms | **1.2 ms** | 5% |
| `graph_qwen3_moe_layer_attention` | 0.116 ms | **5.6 ms** | 25% |
| 单层 MoE 段合计 | 0.329 ms | **15.8 ms** | 70% |

理想 overlap 上限：把 7.4 ms 的 comm 完全藏到 7.2 ms 的 expert 后面 → 节省 ~7 ms / replay ≈ **31%**。
现实预期：30%–60% 的 overlap → 节省 **2–4 ms / replay ≈ 9%–18%**。

## 二、已有可复用基础设施

### 2.1 sglang 现成的 dispatch_a/dispatch_b/combine_a/combine_b 划分

`Qwen3MoeSparseMoeBlock` 已经有按算子拆分的 `op_*` 接口（用于 TBO/operation-level scheduling）：

- `qwen3_moe.py:420 op_dispatch_a / 428 op_dispatch_b` —— 调 `dispatcher.dispatch_a / dispatch_b`
- `qwen3_moe.py:437 op_experts` —— 调 `run_moe_core`
- `qwen3_moe.py:442 op_combine_a / 450 op_combine_b` —— 调 `dispatcher.combine_a / combine_b`

DeepEP / Mooncake / MoriEP dispatcher 都实现了 `dispatch_a/dispatch_b/combine_a/combine_b` 四个接口：
- `dispatch_a`: 启动 NCCL（async_finish=True），返回 event/hook，**不阻塞**
- `dispatch_b`: 等待 event/hook，返回真正的 dispatch_output
- `combine_a`: 启动 combine NCCL，返回 event
- `combine_b`: 等待并返回 final hidden_states

**KunServe 的 `CrossReplicaStandardDispatcher` 目前没实现这四个方法**——只有同步的 `dispatch / combine`。这是本次需要补的核心接口。

### 2.2 sglang 现成的 SBO (Single-Batch Overlap)

`sglang/srt/batch_overlap/single_batch_overlap.py`:

```python
@dataclass
class CombineOverlapArgs:
    overlap: bool          # 是否启用 down-gemm 内 overlap
    stream: torch.cuda.Stream    # alt_stream
    wait_event: torch.cuda.Event # main stream 等待事件
    num_sms: Optional[int]       # 给 communicate 留几个 SM
    signal: Optional[torch.Tensor] # blackwell 用的 per-block signal
    block_m: int = 64
    threshold: int = 0
```

`_DeepEPDispatcherImplLowLatency._combine_core` 是教科书案例：

```python
def _combine_core(self, hidden_states, topk_ids, topk_weights):
    overlap_args = self.overlap_args
    ctx = nullcontext()
    if overlap_args is not None:
        overlap_args.stream.wait_event(overlap_args.wait_event)   # alt 等 main 释放
        ctx = torch.cuda.stream(overlap_args.stream)              # 切到 alt stream
        ...
    with ctx:
        buffer.low_latency_combine(...)   # NCCL 在 alt stream 上 issue

def combine_b(self, hidden_states, event, hook):
    if overlap_args is not None:
        overlap_args.stream.wait_stream(self.device_module.current_stream())
    hook() if self.return_recv_hook else event.current_stream_wait()
    if overlap_args is not None:
        self.device_module.current_stream().wait_stream(overlap_args.stream)
    return hidden_states
```

这是我们要照搬的形态：side stream 上发起 collective + event 握手 + main stream 等 hook。

### 2.3 cuda_graph_runner 已支持多 stream capture

`cuda_graph_runner.py:407 self.stream_groups = get_stream_groups()` 已经为 pdmux 准备了多 stream 池；`graph_capture(stream=...)` 在 `capture_one_batch_size` 里支持显式选 capture stream。**这条路 graph 安全。**

### 2.4 KunServe 自己的 alt_stream 入口

`Qwen3MoeAttention` 已经接收 `alt_stream` 用于 `apply_qk_norm`（`qwen3_moe.py:561 self.alt_stream`），全模型范围有一个 alt_stream 实例。**MoE block 可以复用同一个 alt_stream，避免再开一条。**

## 三、KunServe MoE block 依赖 DAG 与可 overlap 区间

### 3.1 当前单层 forward 链（每个 rank 上）

```text
prepare_attn (norm + residual)                ┐
  ↓                                            │
attention QKV proj                             │
  ↓                                            │ 5.6 ms
attention compute                              │
  ↓                                            │
attention output proj                          │
  ↓                                            │
attention TP all-reduce  ──── 5.6 ms total ───┘
  ↓
prepare_mlp (norm + residual)                  ── 几十 us
  ↓
router_gate (linear hidden → router_logits)    ── ~50 us
  ↓
topk (argmax + softmax)                        ── ~50 us
  ↓
dispatch._buf_padded_*.copy_(...)              ── ~20 us
  ↓
dispatch.lane_all_gather(hidden_states)        ┐
dispatch.lane_all_gather(topk_ids)             │ 4.3 ms / replay
dispatch.lane_all_gather(topk_weights)         │ (3 条 NCCL 串行)
  ↓                                            ┘
dispatch.remap (topk_ids → local expert id)    ── ~30 us
  ↓
expert Triton kernel (gemm1 + silu + gemm2)    ── 7.2 ms / replay
  ↓
combine.lane_reduce_scatter                    ── 3.1 ms / replay
  ↓
TP all-reduce                                  ── 1.2 ms / replay
  ↓
post-layer (residual + norm)                   ── 几十 us
```

### 3.2 数据依赖分析

| 算子 | 依赖输入 | 后继依赖 | 能否搬到 side stream |
|---|---|---|---|
| dispatch.pad | hidden_states, topk_ids, topk_weights | dispatch.all_gather | ✓（输入就绪后立即起步） |
| dispatch.all_gather × 3 | padded buffers | dispatch.remap → expert | ✓（**主要 overlap 目标**） |
| dispatch.remap | gather 后的 topk_ids | expert | ❌（在 main stream） |
| expert kernel | union hidden + remapped topk | combine | ❌（**主要 compute 锚点**） |
| combine.lane_RS | expert 输出 | TP AR | ✓（**主要 overlap 目标**） |
| TP AR | combine 输出 | post-layer | ✓（小，可顺带） |

**只有 dispatch 和 combine 能搬到 side stream**。expert 是 compute 锚点，必须留在 main stream（或自己有 SM 调度）。

### 3.3 跨层 overlap 为什么不可行

| 试图 overlap 的对 | 依赖原因 | 结论 |
|---|---|---|
| 层 N combine ⊕ 层 N+1 prepare_attn | N+1 attention 输入 = N combine 输出 | 严格串行，无法 overlap |
| 层 N combine ⊕ 层 N+1 attention | 同上 | 无法 overlap |
| 层 N+1 dispatch ⊕ 层 N combine | dispatch 输入 = attention TP AR 输出 = 严格在 combine 之后 | 无法 overlap |
| 层 N attention ⊕ 层 N MoE 任何部分 | attention 输出是 MoE 输入 | 无法 overlap |

**跨层 overlap 没有真正的并行机会**——decoder 是严格链式。所有 overlap 必须落在**单层内部**。

### 3.4 单层内部的 overlap 机会窗口

按依赖关系画出"在 main stream 跑算子 X 的同时 side stream 上能跑什么"：

| main stream 算子 | 此时 side stream 可跑 | 时长上限 |
|---|---|---|
| router_gate + topk | dispatch.pad + dispatch.all_gather(hidden_states) 可以**预启动**（用前一步 prepare_mlp 出来的 hidden_states，topk 还没好但 hidden_states 已经好了） | ~100 us（不够吃 4.3 ms dispatch） |
| dispatch.all_gather(topk_*) | — | — |
| **expert kernel** | **dispatch 的最后几条 NCCL，或者上一 chunk 的 combine** | **7.2 ms 的窗口，主要 overlap 容器** |
| combine.lane_RS | TP AR 的部分（如果按 tile 切） | 1.2 ms |
| TP AR | post-layer 的小活儿（norm/residual） | 几十 us |

**核心 insight**：**expert kernel (7.2 ms) 是唯一足够长的 compute 段，可以做为 overlap 的容器**。要把 dispatch (4.3 ms) 或 combine (3.1 ms) 藏进去，必须满足"expert kernel 已经开始执行的同时，dispatch/combine 还能跑"——这要求 dispatch/combine 的某部分**不在 expert 入口之前的严格依赖链上**。

## 四、Overlap 候选方案（按工程量/收益排序）

### 4.1 候选 P1：dispatch 的 3 条 all_gather 合并为 1 个 ncclGroup（trivial）✅ Implemented (2026-05-28)

- **做法**：在 `_dispatch_static` 的 3 条 `_all_gather_into_tensor` 前后包 `pynccl_comm.group_start()` / `group_end()`。
- **收益**：省 2 条 NCCL launch overhead × 48 层 ≈ **0.5–1 ms / replay**（2–4%）。
- **风险**：极低。需要确认 graph capture 内 `ncclGroupStart/End` 行为正常。
- **工程量**：0.5 天。

#### 4.1.1 落地总结

- `kunserve_pynccl.py`：
  - 新增 `KunServePyNcclGroup.grouped_all_gather_into_tensor(pairs)`，绕开 `register_custom_op`（CUDA graph capture 不依赖该层），直接走 `pynccl_comm.group_start() → for each pair: all_gather → group_end()`。
  - `preheat_for_graph_capture` 末尾追加一段 group bracket 预热（3 条 dummy all-gather），遵守"new NCCL pattern must preheat"红线。preheat 日志加 `grouped_all_gather=True` 标记。
- `kunserve_standard.py`：
  - `__init__` 读 `KUNSERVE_DISPATCH_NCCL_GROUP`（默认 `1`）写入 `self._dispatch_ncclgroup_enabled`。
  - `_dispatch_static` 的 3 条 all-gather 改为：当 enabled 且 `gather_group` 有 `grouped_all_gather_into_tensor` 方法时调一次 grouped 接口，否则 fallback 到原 3 条单独 launch。
  - 第一次命中时打 `dispatch_ncclgroup_active` probe log。
- `CLAUDE.md`：env vars 表新增 `KUNSERVE_DISPATCH_NCCL_GROUP`。

#### 4.1.2 验证步骤

```bash
# A: 默认（grouped on）
ONLY_RUN=kunserve KUNSERVE_CAPTURE_POLICY=fixed_padded \
  KUNSERVE_ROLLOUT_QUANTIZATION=none KUNSERVE_MOE_A2A_BACKEND=none \
  KUNSERVE_MOE_RUNNER_BACKEND=triton KUNSERVE_FORWARD_TIMING_DETAIL=1 \
  KUNSERVE_GRAPH_INTERNAL_TIMING=1 KUNSERVE_GRAPH_INTERNAL_TIMING_INTERVAL=64 \
  bash /workspace/verl/data/compare_kunserve_vs_baseline.sh

# B: 强制 fallback 对照
KUNSERVE_DISPATCH_NCCL_GROUP=0  <其余参数同上>
```

A/B 之间看 `graph_kunserve_dispatch_static_all_gather` 的单层均值是否下降（预期 ~0.090 → ~0.06–0.07 ms），并看 `cuda_graph_replay_launch` / `graph_qwen3_moe_transformer` 是否同步下降。
detail log 里 `[KUNSERVE-DBG] dispatch_ncclgroup_active` 出现表示 grouped 路径已激活。

### 4.2 候选 P2：combine 搬到 side stream + chunked expert（SBO 风格）

#### 4.2.1 思路

把 expert Triton kernel 按 token 维拆成 2 个 chunk：
- `expert_chunk1` 处理 union 前一半（M tokens）
- `expert_chunk2` 处理后一半（M tokens）

时序：

```text
main:    [expert_h1=3.6ms][expert_h2=3.6ms][combine_h2=1.55ms]      end ≈ 8.75ms
alt:     ──────────────  [combine_h1=1.55ms (overlap w/ expert_h2)] end ≈ 5.15ms
```

- main stream 跑 `expert_h1 → expert_h2 → combine_h2`
- alt stream 在 `expert_h1` 完成后，开始 `combine_h1`，与 `expert_h2` 并行
- 在 `combine_h2` 之前 main stream 等 alt stream 的 `combine_h1` 完成

#### 4.2.2 收益估算

单层关键路径：

| 路径 | 时长 |
|---|---|
| 原始（顺序） | dispatch (0.09) + expert (0.149) + combine (0.065+0.025) = **0.329 ms / 层** |
| P2（chunked expert + combine overlap） | dispatch (0.09) + expert_h1 (0.075) + expert_h2 (0.075) + combine_h2 (0.045) = **0.285 ms / 层** |
| 节省 | **0.044 ms / 层 × 48 = 2.1 ms / replay ≈ 9%** |

#### 4.2.3 实现要点

1. **`CrossReplicaStandardDispatcher` 加 `combine_a/combine_b` 方法**：
   - `combine_a(combine_input, overlap_args)`: 切到 `overlap_args.stream`，issue `lane_group._reduce_scatter_tensor`（async），返回 event。
   - `combine_b(...)`: main stream wait alt stream，返回结果。
   - 静态路径 `_combine_static` 改造时保持 `_buf_combine_local_slice` 指针稳定（项目规则）。
2. **expert kernel 按 chunk 调用**：
   - `FusedMoE.run_moe_core` 上层新增 `run_moe_core_chunked(chunk_idx, chunk_count)`，每 chunk 调一次 `TritonRunnerCore.run`，input 切片 `[chunk_start:chunk_end]`。
   - Triton kernel 本身不改；只是分两次调用，每次 M=128。
   - 验证：Triton kernel 在 M=128 时单次时间是否依然约为 M=256 的一半（不能因为 launch overhead 而退化）。
3. **alt_stream 申请**：直接复用 `Qwen3MoeAttention.alt_stream`，或者在 `ModelRunner` 里给 KunServe MoE 块独立 `kunserve_moe_alt_stream`。
4. **CUDA graph 兼容性**：
   - 需要在 graph capture 内做 `stream.wait_event()` / `stream.wait_stream()`。pytorch 支持但需要 cudaStreamWaitEvent 的 capture/replay 不出问题（H100 + CUDA 12 + PyTorch 2.x 是稳定的）。
   - 在 `cuda_graph_runner.capture_one_batch_size` 流程外不需要改，因为 capture 期间不需要切换 capture stream（只是在 capture stream 上额外 issue 一条到 alt stream 的依赖）。
   - **必须 preheat alt stream 上的 NCCL 通信**（lane_group reduce_scatter on alt_stream），否则首次 capture 在 alt 上的 NCCL lazy init 会污染 graph。已有 `preheat_for_graph_capture` 钩子，加一个 alt_stream 上的 dummy reduce_scatter 即可。

### 4.3 候选 P3：dispatch + combine 同时搬到 side stream + chunked dispatch + chunked expert + chunked combine

最激进。3 条 collective 全部 chunk + pipeline：

```text
main:    [d_h1 0.045][e_h1 0.075][e_h2 0.075][c_h2 0.045]          end ≈ 0.240ms
alt:                  [d_h2 0.045][c_h1 0.045 from t=0.165]        end ≈ 0.210ms
```

- 单层关键路径：**0.240 ms**
- 收益：**0.329 - 0.240 = 0.089 ms / 层 × 48 = 4.3 ms / replay ≈ 19%**

但工程量明显更大：
- dispatch 要拆成 2 个 chunk 的 NCCL（用 `lane_all_gather_into_tensor` 切两次输入，配套两个 `gathered_*` 半区 buffer），保证指针稳定。
- expert 必须支持半批次（已经在 P2 里做了）。
- combine 同样切两半。
- main / alt 流之间需要 4 个 event 来串依赖（dispatch_h1→expert_h1, dispatch_h2→expert_h2, expert_h1→combine_h1, expert_h2→combine_h2），且 alt 流上的 NCCL 要 preheat 两次（dispatch 类型 + combine 类型）。

**建议先 P2 跑通拿到 9% 收益，再决定是否扩展到 P3 拿额外 10%。**

### 4.4 候选 P4：TBO（two-batch overlap）启用

- sglang 已有完整 TBO 框架（`batch_overlap/two_batch_overlap.py`、`op_dispatch_a/b`、`op_combine_a/b`）。
- TBO 把单 batch 拆成 2 个 micro-batch，跨 micro-batch overlap dispatch/expert/combine。
- 看似工程量最小（复用现有框架），但：
  - decode batch 已经是 M=256 这种小尺寸，再切成 128 + 128，Triton kernel 效率可能掉。
  - TBO 主要为 prefill 设计，decode 上未必有正收益。
  - 与 KunServe `CrossReplicaStandardDispatcher` 的 GLOBAL EP 语义未必兼容（要先验证 TBO 调度器对 cross-replica EP 是否安全）。
- **暂不推荐。先以 P2 这种"轻量 SBO-like 改造"为主**。

### 4.5 候选 P5：原 SBO down-gemm 信号 overlap

- DeepEP low-latency dispatcher 用 `signal` 张量做 per-block 进度通知，combine kernel 看 signal 决定是否可以读这一行。
- Triton expert kernel 没有这种 per-block signal 机制，要写就得改 Triton kernel 或者新增一个 CUDA kernel。
- **暂不推荐**——这是 B1 的变种，工程量超出本路线目标。

## 五、推荐执行顺序

1. **阶段 A（0.5 天）**：实现 P1（dispatch 3 个 all_gather 用 `ncclGroupStart/End` 合并）。
   - 改动文件：`kunserve_standard.py::_dispatch_static`、可能要在 `kunserve_pynccl.py` 暴露一个 `group_start/group_end` 接口（或直接拿 `pynccl_comm.group_start()`）。
   - 测：cuda_graph_replay_launch 是否下降。

2. **阶段 B（3–5 天）**：实现 P2（combine 上 alt_stream + chunked expert）。
   - **B.1**（1 天）：在 `CrossReplicaStandardDispatcher` 添加 `combine_a/combine_b` 接口，先用 alt stream 但不切 chunk，验证 alt stream 上 graph-safe NCCL 行为。
   - **B.2**（1 天）：在 `FusedMoE` / `qwen3_moe.py` 上添加"分两次调用 expert kernel"的代码路径。env-gate `KUNSERVE_OVERLAP_EXPERT_COMBINE=1`。
   - **B.3**（1 天）：把 B.1 和 B.2 拼起来，调好 event 握手。
   - **B.4**（1 天）：preheat 钩子、正确性 dump 对照、性能测量。
   - **B.5**（1 天）：缓冲量调优；如果发现 Triton kernel 在 M=128 不到一半时间，考虑用更小的 chunk_count（例如 chunk_count=3）或者动态选择。

3. **阶段 C（视 P2 收益决定）**：
   - 若 P2 收益≥ 5% 且数值稳定 → 评估 P3（再 chunk dispatch）。
   - 若 P2 收益不足 5% 或 chunk 后 Triton kernel 退化严重 → 改攻 expert kernel 本身（切 FP8 deepep / 调 Triton autotune）。

4. **阶段 D**：路线稳定后回看 Phase E（idle keepalive）。

## 六、必须遵守的红线

1. **`_buf_*` 指针稳定性**（项目规则）：chunked 实现新增的 buffer 必须在 `__init__` / `_allocate_static_buffers` 中分配；运行期只能 `copy_/zero_/fill_` in-place。
2. **alt_stream 上的 NCCL 必须先 preheat**（项目规则 + NCCL lazy init 既往坑）：
   - 在 `preheat_for_graph_capture` 里增加一个 `with torch.cuda.stream(alt_stream): lane_group.reduce_scatter_tensor(...)` 的 dummy call。
   - preheat 完成后必须打 `[KUNSERVE-MS] alt-stream NCCL preheat done` 日志。
3. **不能让 P2/P3 路径在 capture 中悄悄 fallback 到 raw torch.distributed**——必须仍然走 KunServePyNccl 注册路径，否则 hard fail。
4. **不能改变 expert kernel 的数值行为**：chunk 切分必须保证 `expert(union[0:M]) || expert(union[M:2M])` 与 `expert(union[0:2M])` bitwise / 至少 numerically 等价（routing weight、scaling factor、reduce 顺序都不变）。**切分点必须在 row 维度，不能跨 expert 维度**。
5. **alt_stream 异常 fallback**：env off / overlap_args=None 时，必须回到原 main stream 串行路径，behavior 完全等价。
6. **Phase F 兼容**：新增 chunked path 必须保留 `if not phase_f_enabled` 的 Phase D 回退分支，否则会破坏现有 fallback 链。
7. **CUDA graph capture 边界**：每个 (variant, bs) 的 graph 内 stream 拓扑结构必须固定。不能动态根据 batch size 改 chunk_count——否则 capture 与 replay 拓扑不一致。chunk_count 在 dispatcher `__init__` 时锁死。
8. **Timing 注入对 alt_stream 友好**：`kunserve_timing_scope` 用的 `cudaEvent` 必须 record 在正确的 stream 上，否则会得到错误的 elapsed_ms。需要在 detail timing scope 里加 stream 参数支持。

## 七、正确性与性能验证 plan

### 7.1 正确性

- **逐层 hidden_states diff**：在 overlap 路径 vs 串行路径之间，比较每层 MoE 输出的 max abs diff。bf16 路径下应该 ≤ 1e-3。
- **streaming dump tail 对照**（参考 [[finding_fp32_lane_reduce]]）：分别跑 5K token 和 25K token 的 decode，比较 tail 输出的 perplexity / 是否出现乱码。
- **批 size 扫描**：M_replica = 64 / 128 / 256 / 512 都跑一遍，确认 chunk 切分对小 batch 也安全（M=64 切 2 chunk 是 M=32，Triton 可能很难看）。

### 7.2 性能

- **graph 内 stage 时间**：
  - `graph_qwen3_moe_layer_mlp`（整段）：应该下降
  - `graph_fused_moe_core`（expert）：应该保持 ≈ 0.149 ms（不能因为 chunk 而退化超过 5%）
  - `graph_kunserve_combine_static_*`（combine）：应该保持或下降
- **总指标**：
  - `graph_qwen3_moe_transformer`：22.6 → 期望 20.5（P2）或 19（P3）
  - `cuda_graph_replay_launch`：观察是否同步下降（这是 host 视角，理论上下降不明显，因为 graph 内 GPU 端 overlap 不直接影响 host enqueue 时间）
- **A/B**：用 `compare_kunserve_vs_baseline.sh` 跑 overlap on vs off，比较 token throughput。

## 八、关键代码位置速查

| 文件 | 行/符号 | 作用 |
|---|---|---|
| `models/qwen3_moe.py` | `Qwen3MoeSparseMoeBlock.forward_normal` (297) | MoE block 入口，要插入"chunked path"分支 |
| `models/qwen3_moe.py` | `op_dispatch_a/b op_combine_a/b` (420–454) | TBO 接口，本路线 P2 也会复用 `dispatch_a/b/combine_a/b` 命名 |
| `models/qwen3_moe.py` | `self.alt_stream` (561) | 已存在的 alt stream，可复用 |
| `layers/moe/fused_moe_triton/layer.py` | `FusedMoE.forward_impl` (1365) | dispatch → core → combine 链路 |
| `layers/moe/token_dispatcher/kunserve_standard.py` | `_dispatch_static` (932), `_combine_static` (1122) | 静态 graph 路径，本路线主要改造对象 |
| `layers/moe/token_dispatcher/deepep.py` | `_DeepEPDispatcherImplLowLatency._combine_core` (756), `combine_b` (744) | alt_stream + event 握手模板 |
| `batch_overlap/single_batch_overlap.py` | `CombineOverlapArgs` (63), `compute_overlap_args` (81) | 直接复用的 dataclass |
| `distributed/kunserve_pynccl.py` | `_reduce_scatter_tensor` (262), `preheat_for_graph_capture` (278) | preheat 的位置 |
| `model_executor/model_runner.py` | `preheat_for_graph_capture` 调用点 | 加 alt-stream preheat 钩子 |
| `model_executor/cuda_graph_runner.py` | `capture_one_batch_size` (725), `stream_groups` (407) | 多 stream graph capture 基础设施 |

## 九、风险与未决问题

1. **Triton kernel 在 M=128 是否真的接近一半时间**？
   - 风险：launch overhead 占比上升，造成 chunked 路径 expert 总时间大于原始单次调用。
   - 实验：先单独测 `triton.fused_moe_kernel(M=256)` vs `2 × triton.fused_moe_kernel(M=128)` 的实测时间，再决定 chunk_count。
2. **alt stream 上 NCCL 在 CUDA graph capture 内的稳定性**：
   - DeepEP 已经验证可用，但用的是 `Buffer.capture()` / `low_latency_*`，跟 KunServe 的 PyNccl 不完全一样。
   - 需要写一个 micro-test：`graph_capture context manager` 里在 alt stream 上 issue `lane_group._reduce_scatter_tensor`，replay 后验证结果正确。
3. **chunk 边界的 dispatch buffer 切分**：
   - 现在 `_buf_union_hidden[2*M, H]` 是连续的，可以 `narrow(0, 0, M)` 和 `narrow(0, M, M)` 各拿一半，不需要新 buffer。
   - 但要确认 NCCL `_reduce_scatter_tensor` 接受 `narrow` 后的视图作为 input/output（应该可以，contiguous on dim 0）。
4. **本地 TP all-reduce 在 chunked combine 后的位置**：
   - 当前 TP AR 由 `Qwen3MoeSparseMoeBlock.forward_normal` 在 `experts()` 之后做。chunked combine 后，可以让 TP AR 也分两次（chunk1 和 chunk2 各做一次），或者合并后做一次。
   - 选合并：等两 chunk 的 combine 都完成 → concat → TP AR（一次）。少一条 NCCL 启动但 critical path 不变。
   - 选分开：combine_h1 完成后立即 TP_AR_h1 在 alt stream，和 main stream 上的 expert_h2 / combine_h2 进一步 overlap。但 NCCL 启动数翻倍。
   - **建议先选合并，简单稳妥**。
5. **是否启用 route A composite op**：
   - 如果 chunked combine 后还要做 TP AR，可以顺手把 fused composite op（route A）也开起来，把 chunk_combine 的 `lane_RS` 和 `TP_AR` 仍然合并成一条 custom op。
   - 不强求，先各跑各的。

## 十、后续路线衔接

如果 P2 + P3 都跑完仍想再压：

- **F1：把 expert Triton kernel 替换成 DeepGEMM FP8**——`fused_moe_core` 单层从 0.149 ms 降到 ~0.08 ms（KunServe 已有 deepep 路径作为参考）。这是另一条 ~15% 收益的独立路线。
- **F2：把 chunk_count 提升到 3 或 4**——更细粒度 overlap，但 Triton kernel 在很小 batch 上效率下降越来越严重，收益递减。
- **F3：再回头看 B1（SM-side reduce kernel）**——在 chunked pipeline 已经把 launch overhead 摊薄之后，B1 的边际收益更小，性价比更低。

本文档维护：路线 A、P1、P2 落地后回来更新"已落地"小节，把预期收益替换为实测值；路线 B 暂存 [[lane_combine_tp_allreduce_fusion_准备文档]] 第六节作为参考资料。
