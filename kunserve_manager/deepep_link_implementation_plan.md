# KunServe 跨实例 MoE 通信:DeepEP 链路实现方案

> 分支：`feat/deepep-comm`
> 状态：设计阶段（开始写代码前的方案）
> 最后更新：2026-06-03（修订：传输层直接用 DeepEP，不走 PyTorch NCCL）
>
> 目标：用 DeepEP 的 **token-routed dispatch/combine** 替换当前的 dense all-gather + reduce-scatter，抹掉跨实例通信开销（dense 路径实测 ~2.3–5ms/step）。
>
> **三条原则（本方案的出发点）：**
> 1. **不盲信现有 DeepEP 代码**。仓库里 `kunserve_comm_backend=deepep` 那套（`MaybeTboDeepEPDispatcher` + NVSHMEM + FP8 + deep_gemm）**从未端到端跑通**，当作"脚手架/参考"，其假设（尤其 FP8-only 的硬 assert）要重新审视，不照搬。
> 2. **传输层直接用 DeepEP**，不再像现在这样用 PyTorch NCCL（dense all-gather / Phase G a2a）。NVSHMEM/IBGDA 环境由另一条线解决，本方案**假设 H20 上 NVSHMEM 已可用**。
> 3. **精度解耦，bf16 是今后的一等公民**。通信精度与专家计算精度必须与传输解耦：**先用 fp8/deep_gemm 把 DeepEP 跑通**（现有脚手架就是这条，改动最小），**bf16/triton 作为后续里程碑**（设计上预留好接口，不现在实现）。

---

## 1. 现状批判（哪些能用、哪些不能信）

仓库里已有三条跨实例链路，逐条评估：

| 链路 | 实现 | 能复用的部分 | 不能信/要改的部分 |
|---|---|---|---|
| ① dense | `CrossReplicaStandardDispatcher`（sglang backend，NCCL） | static remap 接线、`reduce_results=False` 语义、lane/runtime group 解析、graph-safe 静态 buffer 写法 | dense 发全部 token——要被替换的开销源；**本方案不再以它为传输** |
| ② Phase G | `CrossReplicaTokenA2ADispatcher`（NCCL，`KUNSERVE_PHASE_G=1`） | **routing 元数据计算**（owner 判定、send/recv counts、combine 反向映射）——可作为理解参考 | NCCL a2a，eager-only；**本方案不用它做传输**，但其路由语义可对照 |
| ③ DeepEP | `MaybeTboDeepEPDispatcher`（`kunserve_comm_backend=deepep`） | **本方案的基础**：buffer 按 (group,mode) 缓存、`register_balloon_global_runtime_bundle` 的 GLOBAL metadata/static-remap 接线、LOCAL 强制 Standard 防 NVSHMEM double-init、LL dispatch/combine 调用 | **FP8/deep_gemm 写死**（`model_runner.py:1402-1417` assert）要拆成精度策略；**没跑通**，warmup/commit/LL 路径要逐段验证修复 |

**结论**：以 ③ 为基础，**先让它真正跑起来（fp8）**，同时把"FP8-only assert"换成精度策略，为今后 bf16 留口。①②不再作为传输路径。

---

## 2. 设计总览:三层解耦（传输固定为 DeepEP）

```text
              topk_ids(GLOBAL physical) + hidden_states
                              │
        ┌─────────────────────▼─────────────────────┐
        │ L1 Routing/Metadata（精度无关、传输无关）   │
        │  static remap → GLOBAL physical id          │
        │  topk → 每 expert/rank 目的地（DeepEP 用）   │
        └─────────────────────┬─────────────────────┘
                              │
        ┌─────────────────────▼─────────────────────┐
        │ L2 Transport = DeepEP（唯一）               │
        │  low_latency_dispatch / low_latency_combine │
        │  dispatch_dtype ∈ { fp8(先), bf16(今后) }   │
        └─────────────────────┬─────────────────────┘
                              │ per-expert grouped recv (+ masked_m)
        ┌─────────────────────▼─────────────────────┐
        │ L3 Runner Adapter（精度策略落地）           │
        │  fp8  → deep_gemm grouped GEMM（先，原生）   │
        │  bf16 → triton fused_moe（今后,需布局适配)  │
        └─────────────────────────────────────────────┘
```

**精度策略（PrecisionPolicy）独立配置：**

| 配置项 | 取值 | 现在默认 | 今后 | 说明 |
|---|---|---|---|---|
| `transport` | **deepep** | deepep | deepep | 直接用 DeepEP，不走 PyTorch NCCL |
| `dispatch_dtype` | fp8 / bf16 | **fp8** | bf16 | DeepEP 通信 payload 精度（`use_fp8` 开关） |
| `expert_runner` | deep_gemm / triton | **deep_gemm** | triton | 专家 GEMM 后端 |

> 关键解耦点：**transport=deepep 不应再强制 expert_runner=deep_gemm**。DeepEP 支持 bf16 dispatch（`use_fp8=False` / `SGLANG_DEEPEP_BF16_DISPATCH`）；专家计算用 deep_gemm(fp8) 还是 triton(bf16) 由 L3 决定。现在先走原生 fp8/deep_gemm，bf16 是 L3 增量。

---

## 3. 各层详细设计

### L1 Routing/Metadata
- 复用 `register_balloon_global_runtime_bundle` 构造的 GLOBAL expert location metadata 与 `logical_to_rank_dispatch_physical_map`（static remap）。
- `ep_dispatch_algorithm=static`：router 输出的 logical id → balloon 后两 replica 互补 half-split 的 **GLOBAL physical id**。**这步错 → DeepEP 路由到错 owner → 乱码**，是首要验证点（model_runner 已有 `[KUNSERVE-DBG]` 日志可查 map 是否非 None）。

### L2 Transport = DeepEP
- 复用 `DeepEPBuffer`（按 group+mode 缓存，避免 LOCAL/GLOBAL、NORMAL/LL 互相覆盖）。
- decode 走 LL：`low_latency_dispatch(hidden, topk_ids, num_max_dispatch_tokens_per_rank, num_experts, use_fp8=(dispatch_dtype==fp8), ...)`。
- combine：`low_latency_combine(x, topk_idx, topk_weights, handle, ...)`。
- prefill/extend 走 NORMAL（`deepep_mode=auto` 自动解析）。

### L3 Runner Adapter（精度落点）
- **fp8/deep_gemm（先做）**：DeepEP grouped 输出（`packed_recv_x` + `masked_m`/`expected_m`）**原生**就是 deep_gemm grouped FP8 GEMM 的输入，基本沿用现有 ③ 路径。
- **bf16/triton（今后,M4）**：DeepEP bf16 grouped 输出 → 转成 triton fused_moe 期望的 (sorted tokens, expert boundaries) 布局 → 跑 bf16 grouped expert → 还原。**这层布局转换是 bf16 的核心新代码**，现在只预留接口。

---

## 4. 必须保持的语义（验收清单）

来自 `deepep_vs_kunserve通信原理与bs80预期.md §4`：
1. attention/KV 不跨 replica，请求不迁移。
2. dispatch 等价 union token：本 rank 需要的 expert 的 token 全收到。
3. combine 回到原 replica、原 token 顺序，shape 与 local MLP 输出一致。
4. padding/phantom rows 不影响 logits。
5. `reduce_results=False`（DeepEP combine 已聚合 → 禁止再 TP all-reduce，否则 double-reduce）。
6. idle replica keepalive 到对端结束（Phase E；DeepEP NORMAL 由 buffer setup 兜，LL 下需确认）。
7. graph replay 下 DeepEP workspace/buffer 指针在 capture 前固定。

---

## 5. CUDA graph 安全性（直接用 DeepEP 的主要理由）

稀疏路由是数据相关的变长，而 graph 要求静态 shape。**DeepEP LL 正是为此设计**：用 `num_max_dispatch_tokens_per_rank`（固定上界）+ `masked_m`（kernel 内有效计数）在 **graph 内处理变长，不需要 host sync、不需要把 a2a pad 成 worst-case dense**。

这就是"直接用 DeepEP 而不是 PyTorch NCCL"的根本原因：裸 NCCL 的 `all_to_all_single` 要进 graph 必须固定 split → 退化成 dense padding，省不下来；DeepEP 没有这个问题。

graph 化要点：
- `num_max_dispatch_tokens_per_rank` 在 capture 前确定，作为上界（运行时实际 token 数 ≤ 它）。
- `DeepEPBuffer` 在 capture 前按 (group, LL mode) 建好并固定。
- warmup 阶段 `ensure_cuda_graph_variant_captured("global")` 完成 GLOBAL 图 capture。

---

## 6. 实现里程碑（DeepEP-direct）

| 里程碑 | 内容 | 传输 | 精度 | graph | 验收 |
|---|---|---|---|---|---|
| **M0** | L1 static remap + RoutingPlan 正确性单测（对拍 dense 的 union 语义；纯单卡可测） | — | — | — | map 非 None；本 rank 需算的 (t,k) 全覆盖 |
| **M1** | **点亮现有 DeepEP 路径端到端**（NVSHMEM 可用后，逐段修 warmup/commit/LL dispatch/combine/balloon 布局，使其真正跑通） | DeepEP | fp8/deep_gemm | eager 先 | smoke 端到端不崩；temp=0 输出对拍 dense 一致、无乱码 |
| **M2** | GLOBAL DeepEP LL 的 CUDA graph capture/replay 稳定 | DeepEP | fp8 | ✅ | `Capturing batches ... variant='global'` 成功；replay 输出仍一致 |
| **M3** | 用 profiling 链路量收益 | DeepEP | fp8 | ✅ | `dispatch/combine` 阶段 ms vs dense 显著下降；baseline-vs-GLOBAL matched bs+KV 残差缩小 |
| **M4(今后)** | **bf16 支持**：`use_fp8=False` + L3 triton 布局适配 | DeepEP | **bf16/triton** | ✅ | bf16 路径对拍 fp8/dense；为无 deep_gemm/想避 FP8 数值的场景留路 |

> M1 是关键且最省事的一步：**不是写新 dispatcher，而是把已有但没跑通的 ③ 修到能跑**（现在 NVSHMEM 假设可用，之前卡的就是它）。M4 的 bf16 才需要实质新代码（L3 适配）。

---

## 7. 文件改动清单（预估）

| 文件 | 改动 |
|---|---|
| `model_executor/model_runner.py` | `register_balloon_global_runtime_bundle`：**把"deepep ⇒ deep_gemm" 的硬 assert 换成 PrecisionPolicy 校验**（fp8→deep_gemm，bf16→triton）；GLOBAL bundle 接精度策略 |
| `layers/moe/token_dispatcher/deepep.py` | LL dispatch/combine：`use_fp8` 由 `dispatch_dtype` 驱动；逐段验证修复没跑通的地方；buffer 缓存确认 |
| `layers/moe/token_dispatcher/kunserve_runner_adapter.py`（**M4 新**） | L3：DeepEP grouped(bf16) → triton fused_moe 布局适配（仅 bf16 时需要） |
| `layers/moe/fused_moe_triton/layer.py` | `register_runtime_bundle` 支持 PrecisionPolicy（dispatch_dtype / expert_runner）；已有 explicit_global_dispatcher 通道可扩展 |
| `srt/server_args.py` / env | `KUNSERVE_DISPATCH_DTYPE`(fp8/bf16)、`KUNSERVE_EXPERT_RUNNER`(deep_gemm/triton)；`transport` 固定 deepep |
| 文档 | 同步 `kunserve_综合分析与优化文档.md` + CLAUDE.md 状态表 + 本文件 checklist |

---

## 8. 验证与回归

- **正确性（每个里程碑）**：temp=0 确定性 + prompt 文本配对，KunServe-GLOBAL 输出与 dense 路径（已验证基线）逐 prompt 对拍：长度比≈1.0、答案一致、无乱码。
- **性能**：`profile_orch.py` + `parse_stage_traces.py`，对照 dense 的 `stages_*.csv`，看 dispatch/combine 阶段 ms 下降；端到端用 baseline-vs-GLOBAL matched bs+KV 残差（铁律：不看噪声大的端到端总和）。
- **回归**：`ONLY_RUN=kunserve bash compare_kunserve_vs_baseline.sh` 必须先过再 claim 完成。

---

## 9. 风险与未决问题

1. **M1 是"修没跑通的现有代码"**，未知点多（warmup/commit/LL handle/balloon static remap 任一段都可能有 bug）——逐段加日志、二分定位。
2. **数值**：fp8 expert（deep_gemm）在 30B 上历史 temp=0 无漂移，仍需重验；80B(linear-attention) 另案。
3. **graph capture**：DeepEP LL 在 KunServe 自定义 capture 路径下的 buffer 固定要确认（③ 的 capture 从未真正验证过）。
4. **M4 L3 布局转换**（grouped↔sorted）是 bf16 的最大不确定性，单测要足。
5. **Phase E keepalive** 在 LL 下要确认 idle replica 不 hang。
6. **拆 assert 的向后兼容**：不破坏现有 ③ 的默认行为，新策略走新参数。

---

## 10. 一句话路线

**传输层直接用 DeepEP（不走 PyTorch NCCL）。先把现有但没跑通的 fp8/deep_gemm DeepEP 路径修到能端到端跑通并 graph 化（M1–M3），同时把 FP8-only 的硬约束换成精度策略；bf16（DeepEP bf16 dispatch + triton 布局适配）作为今后的一等公民里程碑（M4）预留接口、按需再实现。**
