# KunServe 跨实例 MoE 通信:DeepEP 链路实现方案

> 分支：`feat/deepep-comm`
> 状态：设计阶段（开始写代码前的方案）
> 最后更新：2026-06-03
>
> 目标：用 **token-routed dispatch/combine** 替换当前的 dense all-gather + reduce-scatter，抹掉跨实例通信开销（dense 路径实测 ~2.3–5ms/step）。
>
> **两条硬性原则（本方案的出发点）：**
> 1. **不盲信现有 DeepEP 代码**。仓库里 `kunserve_comm_backend=deepep` 那套（`MaybeTboDeepEPDispatcher` + NVSHMEM + FP8 + deep_gemm）**从未端到端跑通**，只能当作"脚手架/参考"，其假设(尤其 FP8-only)要重新审视，不照搬。
> 2. **bf16 与 fp8 都要是一等公民**。通信精度、专家计算精度必须与"是否用 DeepEP 传输"解耦；默认走 **bf16/triton**（沿用已验证、无漂移的数值栈），fp8/deep_gemm 作为可选。
>
> H20 上 NVSHMEM/驱动等环境问题本方案**暂不处理**，假设传输层可用；环境是另一条工作线。

---

## 1. 现状批判（哪些能用、哪些不能信）

仓库里已有三条跨实例链路，逐条评估：

| 链路 | 实现 | 能复用的部分 | 不能信/要改的部分 |
|---|---|---|---|
| ① dense | `CrossReplicaStandardDispatcher`（sglang backend） | 静态 buffer 管理、graph-safe 写法、lane/runtime group 解析、`reduce_results=False` 语义、static remap 接线 | 发全部 token（dense），是要被替换的开销源 |
| ② Phase G | `CrossReplicaTokenA2ADispatcher`（`KUNSERVE_PHASE_G=1`） | **routing 元数据计算**（owner-in-lane、send/recv counts、flat (t,k) 索引、combine 反向映射）——这块是 token-route 的核心，可直接演进 | **eager-only**，graph 下自动回退①；NCCL a2a 是动态 shape，未 graph 化 |
| ③ DeepEP | `MaybeTboDeepEPDispatcher`（`kunserve_comm_backend=deepep`） | buffer 按 (group, mode) 缓存的思路、`register_balloon_global_runtime_bundle` 的 GLOBAL metadata/static-remap 接线、LOCAL 强制 Standard 防 NVSHMEM double-init | **FP8/deep_gemm 写死**（`model_runner.py:1402-1417` assert）；**没跑通**；LL 输出布局与 triton runner 不兼容；强绑 NVSHMEM |

**结论**：
- ②的 **routing 元数据** 是最有价值的现成资产，新 dispatcher 在它基础上演进。
- ③的 **接线/状态机**（warmup/commit/metadata/static-remap）可复用，但**它的 FP8-only assert 和"DeepEP 必配 deep_gemm"要拆掉**，换成"精度策略"。
- ①给出 graph-safe 静态 buffer 的工程范式，照抄其稳定性约束。

---

## 2. 设计总览：三层解耦

把跨实例 MoE 通信拆成**三个正交的层**，精度与传输互不绑定：

```text
              topk_ids(GLOBAL physical) + hidden_states
                              │
        ┌─────────────────────▼─────────────────────┐
        │ L1 Routing/Metadata（精度无关、传输无关）   │
        │  - 由 topk_ids 算每个目标 rank 的 token 清单 │
        │  - send/recv counts、flat(t,k) 索引          │
        │  - combine 反向映射 handle                   │
        └─────────────────────┬─────────────────────┘
                              │ (routing plan)
        ┌─────────────────────▼─────────────────────┐
        │ L2 Transport（可插拔后端）                  │
        │   backend ∈ { nccl_a2a, deepep }            │
        │   dispatch: 把本 rank 需算的 token 收过来   │
        │   combine : 把 expert 输出发回原 token       │
        │   dtype   ∈ { bf16, fp8 }                   │
        └─────────────────────┬─────────────────────┘
                              │ (recv tokens, per-expert grouping)
        ┌─────────────────────▼─────────────────────┐
        │ L3 Runner Adapter（精度策略落地）           │
        │   runner ∈ { triton(bf16), deep_gemm(fp8) } │
        │   把 L2 的 recv 布局 → runner 期望布局 → 跑  │
        │   → 还原 → 交回 L2 combine                   │
        └─────────────────────────────────────────────┘
```

**精度策略（PrecisionPolicy）独立配置：**

| 配置项 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `dispatch_dtype` | bf16 / fp8 | **bf16** | 通信 payload 精度 |
| `expert_runner` | triton / deep_gemm | **triton** | 专家 GEMM 精度后端 |
| `transport` | nccl_a2a / deepep | **nccl_a2a**(先) → deepep | 传输实现 |

> 关键点：**transport=deepep 不再强制 fp8**。DeepEP 本身支持 bf16 dispatch（`SGLANG_DEEPEP_BF16_DISPATCH`）；专家计算用 triton 还是 deep_gemm 由 L3 适配，与传输无关。这就是"bf16 一等公民"的落点。

---

## 3. 各层详细设计

### L1 Routing/Metadata（演进自 Phase G）

输入：`topk_ids`（已被 static remap 成 GLOBAL physical id，shape `[M, K]`）、`topk_weights [M,K]`、`hidden_states [M,H]`、`local_expert_mapping[num_experts]`（physical→本 rank compact row，或 -1）。

产出一个 **RoutingPlan**（dataclass）：
```python
@dataclass
class RoutingPlan:
    # 本 rank 要发给每个目标 rank 的 token 数（含自身）
    send_counts: Tensor            # [world]  (graph: 固定上界 + mask)
    recv_counts: Tensor            # [world]
    # 发送侧 flatten：每个 send slot 对应的 (origin_token, k, weight, dest_expert_local)
    send_token_idx: Tensor         # [S]
    send_k_idx: Tensor             # [S]
    # combine 反向：每个 recv slot 算完后要回到的 (origin_rank, origin_token, k)
    recv_origin: Tensor            # [R, ...]
    # 本 rank recv 后每个 token 的目标 local expert row
    recv_expert_local: Tensor      # [R]
```
- owner 判定沿用 Phase G 的 `owner_global = topk_ids // num_local_experts`，再 → lane-local rank。
- **不属于本卡的 (t,k) 不进入 send**（这就是省掉 dense 无效 payload 的来源）。

### L2 Transport（可插拔）

统一接口：
```python
class CrossReplicaTransport(Protocol):
    def dispatch(self, hidden: Tensor, plan: RoutingPlan, dtype) -> RecvBatch: ...
    def combine(self, expert_out: Tensor, plan: RoutingPlan, dtype) -> Tensor: ...
```

**后端 A：`nccl_a2a`（先做，bf16 友好，无新依赖）**
- 用 cross-replica / lane NCCL group 的 `all_to_all_single`。
- eager 模式：用真实 `send_counts/recv_counts`（动态 split），最省。
- graph 模式：split 必须固定 → 每对 rank 固定 `M_max`（`num_max_dispatch_tokens_per_rank`）+ mask；和 dense 比，省在"按 expert 归并 + 跳过无效"，但有 padding 上界（见 §5 的取舍）。

**后端 B：`deepep`（后做，graph 内变长高效）**
- 复用 `DeepEPBuffer`（按 group+mode 缓存）。
- `low_latency_dispatch(..., use_fp8 = (dispatch_dtype==fp8))` —— bf16 时关掉 fp8。
- 输出是 per-expert grouped 布局（`packed_recv_x` + `masked_m`），交给 L3 适配。

### L3 Runner Adapter（精度策略落地，**新代码的重点**）

职责：把 L2 的 recv 布局喂给指定 runner，跑完再还原。
- `triton` 分支：把 recv tokens 按 expert 排序对齐（`moe_align`），调现有 Triton fused_moe（bf16）。**DeepEP grouped 输出 → triton 期望的 (sorted tokens, expert boundaries) 的转换是这一层的核心难点**，但两边本质都是"按 expert 分组的 token"，可做零拷贝/轻量 gather。
- `deep_gemm` 分支：DeepEP grouped 输出可直接喂 grouped FP8 GEMM（与现有 ③ 路径一致）。
- 两分支输出统一成 `[R, H]`，交回 L2 combine。

---

## 4. 必须保持的语义（验收清单）

来自 `deepep_vs_kunserve通信原理与bs80预期.md §4`，逐条要测：
1. attention/KV 不跨 replica，请求不迁移。
2. dispatch 等价 union token：本 rank 需要的 expert 的 token 全收到。
3. combine 回到原 replica、原 token 顺序，shape 与 local MLP 输出一致。
4. padding/phantom rows 不影响 logits。
5. `reduce_results=False`（combine 已聚合，禁止再 TP all-reduce → 否则 double-reduce）。
6. idle replica keepalive 到对端结束（Phase E）。
7. graph replay 下所有 buffer 指针稳定（静态分配，不在 dispatch/combine 内重分配）。

---

## 5. CUDA graph 安全性（核心难点，必须想清楚）

稀疏路由天然是**数据相关的变长**，而 graph 要求静态 shape。两种 transport 的处理：

- **nccl_a2a + graph**：必须 pad 到固定 `M_max`（每个目标 rank 的上界）+ mask。
  - 取舍：`M_max` 设太大 → 退化成 dense（无收益）；设太小 → 溢出丢 token（错）。
  - 方案：`M_max = ceil(capture_max_m × K × peer_fraction × 安全系数)`，并在运行时 assert 实际 count ≤ M_max（超了就标记需要重 capture / 报错）。
  - 静态 buffer 全部在 `__init__`/`_allocate_static_buffers` 里分配，dispatch/combine 内只 `copy_/zero_/fill_`（照抄①的约束）。
- **deepep + graph**：DeepEP LL 用 `num_max_dispatch_tokens_per_rank` + `masked_m` 在 kernel 内处理变长，**这正是 DeepEP 比裸 NCCL 强的地方**——graph 内变长不用 host sync。

> 工程现实：**"既要稀疏、又要 graph、又要不 padding"基本只有 DeepEP 能给**。所以路线是：先用 `nccl_a2a` 在 **eager** 下把 L1/L3 的正确性跑通（bf16，沿用已验证数值），再上 `deepep` transport 拿到 graph 内的真正收益。

---

## 6. 实现里程碑（建议顺序）

| 里程碑 | 内容 | transport | 精度 | graph | 验收 |
|---|---|---|---|---|---|
| **M0** | L1 RoutingPlan 计算 + 单测（对拍 dense 的 union 语义） | — | — | — | 单测：plan 能覆盖所有本 rank 需算的 (t,k) |
| **M1** | nccl_a2a eager 全链路（L1+L2A+L3-triton） | nccl_a2a | bf16/triton | eager | temp=0 输出与 dense 路径一致 |
| **M2** | 接入 `register_balloon_global_runtime_bundle`（新 backend 值，拆掉 FP8 assert，换 PrecisionPolicy） | nccl_a2a | bf16/triton | eager | smoke 端到端跑通，balloon commit→GLOBAL 正确 |
| **M3** | nccl_a2a graph 化（静态 buffer + M_max + mask） | nccl_a2a | bf16/triton | ✅ | capture 成功；输出仍一致 |
| **M4** | deepep transport（bf16 dispatch + triton runner 适配） | deepep | bf16/triton | ✅ | 对拍 M3 输出；stage profiler 看 dispatch/combine 降 |
| **M5** | fp8/deep_gemm 选项打通 + 性能对照 | deepep | fp8/deep_gemm | ✅ | temp=0 质量验证；vs dense 的 ms 收益 |

> M1–M3 先把 **bf16 正确性 + 收益**拿到手（不依赖 NVSHMEM），M4–M5 再上 DeepEP/FP8。这样即使 H20 的 NVSHMEM 短期起不来，也有一条能跑、能省的 bf16 路径。

---

## 7. 文件改动清单（预估）

| 文件 | 改动 |
|---|---|
| `layers/moe/token_dispatcher/kunserve_routed.py`（**新**） | L1 RoutingPlan + 新 dispatcher `CrossReplicaRoutedDispatcher`（dispatch/combine） |
| `layers/moe/token_dispatcher/kunserve_transport.py`（**新**） | L2：`NcclA2ATransport` + `DeepEPTransport` + Protocol |
| `layers/moe/token_dispatcher/kunserve_runner_adapter.py`（**新**） | L3：triton / deep_gemm 适配 + grouped↔sorted 布局转换 |
| `layers/moe/token_dispatcher/kunserve_token_a2a.py` | 抽出可复用的 routing 计算（owner-in-lane 等）到 L1，避免重复 |
| `model_executor/model_runner.py` | `register_balloon_global_runtime_bundle`：新增 `kunserve_comm_backend` 取值（或 PrecisionPolicy 参数），**移除"deepep⇒deep_gemm" 的硬 assert**，改为按策略校验；GLOBAL bundle 用新 dispatcher |
| `layers/moe/fused_moe_triton/layer.py` | `register_runtime_bundle` 支持显式传入新 dispatcher + PrecisionPolicy（已有 explicit_global_dispatcher 通道，扩展即可） |
| `srt/server_args.py` / env | 新增 `KUNSERVE_DISPATCH_DTYPE`(bf16/fp8)、`KUNSERVE_EXPERT_RUNNER`(triton/deep_gemm)、`KUNSERVE_XREP_TRANSPORT`(nccl_a2a/deepep) |
| 文档 | 同步 `kunserve_综合分析与优化文档.md` + CLAUDE.md 状态表 + 本文件 checklist |

---

## 8. 验证与回归

- **正确性（每个里程碑都做）**：temp=0 确定性 + prompt 文本配对，KunServe-GLOBAL 输出与 dense 路径（已验证基线）逐 prompt 对拍，长度比≈1.0、答案一致、无乱码。
- **性能**：复用 `profile_orch.py` + `parse_stage_traces.py`，对照 dense 的 `stages_*.csv`，看 `dispatch_all_gather[KS]` + `combine_reduce_scatter[KS]` 阶段 ms 下降；端到端用 baseline-vs-GLOBAL matched bs+KV 残差（铁律：不看噪声大的端到端）。
- **回归**：`ONLY_RUN=kunserve bash compare_kunserve_vs_baseline.sh` 必须先通过再 claim 完成（CLAUDE.md 规则）。

---

## 9. 风险与未决问题

1. **L3 布局转换**（DeepEP grouped ↔ triton sorted）是新代码最容易出 bug 的地方；M4 要重点单测。
2. **graph 内 M_max 上界**：估太小会丢 token。需要一个"运行时检测溢出 → 触发重 capture 或安全 fallback 到 dense"的兜底。
3. **nccl_a2a graph 化收益有限**（padding），真正收益在 deepep transport；要管理预期。
4. **拆 FP8 assert 的连带影响**：现有 ③ 路径默认行为不能被破坏（保持向后兼容，新行为走新 backend 值/策略）。
5. **Phase E keepalive** 在新 transport 下要重新接（idle replica 不能 hang）。
6. **数值**：fp8 expert（M5）在 30B 上历史无漂移，但仍需 temp=0 重验；80B(linear-attention) 另案。

---

## 10. 一句话路线

**先用 `nccl_a2a + bf16 + triton` 在 eager 下把 token-routed 的正确性与基本收益跑通（M0–M3，不依赖 NVSHMEM/FP8），再切到 `deepep` transport 拿 graph 内的真正低延迟（M4），最后开放 `fp8 + deep_gemm` 作为高性能可选（M5）。全程精度/传输解耦,bf16 是默认一等公民。**
