# KunServe × DeepEP 架构总览（NORMAL + LL）

更新时间：2026-06-16
适用分支：`/workspace/sglang-deepep`，`KUNSERVE_COMM_BACKEND=deepep`

> 本文是 **DeepEP 后端**（`moe_a2a_backend=deepep` + `moe_runner_backend=deep_gemm` + fp8）的架构总览，结构对齐 `/workspace/kunserve_architecture.md`（那份是 **sglang/dense 后端**：PyNccl registered collective + lane all-gather + Triton + post-commit fixed_padded graph）。
> DeepEP 后端有两条数据面：**NORMAL**（prefill，`buffer.dispatch`）和 **LOW_LATENCY / LL**（decode，`low_latency_dispatch`，NVSHMEM）。M1 = 强制 GLOBAL NORMAL（`SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL=1`，**实测全对**）；M2 = AUTO（decode 走 LL，`...=0`，**实测乱码**，bug 见 `deepep_bug_analysis.md`）。
> 两条数据面的逐文件逐函数细节见 `deepep_normal_vs_ll_detail.md`。

目录：
1. [KunServe 是什么 / 解决什么问题](#1)
2. [整体架构图](#2)
3. [固定例子：30B 两个 TP=2 实例，128 experts](#3)
4. [数据面：一层 decode 的完整 forward 链路](#4)
5. [控制面：balloon 状态机与三个进程组](#5)
6. [GLOBAL CUDA graph 链路（为什么 deepep 不 capture）](#6)

---

<a id="1"></a>
## 1. KunServe 是什么 / 解决什么问题

KunServe 是 verl GRPO rollout 用 SGLang 生成时的一个 **KV 显存弹性方案**。当 KV cache 显存吃紧、请求开始排队，与其让请求长期等待，不如让**每个 replica 把自己一半的 MoE expert 权重（VMM physical pages）临时借给 KV cache**，从而扩大可用 KV slots、把排队的请求放进来继续 decode。

被借走（offload）的 experts 不再本地计算，而是通过**跨 replica expert sharing** 补齐：本 replica 只算自己保留的 expert，缺的 expert 由 peer replica 算，再把结果交换回来。

核心约束（与 dense 后端一致）：
- **Attention / KV cache 仍归属原 replica**（不跨实例迁移）。
- 每个 replica 内部仍是原生 `TP=2` attention + local TP all-reduce。
- **只有 MoE expert 计算跨 replica 补齐。**
- BALLOON 后请求**不能乱码**；最后一个 replica drain 时**不能 hang**。
- 性能尽量接近原生单实例 `TP=4`。

**DeepEP 后端与 dense 后端的区别**：dense 后端用自建 PyNccl + lane all-gather + Triton + bf16；DeepEP 后端用 **DeepEP 的 all-to-all（NORMAL）/ NVSHMEM RDMA（LL）+ deep_gemm grouped GEMM + fp8**，理论上跨实例传输更省、decode 更快——这正是要打通 LL 的动机（NORMAL vs LL 性能差距大）。

---

<a id="2"></a>
## 2. 整体架构图

```text
verl / Ray
  AgentLoopManager
    KunServeController（kunserve_manager/controller.py）
      - 初始化 runtime_group（跨实例 EP=4 group）
      - warmup_balloon: 注册 GLOBAL bundle（DeepEP dispatcher），NVSHMEM 预热
      - poll /kunserve/status，发现 expand_requested=True → enter BALLOON
      - prepare_balloon / commit_balloon

SGLang replica 0  (TP=2, EP=2)            SGLang replica 1  (TP=2, EP=2)
  scheduler rank 0/1                         scheduler rank 2/3
  model_runner rank 0/1                      model_runner rank 2/3
    FusedMoE.local_bundle  (LOCAL)             FusedMoE.local_bundle  (LOCAL)
    FusedMoE.global_bundle (GLOBAL/DeepEP)     FusedMoE.global_bundle (GLOBAL/DeepEP)

跨实例 EP=4 数据面（balloon 后）:
  GLOBAL rank 0,1,2,3 = runtime_group
    - NORMAL: buffer.dispatch（NVLink all-to-all，prefill）
    - LL:     low_latency_dispatch（NVSHMEM RDMA put/get，decode）
  专家分布（见 §3）跨两个 replica，128 experts 仍完整。
```

与 dense 后端**最大的结构差异**：
- dense 后端 GLOBAL 通信走 **lane group**（replica0_lane_i + replica1_lane_i 的 2-rank 子组）+ all-gather/reduce-scatter；
- **DeepEP 后端 GLOBAL 通信走整个 `runtime_group`（EP=4，4 rank）**，由 DeepEP Buffer 内部的 all-to-all（NORMAL）或 NVSHMEM（LL）完成，**没有 lane group**。
- dense 后端 GLOBAL **capture CUDA graph**；DeepEP 后端 GLOBAL **永远 eager**（见 §6）。

---

<a id="3"></a>
## 3. 固定例子：30B 两个 TP=2 实例，128 experts

模型 Qwen3-30B-A3B：128 个 routed experts，hidden=2048，moe_intermediate=768（w13=gate+up=2×768=1536，w2=768→2048），topk=8。两个 replica，每个 `TP=EP=2`，共 4 GPU。

balloon 前：每个 replica 是完整模型，EP=2 → 每张卡持 64 个 expert（rank0/2 持 physical 0–63，rank1/3 持 physical 64–127）。

**balloon 后**每张卡只保留 32 个 expert（offload 另 32 个），GLOBAL EP=4 的专家归属（**非连续**）：

| GLOBAL rank | replica | EP lane | 保留 physical experts | VMM 张量里的 narrow（start） |
| --- | --- | --- | --- | --- |
| 0 | replica 0 | lane 0 | **0..31**   | rank0 原持 0–63 → 留**前缀** start=0 |
| 1 | replica 0 | lane 1 | **64..95**  | rank1 原持 64–127 → 留**前缀** start=0 |
| 2 | replica 1 | lane 0 | **32..63**  | rank2 原持 0–63 → 留**后缀** start=32 |
| 3 | replica 1 | lane 1 | **96..127** | rank3 原持 64–127 → 留**后缀** start=32 |

（来源：`kunserve_manager/layout.py:build_complementary_physical_to_logical_map`；weight_probe 实测 start=0/0/32/32 与此一致。）

**静态 remap**（`ep_dispatch_algorithm=static`，`logical_to_rank_dispatch_physical_map`）把 logical expert id 映射成 **dispatch id**，使 `dispatch_id // 32 == owner_rank`，于是 DeepEP 把 token 路由到正确的 rank。实测 layer0：`logical [0,32,64,96] → dispatch [0,64,32,96]`，即 GLOBAL 顺序 = `[phys0–31, phys64–95, phys32–63, phys96–127]`。

> 这一整套（非连续 ownership + remap + 权重 narrow）已被数值对拍 `verl/data/parity_deepep_ll_remap.sh` 和真实 run 的 `weight_probe` 验证**正确**。

---

<a id="4"></a>
## 4. 数据面：一层 decode 的完整 forward 链路

一层 MoE 在 balloon-GLOBAL 下的完整链路（公共前缀两条路一致，分叉点见 §「★」）：

```text
scheduler.event_loop_overlap → run_batch → tp_worker.forward_batch_generation
 → model_runner.forward → _forward_raw
   → [DR-7] _negotiate_balloon_deepep_is_extend(forward_batch)
        在 runtime_group 上 all-gather 各 rank 的 is_extend，OR 起来：
        任一 rank 是 prefill(EXTEND) → 本 step 全 rank 用 NORMAL；全 decode → 用 LL
 → model.forward → qwen2_moe.py layer() → qwen3_moe.py:901 self.mlp()
 → qwen3_moe.py:269 Qwen3MoeSparseMoeBlock.forward
   → moe_a2a_backend.is_deepep() == True → forward_deepep(hidden, forward_batch)
     → self.gate(hidden) → router_logits
     → self.topk(hidden, router_logits,
            num_token_non_padded=forward_batch.num_token_non_padded,   # padding 行 topk→ -1
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(layer_id))  # 静态 remap
     → self.experts(hidden, topk_output)   # FusedMoE/EPMoE
       → ep_moe/layer.py forward_impl → fused_moe_triton/layer.py:1440 dispatcher.dispatch(...)
         ──────────────── ★ 模式分叉 ────────────────
```

**decode（LL）一层的链路**（M2，乱码路径）：

```text
DeepEP LL 数据面（_DeepEPDispatcherImplLowLatency）：
 1. dispatch_a: topk_ids(int64) → _dispatch_core:
       buffer.low_latency_dispatch(hidden_fp8, topk_ids, num_max=128, num_experts=128,
                                   use_fp8=True, async_finish=True)
       → 返回 packed_recv_x[32, num_max*4=512, hidden] (按 expert 分组, front-pack 到 masked_m)
       → self.handle = (src_info, layout_range, num_max, hidden, num_experts)
 2. dispatch_b: hook()/event 等 RDMA 到达
 3. runner（deep_gemm masked）：
       grouped_gemm_nt_f8f8bf16_MASKED((hidden,scale),(w13,w13_scale), gateup, masked_m)   # GEMM-0
       silu_and_mul_masked_post_quant_fwd(gateup → down_in fp8)                              # act+量化
       grouped_gemm_nt_f8f8bf16_MASKED((down_in,scale),(w2,w2_scale), down_out, masked_m)    # GEMM-1
 4. combine_a/_combine_core:
       buffer.low_latency_combine(down_out, topk_idx=topk_ids, topk_weights, self.handle)
       → kernel 内部: 对每个 token, Σ_k topk_weight_k · down_out[dispatch_id_k 槽], 跨 rank reduce
       → topk_idx_reg < 0(padding) 的槽 continue 跳过(internode_ll.cu:835/871)
 5. combine_b: 等待 → 返回 combined[n_tok, hidden]
 → forward_deepep 返回 final_hidden_states（reduce_results=False，combine 已聚合）
```

**prefill（NORMAL）一层的链路**（M1，正确路径）：

```text
DeepEP NORMAL 数据面（_DeepEPDispatcherImplNormal）：
 1. dispatch_a: sglang_per_token_group_quant_fp8(hidden) → fp8
 2. _dispatch_core: get_dispatch_layout(topk_ids) + buffer.dispatch(...)   # 动态 all-to-all
       → recv_x, recv_topk_ids, recv_topk_weights, num_recv_tokens_per_expert, handle
 3. runner（deep_gemm contiguous）:
       ep_scatter(recv 按 expert 散成连续) → m_indices
       grouped_gemm_nt_f8f8bf16_CONTIGUOUS(..., m_indices)   # GEMM-0
       silu_and_mul + quant
       grouped_gemm_nt_f8f8bf16_CONTIGUOUS(..., m_indices)   # GEMM-1
 4. post_permute: ep_gather(down_out, topk_ids, topk_weights, output_index)  # ★ 权重在此乘
 5. combine: buffer.combine(x, handle)   # 只跨 rank 求和(权重已乘)
```

**关键分叉**（详见 detail 文档 D1–D8）：dispatch kernel、是否 front-pack、**masked vs contiguous GEMM**、act 量化函数、**权重乘的位置**（LL 在 combine 内、NORMAL 在 ep_gather）、combine kernel。

> attention/KV 部分两条路完全一致：balloon 后 attention 仍读本 replica 的（已扩展的）KV pool，与 MoE 后端无关。纯 LOCAL（balloon 前）请求实测正确，说明 attention/KV 没问题。

---

<a id="5"></a>
## 5. 控制面：balloon 状态机与三个进程组

### 状态机（model_runner `_balloon_runtime_variant`）

```text
 "local"  ──warmup_balloon──▶ 注册 global_bundle(DeepEP dispatcher)，NVSHMEM 预热
    │
    │  controller 发现 expand_requested
    ▼
 prepare_balloon  ── 确认 bundle / runtime_group / layout
    │
    ▼
 commit_balloon   ── VMM borrow(尾部 32 expert 的 pages) → remap 进 KV region → KV 扩容
    │              ── switch_runtime_bundle("global")：dispatcher/runner/active_mapping/reduce_results 切换
    ▼
 "global"  ── 进入 GLOBAL 数据面；每个 forward 先 DR-7 协商 NORMAL/LL，再 Phase-E 协商 bs/eager
```

DeepEP 特有：`SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL=0` 时，commit 还会把 **LOCAL bundle 强制换成 StandardDispatcher+Triton**（`_maybe_force_local_standard_dispatcher`，model_runner.py:771），避免 LOCAL 也建 NVSHMEM buffer 造成 double-init。所以 M2 里：LOCAL=StandardDispatcher（bf16/triton，实测正确），GLOBAL=LL（fp8/deep_gemm，乱码）。

### 三个进程组（DeepEP 后端）

| 组 | 成员 | 用途 |
| --- | --- | --- |
| **local TP group** | 各 replica 内 2 rank（{0,1} / {2,3}） | SGLang 原生 attention TP all-reduce；router logits all-reduce |
| **runtime_group** | 跨实例 EP=4（{0,1,2,3}） | ① DeepEP dispatch/combine 的 host 侧 PG；② DR-7 `is_extend` 协商；③ Phase-E `negotiate_balloon_step_bs`（bs/force_eager） |
| **DeepEP NVSHMEM team** | 在 runtime_group 上建（`Buffer(low_latency_mode=True)`） | LL 的 device 侧 RDMA put/get；NORMAL 的 `buffer.dispatch` all-to-all。需 `NVSHMEM_DISABLE_NCCL=1` 避免与宿主 NCCL team 冲突死锁 |

（对比 dense 后端：runtime_group 相同，但 dense 用 **lane group ×2** 做 GLOBAL 通信、不建 NVSHMEM；DeepEP 不用 lane、改用 runtime_group + NVSHMEM。）

### Phase E（非对称负载 keepalive + cached negotiate）

GLOBAL 是 lockstep collective：一个 replica 先 drain 完所有请求后**不能停**，必须发 `ForwardMode.IDLE` keepalive batch（`reqs=[]`、shape 按协商 `target_bs`、padding 写 `dummy_kv_slot`、attention 读 `phantom_req_idx`），否则 peer 的 collective 挂。
逐 step 协商太贵 → `KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=256`：256 步只协商 1 次、其余复用缓存；本地出现 EXTEND/mixed prefill 时所有 rank force eager（但见 §6：deepep GLOBAL 本就 eager，此标志对 GLOBAL 无实际作用）。

---

<a id="6"></a>
## 6. GLOBAL CUDA graph 链路（为什么 deepep 不 capture）

**结论：DeepEP 后端的 GLOBAL forward 永远 eager，从不 capture CUDA graph。** 实测 `forward_select` 日志：所有 balloon forward `variant=global, can_run=False, replay_enabled=False`，`captured_variants=['local']`。

代码依据 `model_runner.py:2046`（`_capture_global_cuda_graph_after_balloon_commit`）：
```python
if backend_lower != "sglang" or policy_lower != "fixed_padded":
    # recapture skipped → return existing(=False for deepep)
```
即 **只有 `comm_backend=="sglang"` 且 `capture_policy=="fixed_padded"` 才 post-commit capture GLOBAL graph**；`comm_backend=="deepep"` 直接 skip。

为什么 deepep 不 capture：
1. **LL 的 NVSHMEM 通信在 graph 内 capture 困难**：`low_latency_dispatch` 的 RDMA put/get + recv-hook 在 stream capture 下受限（参考：探针里 `.item()` 在 capture 期直接报 `operation not permitted when stream is capturing`）。
2. **DeepEP 动态 layout**：NORMAL 的 `get_dispatch_layout`/`buffer.dispatch` 是数据相关的动态 token 计数，与固定 shape 的 graph 不兼容（SGLang 原生也会因 `deepep_mode=normal` 关 graph）。
3. dense 后端之所以能 capture，是因为它用 **fixed_padded 静态 buffer + all_gather_into_tensor**（graph-safe），DeepEP 没有这套。

**影响**：
- LOCAL graph 照常 capture/replay（balloon 前 throughput 高，~500+）。
- GLOBAL（balloon 后）全程 eager → 每 step 有 Python/launch 开销，吞吐低于 dense+graph 与原生 TP=4。这也是为什么 `KUNSERVE_CAPTURE_POLICY` 在 deepep 分支被设为 `disabled`、不起作用。
- Phase-E 的 `force_eager` 标志对 GLOBAL 是 no-op（已经 eager）；它原本是为 dense 后端的 GLOBAL graph replay 服务的。

> 历史教训（来自 dense 后端）：**pre-capture**（commit 前 capture）会导致 balloon 后乱码，因为 capture 记录的 pointer/DAG 在 `commit_balloon` 后（expert/KV VMM remap、allocator 容量、dummy KV slot）全变了。所以 dense 后端改成 **post-commit capture**。DeepEP 后端目前干脆不 capture GLOBAL。
