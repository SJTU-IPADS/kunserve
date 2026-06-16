# DeepEP NORMAL vs LL：逐文件逐函数详细对照

更新时间：2026-06-16
配套：`deepep_architecture.md`（总览）、`deepep_bug_analysis.md`（bug 分析）

本文把 **NORMAL（prefill，M1 正确）** 和 **LOW_LATENCY/LL（decode，M2 乱码）** 两条数据面，从 dispatcher 到 runner 到 combine，逐文件逐函数写清楚，并标注每个**分叉点 D1–D9** 的差异、已做的**验证手段与结果**。模型固定例子见架构文档 §3（128 expert / 32 per rank / hidden=2048 / w13=1536 / w2=768 / topk=8 / fp8）。

---

## 0. 公共前缀（两条路完全一致，已验证非病因）

文件 `models/qwen3_moe.py`：
```
Qwen3MoeSparseMoeBlock.forward (269) ── moe_a2a_backend.is_deepep() → forward_deepep (371)
  gate(hidden) → router_logits
  topk(hidden, router_logits,
       num_token_non_padded=forward_batch.num_token_non_padded,        # padding 行 → topk = -1
       expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(layer_id))  # 静态 remap
  experts(hidden, topk_output)  → FusedMoE / EPMoE
```
- **topk + remap 两条路共用同一函数同一 info**，所以 dispatch id 完全相同。remap 正确性已由 `weight_probe`（4 rank start=0/0/32/32）+ `parity_deepep_ll_remap.sh` 验证。
- **NORMAL/LL 的选择**：`deepep_mode=auto` → decode 用 LL、prefill 用 NORMAL；外加 DR-7 跨实例 `is_extend` 协商（任一 rank prefill → 全 rank NORMAL）。

模式选择器：`token_dispatcher/deepep.py:MaybeTboDeepEPDispatcher`，`num_inner_dispatchers = 2 if is_tbo_enabled() else 1`。**TBO 默认关**（实测 detail log 0 条 tbo 痕迹），`_execute` 直通 `_inners[0]`。→ **D9 TBO 已排除**。

---

## 1. NORMAL 路径（M1，✅ 输出正确）

### 1.1 dispatcher：`_DeepEPDispatcherImplNormal`（deepep.py:518–679）
```
dispatch_a (526):
  topk_weights, topk_ids = topk_output; topk_ids→int64
  hidden = sglang_per_token_group_quant_fp8(hidden, 128, column_major/ue8m0=DEEPGEMM_SCALE_UE8M0)   # ← NORMAL 也 fp8 dispatch
  return (hidden_fp8, topk_ids, topk_weights)
dispatch_b (549) → _dispatch_core (572):
  buffer.get_dispatch_layout(topk_ids, num_experts) → num_tokens_per_rank/expert（动态计数）
  buffer.dispatch(x, topk_idx, topk_weights, num_tokens_per_*, expert_alignment=128, config=normal_dispatch_config)
    → recv_x, recv_topk_ids, recv_topk_weights, num_recv_tokens_per_expert, self.handle
  返回 DeepEPNormalDispatchOutput(hidden, scale, recv_topk_ids, recv_topk_weights, num_recv_tokens_per_expert)
```
特征：**动态 all-to-all**（NVLink），token 按身份路由，recv 端拿到的是「按 expert 排好的 recv token + 对应的 recv_topk_*」。

### 1.2 runner：`moe_runner/deep_gemm.py` — **contiguous** GEMM
```
@register_pre_permute("deepep_normal","deep_gemm")  pre_permute_deepep_normal_to_deep_gemm (611):
  ep_scatter(hidden, scale, topk_ids, num_recv_tokens_per_expert_gpu, → input_tensor, m_indices, output_index)
     # 把 recv token 散布成「每 expert 连续」布局；m_indices[token]=该 token 属哪个 local expert
  DeepGemmRunnerInput(use_masked_gemm=False, m_indices=...)
run (119) → _run_contiguous_gemm (135):
  grouped_gemm_nt_f8f8bf16_CONTIGUOUS((hidden,scale),(w13,w13_scale), gateup, m_indices)
  silu_and_mul + 量化
  grouped_gemm_nt_f8f8bf16_CONTIGUOUS((down_in,scale),(w2,w2_scale), down_out, m_indices)
@register_post_permute("deep_gemm","deepep_normal")  post_permute_deep_gemm_to_deepep_normal (703):
  ep_gather(down_out, topk_ids, topk_weights, output_index, gather_out)   # ★ topk_weights 在 gather 里乘
  DeepEPNormalCombineInput(gather_out, topk_ids, topk_weights)
```

### 1.3 combine：`_combine_core` (655)
```
buffer.combine(x, self.handle)   # 只跨 rank 求和（权重已在 ep_gather 乘过）
```

---

## 2. LL 路径（M2，❌ 输出乱码）

### 2.1 dispatcher：`_DeepEPDispatcherImplLowLatency`（deepep.py:681–989）
```
__init__: self.num_max_dispatch_tokens_per_rank = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK (128)
          self.handle = None
dispatch_a (693):
  topk_ids→int64; expected_m = (n_tok*group_size*topk + num_experts)//num_experts
  → _dispatch_core (784):
      use_fp8 = not SGLANG_DEEPEP_BF16_DISPATCH (H20: True)
      buffer.low_latency_dispatch(hidden, topk_ids, num_max(128), num_experts(128),
            use_fp8=True, round_scale=DEEPGEMM_BLACKWELL(H20:False), use_ue8m0=DEEPGEMM_BLACKWELL(H20:False),
            async_finish=not return_recv_hook, return_recv_hook=...)
      → (packed_recv_x_fp8, packed_recv_x_scales), self.packed_recv_count, self.handle, event, hook
      packed_recv_x: [num_local=32, num_max*num_ranks=512, hidden]  按 expert 分组、front-pack 到 masked_m[g]
      handle = (src_info, layout_range, num_max=128, hidden, num_experts=128)
dispatch_b (719): hook() if return_recv_hook else event.current_stream_wait()  # 等 RDMA 到达
  返回 DeepEPLLDispatchOutput(hidden_fp8, scale, topk_ids, topk_weights, masked_m, expected_m)
```
特征：**固定容量 num_max=128/rank**、NVSHMEM RDMA、按 expert 分组、front-pack。decode batch≪128 不溢出；re-prefill(12012 token)走 NORMAL 不碰 LL。

### 2.2 runner：`moe_runner/deep_gemm.py` — **masked** GEMM（`_run_masked_gemm` 218）
```
@register_pre_permute("deepep_ll","deep_gemm") (569): passthrough，use_masked_gemm=True，masked_m
_run_masked_gemm:
  [scale 处理] DEEPGEMM_SCALE_UE8M0 ? _cast_to_e8m0 : get_mn_major_tma_aligned_tensor(hidden_states_scale)
  grouped_gemm_nt_f8f8bf16_MASKED((hidden,scale),(w13,w13_scale), gateup, masked_m, expected_m)     # GEMM-0
  silu_and_mul_masked_post_quant_fwd(gateup → down_in fp8, down_in_scale,
        column_major_scales=True, scale_tma_aligned=True, fuse_silu_and_mul=True, masked_m)          # act 量化
  get_mn_major_tma_aligned_tensor(down_in_scale)
  grouped_gemm_nt_f8f8bf16_MASKED((down_in,scale),(w2,w2_scale), down_out, masked_m, expected_m)      # GEMM-1
  # down_out = torch.empty → padding 行([masked_m[g],512)) 是 NaN（无害，combine 跳过 -1；ZERO_PAD 实验证清零不修复）
@register_post_permute("deep_gemm","deepep_ll") (595): passthrough → DeepEPLLCombineInput(down_out, topk_ids, topk_weights)
```

### 2.3 combine：`_combine_core` (946) → `low_latency_combine`
```
buffer.low_latency_combine(x=down_out, topk_idx=topk_ids, topk_weights=topk_weights, handle=self.handle,
                           async_finish=not return_recv_hook, return_recv_hook=...)
deep_ep/buffer.py:683: runtime.low_latency_combine(x, topk_idx, topk_weights, src_info, layout_range, ...,
                                                   num_max_dispatch_tokens_per_rank, num_experts, ...)
CUDA kernel internode_ll.cu combine (555):
  for token_idx < num_combined_tokens:
    for k in num_topk:
      topk_idx_reg = topk_idx[token_idx*num_topk + k]
      if (topk_idx_reg < 0) continue;                                   # ← -1 padding 跳过(835/871)
      buffer = rdma_recv_x + (topk_idx_reg * num_max + token_idx) * bytes_per_slot   # 按(全局expert, token)索引
      Σ topk_weight_k · decode_and_accumulate(buffer)                   # 权重在 kernel 内乘 + 跨 rank reduce
```
combine_b (934): 等待 → 返回 combined[n_tok, hidden]。`reduce_results=False`，combine 已聚合 EP world，不再 TP all-reduce。

---

## 3. 分叉点清单 D1–D9 + 验证状态

| ID | 分叉 | NORMAL | LL | 验证手段 | 结果 |
|----|------|--------|----|---------|------|
| D1 | dispatch kernel | `get_dispatch_layout`+`dispatch`（动态） | `low_latency_dispatch`（固定容量/front-pack） | `parity_deepep_ll_remap.sh`（bf16+fp8+TP_REPLICATE 全 PASS） | ✅ 正确 |
| D2 | fp8 dispatch scale | `sglang_per_token_group_quant_fp8` | LL 内部产 scale（H20 round/ue8m0=False） | `[MASKED-REF]` GEMM-0 反量化对拍 ref≈act | ✅ 正确 |
| D3 | pre-permute | `ep_scatter`→连续+m_indices | passthrough（masked_m） | — | 低嫌疑 |
| D4 | GEMM-0 (w13) | contiguous | **masked** | `[MASKED-REF]` ref≈act(~0.002) | ✅ 正确 |
| D5 | act 量化 | NORMAL silu+quant | **`silu_and_mul_masked_post_quant_fwd`** | `[MASKED-REF-DIN]` ref≈act(~fp8噪声 0.005) | ✅ 正确 |
| D6 | GEMM-1 (w2) | contiguous | **masked** | `[MASKED-REF2]` ref≈act(~0.002) | ✅ 正确 |
| D7 | 权重乘的位置 | `ep_gather`（combine 前乘） | `low_latency_combine` 内部乘 | `[LL-CMB]` tw_rowsum=1.0 | ✅ 权重对 |
| D8 | combine kernel | `buffer.combine`（只求和） | **`low_latency_combine`**（权重乘+reduce+ -1 跳过） | parity round-trip PASS + 读 internode_ll.cu(-1 跳过自洽) + `[LL-CMB-OUT]`(非 NaN/量级 sane) | ⚠️ **部分**：聚合量全过,但**对"值对 token 错位"的保范数错误是盲的** |
| D9 | TBO | — | `MaybeTbo` 包装 | 未开,0 痕迹 | ✅ 排除 |

---

## 4. 已排除的非病因（实测）

| 嫌疑 | 排除依据 |
|---|---|
| TBO | `enable_two_batch_overlap` 未开，detail log 0 条 tbo |
| NaN padding（down_out） | `KUNSERVE_ZERO_PAD=1` 清零后仍乱；且 combine kernel 显式 `if topk_idx_reg<0 continue` |
| CUDA graph 在 prefill 上重放 | 所有 forward `can_run=False` 全程 eager，GLOBAL 图从不 capture（§架构 6） |
| 累积发散 | `[MOE-IO]` layer0 残差范数全程稳定 ~0.07，不增长不 NaN |
| **retract / re-prefill** | 实测乱码请求 `prefill_count=1`（**根本没 resume 过**）；re-prefill 走 NORMAL |
| 权重 provenance | `weight_probe` 4 rank start=0/0/32/32 与 layout 一致；跨 rank 校验和相同=副本权重相同 |
| fp8 量化误差 | NORMAL 也 fp8 且正确；`[MASKED-REF*]` 三段误差都在 fp8 噪声级 |

---

## 5. 验证手段（探针）一览（都在 `KUNSERVE_DETAIL_LOG` 门控）

| 探针 | 文件 | 内容 |
|------|------|------|
| `[MASKED-REF]` | deep_gemm.py | GEMM-0 反量化 bf16 参考 vs kernel gateup（D4） |
| `[MASKED-REF-DIN]` | deep_gemm.py | silu·mul 后 down_in 反量化 vs bf16 参考（D5） |
| `[MASKED-REF2]` | deep_gemm.py | gateup→silu→@w2 全程 bf16 参考 vs down_out（D5+D6） |
| `[MASKED-GEMM]` | deep_gemm.py | masked_m / 形状 / valid-row NaN / down 均值 |
| `[LL-CMB]` | deepep.py | combine 输入 topk_min/max、tw_rowsum、x_in_nan、recv_count（D8 输入） |
| `[LL-CMB-OUT]` | deepep.py | combine 输出 范数/NaN（D8 输出） |
| `[MOE-IO]` | qwen3_moe.py | layer0 MoE 输入/输出范数（跨步,含 re-prefill）；capture 期跳过 |
| `[RETRACT]/[RESUME-PREFILL]` | scheduler.py | 抢占/重 prefill 事件（rid+长度） |
| `weight_probe` | layer.py | 各 rank 权重 narrow 的 start/校验和 |

> **方法论教训**：上面所有"反量化对拍"都用 **kernel 自己的输入** 算参考，所以是 **input-faithful**——上游/路由若错，"垃圾进垃圾出"也会 ref≈act 通过。聚合范数（MOE-IO/LL-CMB-OUT）同理对"保范数 token 错位"盲。这是为什么 D1–D8 看似"全对"却仍乱码的根本原因，详见 `deepep_bug_analysis.md`。
