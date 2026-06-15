# DeepEP GLOBAL MoE:LL vs NORMAL 完整代码路径追踪 + 打点计划

目的:M1(NORMAL)输出**完全正确**,M2(LL)输出**乱码**(同 prompt、同模型、同 fp8)。bug 必在两条路径的**分叉**处。本文逐文件逐函数把两条路走一遍,标出每个分叉点(D1…Dn),并规划在分叉处打 log / 数值对拍。原则:从已验证正确(NORMAL)出发,对每个 LL 独有步骤做"反量化 bf16 参考对拍",定位第一个偏离点。

模型:Qwen3-30B-A3B,128 physical experts,4 GLOBAL EP rank(2 replica×TP2),每 rank 32 local experts,hidden=2048,intermediate=768(w13=2×768=1536),topk=8,fp8 block(128×128)量化,deep_gemm runner。

---

## 0. 共同前缀(两条路完全一致,非嫌疑)

```
scheduler.event_loop_overlap → run_batch → tp_worker.forward_batch_generation
 → model_runner.forward → _forward_raw
   → [DR-7] _negotiate_balloon_deepep_is_extend(forward_batch)   # 决定本 forward 用 NORMAL 还是 LL
 → model.forward → qwen2_moe.py:665 layer() → qwen3_moe.py:901 self.mlp()
 → qwen3_moe.py:269 Qwen3MoeSparseMoeBlock.forward
   → 因 moe_a2a_backend.is_deepep() → 285 forward_deepep(hidden_states, forward_batch)
     → 376 self.gate(hidden_states) → router_logits          # 两路一致
     → 377 self.topk(hidden_states, router_logits,
              num_token_non_padded=forward_batch.num_token_non_padded,
              expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(layer_id))
              # topk + 静态 remap(logical→physical),两路一致(已验证 remap 正确)
     → 387 self.experts(hidden_states, topk_output)           # FusedMoE/EPMoE
       → ep_moe/layer.py:174 forward → 183 forward_impl → fused_moe_triton/layer.py:1440
         → self.dispatcher.dispatch(...)   # ←★ 第一个大分叉:dispatcher impl
```

**D0(模式选择)**:`deepep_mode=auto` 下,decode→LOW_LATENCY,prefill→NORMAL。`MaybeTboDeepEPDispatcher` 选 `_DeepEPDispatcherImplNormal` 或 `_DeepEPDispatcherImplLowLatency`。注意 trace 里 **TBO 开着**(`two_batch_overlap.py` 出现在栈上)——`num_inner_dispatchers = 2 if is_tbo_enabled()`,需确认 TBO 是否引入额外分叉。

---

## 1. NORMAL 路径(M1,✅ 正确)

文件:`token_dispatcher/deepep.py` `_DeepEPDispatcherImplNormal`(518–679)

```
dispatch_a (526):
  - sglang_per_token_group_quant_fp8(hidden, 128, column_major=UE8M0, ue8m0=UE8M0)   # fp8 量化
  - return (hidden_fp8, topk_ids, topk_weights)
dispatch_b (549) → _dispatch_core (572):
  - buffer.get_dispatch_layout(topk_ids, num_experts)  → num_tokens_per_rank/expert
  - buffer.dispatch(x, topk_idx, topk_weights, num_tokens_per_*..., expert_alignment=128)
    → recv_x, recv_topk_ids, recv_topk_weights, num_recv_tokens_per_expert, handle
  - 返回 DeepEPNormalDispatchOutput(hidden, scale, recv_topk_ids, recv_topk_weights, num_recv_tokens_per_expert)
```
runner(`moe_runner/deep_gemm.py`):
```
pre_permute_deepep_normal_to_deep_gemm (611):
  - ep_scatter(hidden, scale, topk_ids, num_recv_tokens_per_expert_gpu, ... input_tensor, m_indices, output_index)
    # 按 expert 把 recv token 散布成连续布局,得 m_indices(每 token 属哪 expert)
  - DeepGemmRunnerInput(use_masked_gemm=False, m_indices=...)
run (119) → _run_contiguous_gemm (135):
  - grouped_gemm_nt_f8f8bf16_CONTIGUOUS((hidden,scale),(w13,w13_scale), gateup, m_indices)
  - silu_and_mul(...) + 量化   ← NORMAL 的 act 量化路径
  - grouped_gemm_nt_f8f8bf16_CONTIGUOUS((down_in,scale),(w2,w2_scale), down_out, m_indices)
post_permute_deep_gemm_to_deepep_normal (703):
  - ep_gather(down_out, topk_ids, topk_weights, output_index, gather_out)
    # ←★ 权重 topk_weights 在 gather 里乘上(combine 之前)
  - DeepEPNormalCombineInput(gather_out, topk_ids, topk_weights)
combine_a (633)/_combine_core (655):
  - buffer.combine(x, handle)    # ←★ 不传 topk_weights(已在 ep_gather 乘过),只跨 rank 求和
```

---

## 2. LL 路径(M2,❌ 乱码)

文件:`token_dispatcher/deepep.py` `_DeepEPDispatcherImplLowLatency`(681–989)

```
dispatch_a (693) → _dispatch_core (784):
  - use_fp8 = not SGLANG_DEEPEP_BF16_DISPATCH   (H20: True)
  - buffer.low_latency_dispatch(hidden, topk_ids, num_max_dispatch_tokens_per_rank, num_experts,
        use_fp8=True, round_scale=DEEPGEMM_BLACKWELL(H20:False), use_ue8m0=DEEPGEMM_BLACKWELL(H20:False))
    → (packed_recv_x_fp8, packed_recv_x_scales), packed_recv_count, handle
    # ←★ 固定容量 [num_local, num_max*num_ranks, hidden],按 expert 分组,front-pack 到 masked_m
  - expected_m = (n_tok*group_size*topk + num_experts)//num_experts
  - return DeepEPLLDispatchOutput(hidden_fp8, scale, topk_ids, topk_weights, masked_m, expected_m)
```
runner:
```
pre_permute_deepep_ll_to_deep_gemm (569):  passthrough,use_masked_gemm=True,masked_m
run (119) → _run_masked_gemm (218):
  - [若 UE8M0] _cast_to_e8m0_with_rounding_up(scale) [否则] get_mn_major_tma_aligned_tensor(scale)  ←★ D-act-scale
  - grouped_gemm_nt_f8f8bf16_MASKED((hidden,scale),(w13,w13_scale), gateup, masked_m, expected_m)   # GEMM-0 ✅已验证算术正确(MASKED-REF)
  - silu_and_mul_masked_post_quant_fwd(gateup, down_in, down_in_scale, 128, masked_m,
        column_major_scales=True, scale_tma_aligned=True, fuse_silu_and_mul=True)   # ←★ D-act:LL 独有 act+量化,未验证
  - get_mn_major_tma_aligned_tensor(down_in_scale)
  - grouped_gemm_nt_f8f8bf16_MASKED((down_in,scale),(w2,w2_scale), down_out, masked_m, expected_m)   # ←★ D-gemm1:GEMM-1,未验证
  - [down_out padding 行 = torch.empty = NaN;ZERO_PAD 实验已证清零不修复]
post_permute_deep_gemm_to_deepep_ll (595):  passthrough(hidden, topk_ids, topk_weights)  ←★ 权重 NOT 在此乘
combine_a (895)/_combine_core (946):
  - buffer.low_latency_combine(x, topk_idx=topk_ids, topk_weights=topk_weights, handle)
    # ←★ D-combine:权重在 combine 内部乘 + 跨 rank reduce(NORMAL 是 ep_gather 乘+buffer.combine 求和)
```

---

## 3. 分叉点清单(LL 独有 / 与 NORMAL 不同)+ 嫌疑度 + 打点计划

| ID | 分叉点 | NORMAL | LL | 已验证? | 打点/对拍 |
|----|--------|--------|----|---------|-----------|
| D1 | dispatch kernel | get_dispatch_layout+dispatch(动态、token 身份路由) | low_latency_dispatch(固定容量、masked、front-pack) | 对拍 PASS(parity_deepep_ll_remap) | 已有 [LL-CORE]/[LL-ANOM] |
| D2 | fp8 dispatch scale | column_major/ue8m0 由 sglang_per_token_group_quant_fp8 | low_latency_dispatch 内部产 scale(H20 round_scale/ue8m0=False) | GEMM-0 对拍 PASS=scale 消费对 | 已有 [MASKED-REF] gemm0 |
| D3 | pre-permute | ep_scatter→连续+m_indices | passthrough(masked_m) | — | 低 |
| D4 | GEMM-0 (w13) | contiguous(m_indices) | **masked**(masked_m) | ✅ MASKED-REF 算术正确 | 已有 |
| **D5** | **act 量化** | NORMAL silu_and_mul+quant(contiguous) | **silu_and_mul_masked_post_quant_fwd**(column_major,fuse,masked) | ❌**未验证** | **新:对拍 down_in = dequant?** |
| **D6** | **GEMM-1 (w2)** | contiguous | **masked** | ❌**未验证** | **新:GEMM-1 反量化对拍** |
| D7 | post-permute | ep_gather(**乘 topk_weights**) | passthrough(不乘) | — | 中 |
| **D8** | **combine + 权重** | buffer.combine(只求和,权重已乘) | **low_latency_combine(权重内部乘+reduce)** | combine 机制对拍 PASS,但**真实 topk_weights/对齐未端到端验证** | **新:对拍 combine 后单 token 输出 vs 手算** |
| D9 | TBO | ? | two_batch_overlap 包了 dispatch/combine | ❌未查 | 看 is_tbo_enabled + 子 batch 切分 |

**结论(打点优先级)**:D4(GEMM-0)已证对 → 误差必在其**后**:**D5(act 量化)→ D6(GEMM-1)→ D8(combine 权重/对齐)**。逐级反量化对拍,找第一个 ref≠act 的层。

### 端到端对拍思路(最强定位)
在 `_run_masked_gemm` 内,用本 rank 已有的 fp8 张量,**全程 bf16 反量化**算一份参考:
```
ref_gateup = dequant(hidden_fp8, hsc) @ dequant(w13, w13_scale)            # D4 已证 ≈ kernel gateup
ref_down_in = silu(ref_gateup[:768]) * ref_gateup[768:]                     # D5 参考(bf16,不量化)
ref_down = ref_down_in @ dequant(w2, w2_scale)                              # D6 参考
diff(ref_down[valid], down_output[valid])   # 若大 → D5/D6 是真凶(LL masked act/gemm1 数值错)
```
若 D6 也 ≈,则误差在 **D8 combine**:对拍 combine 后某 token 的 MoE 输出 = Σ_k topk_weight_k · (该 token 第 k 个 expert 的 down_out)。

---

## 4. 打点记录(随迭代更新)

- ✅已加:**[MASKED-REF2]**(`_run_masked_gemm`):对 (g,t=0) 算全程 bf16 参考 `gateup→silu_and_mul→@w2`,GEMM-1 后比 `down_output[g,0]`。判读:
  - `down_max_abs_diff` 大 → **D5(act 量化)或 D6(GEMM-1)是真凶**(LL masked 路径数值错)。
  - `down_max_abs_diff` 小(只 fp8 噪声) → D4–D6 全对 → 真凶在 **D8 combine**(权重/对齐/reduce),下一步对拍 combine 后单 token 输出。
- 已有:[LL-CORE]/[LL-ANOM]/[LL-DISP](deepep.py)、[MASKED-GEMM](deep_gemm.py)、weight_probe(layer.py)。

### 一次跑全覆盖的对拍探针(KUNSERVE_DETAIL_LOG 门控,无需额外 env)
| 探针 | 文件 | 分叉 | 判读 |
|------|------|------|------|
| `[MASKED-REF]` gemm0 | deep_gemm.py | D4 w13 GEMM-0 | diff 大 → GEMM-0 错(已知:小,正确) |
| `[MASKED-REF-DIN]` | deep_gemm.py | **D5** silu_and_mul_masked_post_quant | din_max_abs_diff 大 → **act 量化错** |
| `[MASKED-REF2]` | deep_gemm.py | **D6** w2 masked GEMM-1 | D5 小但 down_max_abs_diff 大 → **GEMM-1 错** |
| `[LL-CMB]` | deepep.py | **D8** combine 输入 | tw_rowsum_mean≠1 / x_in_nan → 权重/输入错 |
| `[LL-CMB-OUT]` | deepep.py | **D8** combine 输出 | 前面都小但 out 乱 → **combine 内部(权重乘/reduce/对齐)错** |

定位逻辑(从前往后第一个"diff 大"即真凶):D4→D5→D6→D8。全小则问题在 runner/combine 之外(残差、attention、采样)。

---

## 5. 【2026-06-15 重大转向】乱码 = retract + re-prefill 触发,不是稳态 MoE 数值

排查结论(基于完整 run "kunserve copy"):
- **纯 LOCAL 请求连贯**(balloon=0 → finish=stop);乱码 100% 在 balloon 请求。
- **乱码与 retract+re-prefill 完美相关**:length(乱码)请求 avg_retract_ms=736s、prefill_count=1.53;stop/abort(连贯)请求 retract=0、prefill=1。
- **D4–D8 全部数值正确**——但这是因为 GEMM 对拍**用 kernel 自己的输入算 ref**,"垃圾进垃圾出"也匹配,所以**测不出上游/映射/累积损坏**。
- **eager 回退问题已澄清**:`forward_select` 日志显示**所有** forward 都 `can_run=False, replay_enabled=False`(GLOBAL 图从没捕,captured_variants=['local'] 但 balloon 都是 variant=global)→ **全程 eager**,包括 22 个 `mode=1`(EXTEND/re-prefill,input_tokens≈12012)。`force_eager` 标志虽恒 False,但 eager 已由 can_run=False 达成。**所以"在 prefill 上重放图"不存在,graph/eager 排除。**
- **真凶定位**:retract→re-prefill 在 **balloon GLOBAL 态**的正确性(疑:扩展 KV 池 slot,或 GLOBAL collective 处理"一副本 re-prefill 12012 token、另一副本 decode/idle"的非对称 forward)。

### 新探针(下次跑全在 KUNSERVE_DETAIL_LOG)
| 探针 | 文件 | 作用 |
|------|------|------|
| `[RETRACT]` | scheduler.py | retract_decode 触发:n、rid+已生成长度、kv_full |
| `[RESUME-PREFILL]` | scheduler.py | prefill batch 含被 retract 过的请求:rid、outlen、retract 次数 |
| `[MOE-IO]` | qwen3_moe.py | layer0 MoE 输入/输出范数,**必记 re-prefill(n_tok>500)+ 其后 5 步 + 1/20 采样** |

### 判读(下次跑)
按时间戳排 [RETRACT]→[RESUME-PREFILL]→[MOE-IO]:
- 若某请求 re-prefill(MOE-IO n_tok 大)**之后**的 decode `in_abs_mean` 突然爆炸 → **实锤 re-prefill 损坏状态**(去查 balloon KV slot / 非对称 forward)。
- 若范数平滑、无跳变 → 回到累积假设。
