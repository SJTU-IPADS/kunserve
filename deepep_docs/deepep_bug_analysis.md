# DeepEP LL GLOBAL decode 乱码：Bug 分析

更新时间：2026-06-16
配套：`deepep_architecture.md`、`deepep_normal_vs_ll_detail.md`

## 0. 现象与硬事实

- **M1（GLOBAL NORMAL，`SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL=1`）= 同 prompt 完全正确。**
- **M2（GLOBAL LL，`...=0`，decode 走 `low_latency_dispatch`）= 乱码**：response 从很靠前就是 `1.1.1.1` / `the the the` / `000` 这类重复垃圾，finish=length 顶满 32K 不停。
- **纯 LOCAL（balloon 前）请求正确**（finish=stop）→ attention/KV/采样没问题，病在 **balloon 后的 GLOBAL LL MoE**。
- 乱码**广泛**：~95% 经 balloon 的请求都 length；少数 stop 的是很快结束的短请求。
- 乱码**与 retract 无关**：实测乱码请求 `prefill_count=1`（没 resume 过）。
- 乱码时**残差范数稳定**（`[MOE-IO]` layer0 in≈0.07 全程不变不 NaN），**combine 输出量级 sane、非 NaN**。

→ 病的性质：**"值对、量级对，但 token 内容错位"的保范数错误**，沿 48 层 / 自回归累积成垃圾。

## 1. 已验证正确 / 已排除（不要再查）

数值对拍 + 真实 run 探针逐一确认（细节见 detail 文档 D1–D9、§4）：
- ✅ D1 dispatch、D2 dispatch-scale、D4 GEMM-0、D5 act 量化、D6 GEMM-1、D7 权重位置 —— 全部 ref≈act（fp8 噪声级）。
- ✅ D8 combine 的 -1 处理（读 `internode_ll.cu:835/871` 显式 `if topk_idx_reg<0 continue`）、tw_rowsum=1、输出非 NaN。
- ✅ 权重 provenance（weight_probe start=0/0/32/32）、remap（layer0 (0,32,64,96)→(0,64,32,96)）。
- ❌ 排除：TBO、CUDA graph（全程 eager，GLOBAL 从不 capture）、NaN-padding（ZERO_PAD 清零仍乱）、累积发散（范数稳）、retract/re-prefill（乱码请求没 resume）、fp8 量化误差。

## 2. 为什么"全验证正确却仍乱码"——根本方法论缺陷

**所有"反量化对拍"都用 kernel 自己的输入算参考**（`[MASKED-REF*]` 用 kernel 拿到的 `hidden_states`/`w13` 算 ref，再比 kernel 输出）。这是 **input-faithful**：
> 如果上游把 **token i 的数据/路由送错了**，kernel 仍会**忠实地**对"错误输入"计算，ref 和 act **都是同一份错误结果**，对拍照样 PASS。

同理 `[MOE-IO]`/`[LL-CMB-OUT]` 是**聚合范数**，对"把 23 个 token 的输出做一次保范数置换"**完全盲**（置换不改范数）。

**所以 D1–D8"全对" + 输出"全乱"可以同时成立**——当且仅当 bug 是一个 **token 级的映射/路由错位**，它既被 input-faithful 对拍漏掉，又被聚合范数漏掉。`parity_deepep_ll_remap.sh` 用**合成数据 + 合成 +e 专家**验证了 dispatch/combine 的**机制**，但没复现真实 run 的若干条件（见 §4）。

## 3. 最可能的 bug（按可能性排序）

### H1（最可能）：LL combine 的 token 映射，在真实条件下错位
combine kernel 按 `rdma_recv_x[topk_idx_reg * num_max + token_idx]` 把 (全局 expert, 原 token) 的专家输出聚合回每个 token；映射靠 dispatch 建的 `handle=(src_info, layout_range, num_max, ...)`。若在真实条件下 **src_info/layout_range 与实际 token 顺序对不上**（或 SGLang 侧传给 combine 的 `topk_idx`/`down_out` 与 dispatch 建 handle 时的顺序错位），则每个 token 拿到**别的 token 的（量级正常的）专家输出** → 保范数乱码、广泛、LL 独有。
- 与所有硬事实吻合：D1–D8 input-faithful 过、范数稳、NORMAL 对（NORMAL 用 `ep_gather`+`output_index` 显式回散，映射路径完全不同）。
- parity 没抓到，是因为它用 `num_max == batch`、无 padding 行、合成数据；真实是 **num_max=128 ≫ batch≈24 + 1 个 -1 padding 行 + 真实 fp8**。

### H2（次可能）：LL 内部"只有 2 个 buffer"的复用 / async 流水
deep_ep 文档明确警告：`low_latency_dispatch`/`low_latency_combine` **只有两个内部 buffer，返回张量复用 buffer，同一时刻不能持有超过 2 个 LL kernel 结果**。
真实 forward 是 48 层 × (dispatch→masked GEMM→combine)，且 `KUNSERVE_DEEPEP_RETURN_RECV_HOOK=0` → `async_finish=True`。若某层 combine 还没读完结果，下一层（或本层）的 LL 调用已复用了那个 buffer → 数据被覆盖成另一份（量级正常的）数据 → 保范数乱码。
- LL 独有（NORMAL 无 2-buffer 约束）、广泛、范数稳、input-faithful 盲（覆盖发生在 kernel 之外）。

### H3（边缘）：真实数据触发的 kernel edge case
`num_max=128 ≫ batch`、front-pack 跨 source-rank、padding 行、真实 topk 分布中某种组合触发 deep_ep LL kernel 的边界 bug（H20 build 特有）。parity 在 `num_max==batch` 下 PASS，不覆盖此条件。

## 4. parity 没复现、真实 run 才有的条件（H1/H3 的根据）

| 条件 | parity | 真实 run |
|---|---|---|
| `num_max_dispatch_tokens_per_rank` | == batch（如 64） | 固定 **128 ≫ batch(~24)** |
| topk 里的 -1 padding 行 | 无（已加 `PAD_ROWS` 但未跑） | 有（LL-CMB `topk_min=-1`） |
| 专家计算 | 合成 `+e` | 真实 masked GEMM（down_out 含 NaN padding 行） |
| async / 2-buffer 流水 | inline 一把过 | 48 层流水 + `async_finish=True` |
| 数据 | 合成 randn | 真实 fp8 hidden（量级/分布不同） |

## 5. 决定性验证方案（按性价比）

### 方案 A（最决定性）：combine token 映射的 marker 注入（诊断 run）
在 `_run_masked_gemm` 输出 `down_out` 后、combine 前，把每个有效槽**写成已知 marker**（如 `down_out[g, slot, 0:4] = encode(global_expert_id, source_rank, slot)`），combine 后在 `combine_b` 检查 `combined[token, :]` 是否等于"该 token 的 topk 专家按 dispatch 顺序应得的 marker 加权和"。
- **若不符** → **H1 实锤**：combine 把 token 映射错了。能直接看出错位规律（差一个 source-rank stride？差 num_max？）。
- 这是唯一绕开 input-faithful + 范数盲点的测试（marker 是受控的、可逐 token 验算）。注意它破坏本次生成，仅作诊断 run，跑几步即可。

### 方案 B（最便宜）：关掉 async / 强同步，测 H2
`KUNSERVE_DEEPEP_RETURN_RECV_HOOK=1`（→ `return_recv_hook=True`，`async_finish=False`，逐 LL 调用同步、不复用未就绪 buffer）重跑 M2。
- **乱码消失** → **H2 实锤**（2-buffer/async 复用）。修法：combine 前同步、或保证 ≤2 个在用结果。
- **仍乱** → 排除 H2，集中查 H1。

### 方案 C：升级 parity 到真实条件
`parity_deepep_ll_remap.sh` 已加 `PAD_ROWS`；再把 `num_max` 从 `==batch` 改成固定 128（≫batch），跑 `PAD_ROWS=4 USE_FP8=1`。
- **REAL 行 diff 变大** → H1/H3 在"num_max≫batch + padding"下复现，定位到 kernel 映射。
- 几秒就能跑，不用整轮训练。

## 6. 建议执行顺序
1. **方案 B**（一个 env，最便宜）先排掉 H2。
2. **方案 C**（几秒 parity）测 num_max≫batch + padding 是否复现。
3. 若 B、C 都不复现 → **方案 A**（marker 注入诊断 run）直接逼出 H1 的映射错位规律。

> 核心判断：bug 几乎可以确定是 **LL combine 这一层的 token 映射**（H1），或 **LL buffer 复用**（H2）。两者都满足"保范数、广泛、LL 独有、input-faithful 盲"。继续做 input-faithful 的反量化对拍不会有新进展——必须用**受控 marker** 或**强同步 A/B** 打穿。

---

## 7. 【2026-06-16 更新】H0 拓扑/PCIe 内存序：已排除

怀疑过：LL 走 NVSHMEM P2P,docstring 警告"PCIe 连接会因 memory ordering 出错";且 `check_nvlink_connections`(deep_ep/utils.py:66)**只在 GPU 名含 'PCIE' 时才强制全 NVLink 检查 → H20(SXM)上被跳过**,deep_ep 不验证 4 卡是否两两 NVLink。

**实测 `nvidia-smi topo -m`:8 张卡两两全 NV18(18-link NVLink,NVSwitch 全互联),用的 2,3,4,6 之间也全 NV18 → 无任何 PCIe 对。H0 排除。** 也与 parity 在同 4 卡 PASS 一致:**LL kernel + NVLink 传输在隔离测试里正确**。

补充澄清(回答"为什么用 RDMA 不用 NVLink"):LL 的"RDMA"只是 API 命名(分配 `num_rdma_bytes`、`nvshmem put/get`);`allow_nvlink_for_low_latency_mode=True`(默认)→ `NVSHMEM_DISABLE_P2P=0` → **单机内物理传输就是 NVLink P2P**;IBGDA 开了但无 IB → fail → 回落 P2P/NVLink。当前 `return_recv_hook=0` → `async_finish=True`(stream-ordered wait,**非** hook-based,所以 docstring 那条"hook 与 nvlink 不兼容"不适用)。

## 8. 重排后的优先级(H0 排除、parity 通过、拓扑 OK 之后)

既然 **LL kernel + NVLink + 单 round-trip(parity)都正确**,bug 只能在 **真实 run 比 parity 多出来的东西**(§4 表):num_max=128≫batch、-1 padding 行、真实 fp8 数据、48 层 async 流水、SGLang a/b split。

- **H1(combine token 映射在真实条件下错位)** —— 仍是最可能。
- **H2(2-buffer/async 流水)** —— 被"单 stream(`KUNSERVE_COMBINE_ALT_STREAM=0`)+ dispatch_b/combine_b 的 stream-ordered wait"**削弱**(同 stream 下 layer N combine 先于 layer N+1 dispatch,不该 race),但未完全排除。
- 现在**最该做的是方案 A(marker 注入)**:它在真实 run、真实 handle 下直接验 combine 的 (expert,slot)→token 映射,是唯一能穿透 input-faithful 盲点的测试。方案 C(parity 升级到 num_max=128≫batch + padding)次之、最便宜。

## 9. 【2026-06-16】parity(NUM_MAX=128 + PAD_ROWS=4 + fp8 + TP)仍 PASS → H2(async)上位

`NUM_MAX=128 PAD_ROWS=4 USE_FP8=1 TP_REPLICATE=1` 跑 parity:四 rank `REAL_max_abs_diff≈0.53`(fp8 噪声),`pad_out_absmax=0.0000`。即复现了 **num_max≫batch + -1 padding + fp8 + TP 复制**,combine token 映射**仍正确**。→ **H1/H3 显著削弱**。

**关键差异浮现**:parity 用 **`async_finish=False`(同步)**,真实 run 用 **`async_finish=True`**(`KUNSERVE_DEEPEP_RETURN_RECV_HOOK=0`)。parity 同步 + 单次 → PASS;真实异步 + 48 层流水 → 乱。

→ **H2(async / deep_ep 2-buffer 流水复用)现在是头号嫌疑**:`async_finish=True` 下两个内部 buffer 被下一次 LL 调用复用;本地 CUDA stream-wait **保证不了跨 rank 的 NVSHMEM put/get 顺序**(对端可能在本端还在读旧结果时就 put 覆盖)→ stale/partial 读 → 保范数垃圾。parity 同步因此天然避开。

**决定性测试(最便宜,先做)= 方案 B**:`KUNSERVE_DEEPEP_RETURN_RECV_HOOK=1` 重跑 M2(→ `async_finish=False`,逐 LL 调用同步,和 parity 一致)。
- **乱码消失** → **H2 实锤**(async/buffer 复用)。临时修法:强同步(慢但对);正解:保证 combine 读完前 buffer 不被复用 / 用 `get_next_low_latency_combine_buffer` 双缓冲。
- **仍乱** → 排除 H2;回到"真实 run 比同步 parity 还多的东西"(SGLang dispatcher a/b split 的 self.handle / packed_recv_count 状态跨层复用)。

## 10. 【2026-06-16】H2(async)排除 → 上方案 A(marker)

`KUNSERVE_DEEPEP_RETURN_RECV_HOOK=1`(→`async_finish=False`,逐 LL 调用同步,与 PASS 的 parity 一致)重跑 M2:仍 **76 length 乱码**(BALLOON {stop:3, length:76})。recv_wait 确认走 hook(`return_recv_hook=True`)。→ **H2(async/2-buffer 复用)排除**:同步照样乱,bug 是确定性的、与时序无关。

至此:H0(拓扑)✗、H2(async)✗;parity 同步+num_max128+padding+fp8+TP 全 PASS,真实同步仍乱。差异只剩 **48 层序列 + 真实 masked GEMM + 真实数据 + SGLang dispatcher 包装**。

**已实现方案 A(marker 注入,`KUNSERVE_MARKER=1`,诊断 run 会破坏生成)**:
- `combine_a`(deepep.py):把 combine 输入每个 local group g 全写成 global dispatch_id = `ep_rank*num_local+g`,并存 topk_ids/topk_weights。
- `combine_b`:验证 `out[t][0] == Σ_{topk_ids[t,k]>=0} topk_weights[t,k]*topk_ids[t,k]`,打 `[LL-MARKER]`(map_max_abs_diff / n_bad / worst token 的 predict vs actual + topk）。
- 透传已加 `KUNSERVE_MARKER`。
判读:
- **map_max_abs_diff 大 / n_bad>0** → **H1 实锤**:真实 run 里 combine 把 token 映射错了(worst token 的 actual 反推出它实际拿到的是哪些 expert → 错位规律)。
- **map_max_abs_diff≈0** → combine 映射在真实 run 也对 → bug 只能在"真实 masked GEMM 的 per-token 输出"或"48 层 dispatcher 状态(self.handle/packed_recv_count)跨层错配",转查那条。
