# KunServe 跨实例 MoE 通信:DeepEP 链路(方案 + 实现 + 进度)

> 分支:`feat/deepep-comm`
> 本文合并自 `deepep_link_implementation_plan.md`(方案)、`deepep_ll_kunserve_moe_infer.md`(实现细节)、`deepep_link_HANDOFF.md`(分支状态),并删除已被实测推翻的陈旧内容。
> 最后更新:2026-06-10

---

## 0. 速览:目标 + 当前进度

**目标**:用 DeepEP 的 token-routed dispatch/combine 替换当前 dense all-gather + reduce-scatter(`CrossReplicaStandardDispatcher`,sglang backend),抹掉跨 replica 通信开销。

**进度表**

| 项 | 状态 | 说明 |
|---|---|---|
| 环境验证(H20) | ✅ 2026-06-10 | DeepGEMM fp8+bf16 可用;DeepEP NORMAL 可用;**DeepEP LL 在纯 NVLink 跑通(hidden=2048,fp8),不需要 IBGDA**(见 §3) |
| M0 路由参考 + 单测 | ✅ | `kunserve_routing_ref.py` + 5 测;**离线参考,未接进 forward** |
| PrecisionPolicy | ✅ | `kunserve_precision.py`;model_runner 把 `deepep⇒deep_gemm` 硬 assert 换成策略;bf16 暂 `NotImplementedError` |
| `KUNSERVE_DISPATCH_DTYPE` env + verl 透传 | ✅ | verl `c1a38530` |
| M4 适配骨架 | ✅ | `kunserve_runner_adapter.py`,签名+契约,实现 `raise NotImplementedError` |
| M1 探针 | ✅ | sglang `54370fef1`:`layer.py` 加 `[M1]` 逐段标(dispatch/expert/combine),env-gated,定位 hang |
| **M1 端到端跑通** | 🔧 进行中 | **deepep GLOBAL 路径(4-rank a2a)从没 green 过**;下一步在 H20 上跑 NORMAL/eager/fp8,逐段 bisect(见 §7) |
| M2 LL + CUDA graph | ⏳ | 未写;环境已验证可行 |
| M3 量收益 | ⏳ | — |
| M4 bf16 | ⏳ | 仅骨架 |

**一句话状态**:周边脚手架(M0/Precision/env/M4 骨架)都齐了,**真正的传输链路还没跑通**——一跑 `deepep` 执行的仍是 legacy `MaybeTboDeepEPDispatcher`(NORMAL),它从未端到端 green。M1 = 把它修通。

**三条出发原则**

1. 不盲信现有 DeepEP 代码——`MaybeTboDeepEPDispatcher`+NVSHMEM+fp8+deep_gemm 那套从未端到端跑通,当脚手架看,逐段验证。
2. 传输层直接用 DeepEP,不走 PyTorch NCCL(dense all-gather / Phase G a2a)。
3. 精度解耦:先 fp8/deep_gemm 跑通,bf16/triton 作为今后一等公民(M4),设计上预留接口。

---

## 1. 设计:三层解耦(传输固定为 DeepEP)

```text
              topk_ids(GLOBAL physical) + hidden_states
                              │
        ┌─────────────────────▼─────────────────────┐
        │ L1 Routing/Metadata(精度无关、传输无关)    │
        │  static remap → GLOBAL physical id          │
        └─────────────────────┬─────────────────────┘
        ┌─────────────────────▼─────────────────────┐
        │ L2 Transport = DeepEP(唯一)                │
        │  decode→low_latency_*  prefill→NORMAL       │
        │  dispatch_dtype ∈ { fp8(先), bf16(今后) }   │
        └─────────────────────┬─────────────────────┘
        ┌─────────────────────▼─────────────────────┐
        │ L3 Runner Adapter(精度策略落地)            │
        │  fp8 → deep_gemm grouped GEMM(原生)        │
        │  bf16 → triton fused_moe(今后,需布局适配)  │
        └─────────────────────────────────────────────┘
```

**精度策略(PrecisionPolicy,独立配置)**

| 配置项 | 取值 | 现默认 | 说明 |
|---|---|---|---|
| `transport` | deepep | deepep | 不走 PyTorch NCCL |
| `dispatch_dtype` | fp8 / bf16 | fp8 | DeepEP 通信 payload 精度(`use_fp8`) |
| `expert_runner` | deep_gemm / triton | deep_gemm | 专家 GEMM 后端 = `moe_runner_backend`,**不另设 env** |

> 关键解耦:transport=deepep **不应**再强制 expert_runner=deep_gemm。`KUNSERVE_DISPATCH_DTYPE`(唯一新增 env)不设时由 runner 推导(deep_gemm→fp8,triton→bf16);策略会拒绝不一致组合(如 deep_gemm+bf16)。默认完全向后兼容。

---

## 2. 决策记录(实现时勿违反)

### DR-1 ⚠️ 稀疏度 ∝ world/k:本套(k=8,4-rank)几乎不稀疏 — 2026-06-10
128 专家、4 rank、每 rank 32、k=8。一个 token 命中某指定 rank 的概率
`1 - C(96,8)/C(128,8) ≈ 0.91`,平均要发往 **~3.6 / 4 张卡** → **近乎稠密**。
- DeepEP 的稀疏红利来自 rank 数远大于 k(64–128 rank 时 8/64≈12.5% 才真稀疏);4-rank 退化到稠密区。
- **本套 DeepEP 的真实收益 = fp8 payload(~2x 字节,dense 当前是 bf16)+ 更轻的 combine(不做全量 all-reduce),不是 rank 级稀疏。** 收益多少必须 M3 实测,别指望"稀疏省通信"。
- 预期校正:M1/M3 若"没省多少"是拓扑决定,不是 bug。

### DR-2 DeepEP 维持 4-rank a2a,不 lane 化 — 2026-06-10
- dense 需要 lane(Phase F):4-rank all-gather 是真冗余([A,A,B,B]),lane 砍掉 2x。
- DeepEP 不需要:half-split 与 lane 对齐 + static remap → **4-rank a2a 跨 lane 传输本就为零**,lane 化数据收益 ≈ 0。
- lane 化反而更麻烦:① 每 lane 一套 DeepEP buffer(NVSHMEM/handle/graph ×2);② 按 rank 切专家范围;③ **倒贴一次 TP all-reduce**——4-rank combine 现在把 token 全 128 专家聚合好(`reduce_results=False`),拆 lane 后 lane0/lane1 各只聚合半边,最终要再 all-reduce 把两半加起来(即 dense 路径那次)。
- 结论:**lane 是 dense 的对症药,不是 DeepEP 的。4-rank a2a 是最贴近标准 DeepEP、最少定制的路,维持不变。**

### DR-3 GLOBAL `reduce_results=False`
DeepEP combine 已把 top-k expert outputs 跨 EP world 聚合回 token 原位;若 FusedMoE 末尾再对 local TP group all-reduce 会 double-reduce。(对照:LOCAL bundle 用 StandardDispatcher,combine 是 no-op,需 `reduce_results=True` 补 TP all-reduce。)

### DR-4 `ep_dispatch_algorithm=static`
BALLOON 后 physical expert layout 变成两 replica 互补;router 输出 logical id 必须经 static metadata 映射到 GLOBAL physical id,否则 DeepEP 路由到错 owner → 乱码。**首要正确性验证点**(model_runner 有 `[KUNSERVE-DBG]` 查 map 非 None)。

### DR-5 为什么 GLOBAL 用 DeepEP 而非 StandardDispatcher
Standard 假设所有 EP rank 一开始就有同一 batch hidden;但跨 replica 时 peer 没有你的 request hidden/KV/scheduler 状态。DeepEP 只把 MoE 所需 hidden dispatch 到 expert owner,算完 combine 回来——正好契合。

### DR-6 为什么不合并实例
live merge 要迁移 req_to_token / KV / radix tree / scheduler queue / sampling / HTTP stream / graph / PG。本方案请求与 KV 永不迁移,只跨 replica 发 MoE hidden states。

---

## 3. 环境验证结果(H20,2026-06-10)

脚本:`/workspace/verl/data/check_h20_deepep_deepgemm.sh`(7 层 A–H)。

| 能力 | 结论 |
|---|---|
| 架构 | H20 = sm_90(Hopper),DeepGEMM FP8/TMA 支持 |
| nvcc | 12.8(torch cu12.9,小版本错位仅性能警告,非阻塞) |
| DeepGEMM | ✅ `fp8_gemm_nt` + `bf16_gemm_nt` 都真跑出结果,JIT OK(新版 API,旧名 `gemm_fp8_fp8_bf16_nt` 已无) |
| DeepEP NORMAL | ✅ 2-GPU NVLink 通(prefill/extend) |
| **DeepEP LL** | ✅ **纯 NVLink 跑通 `low_latency_dispatch`(hidden=2048,fp8)** |
| IBGDA | ❌ 未就绪(`nvidia_peermem` 没加载),但 **单机不需要** |

**关键修正(推翻旧文档的 §9.1)**:单机 4 卡(replica0=GPU0,1 / replica1=GPU2,3,跨实例全程 intranode NVLink)**LL 不需要 IBGDA**。日志里 `device mlx5_X cannot allocate buffer` + `nvshmemi_transport_init: init failed for transport: IBGDA` 是 NVSHMEM 先试 IB-RDMA 失败后**回退 intranode P2P** 的无害噪音。**无需让管理员加载 `nvidia_peermem`,无需重建容器**;`ulimit -l` 已是 unlimited。只有真做跨机 RDMA 才需要 peermem。

---

## 4. 当前代码实现(BALLOON 后一次 MoE infer)

> 这一节描述**目标架构**(LL + graph 是 M2 的终点)。当前 M1 只跑 NORMAL/eager;LL 路径代码存在但从未 green。

### 4.1 一句话
- LOCAL 阶段:两 replica 独立推理。
- BALLOON 阶段:每 replica 只留一半互补的本地 MoE experts,释放另一半权重显存给 KV;MoE 层经跨 replica DeepEP dispatch/combine 把 token hidden 送到持有目标 expert 的 rank 计算,再 combine 回原 token。
- attention/KV/scheduler/队列 全留在各自 replica;**跨 replica 通信只发生在 MoE 层**。

### 4.2 关键代码入口

| 作用 | 文件 / 函数 |
|---|---|
| 启动脚本 | `verl/data/train/run_smoke_test_kunserve_tp2_dual_replica.sh` |
| manager sidecar | `sglang/kunserve_manager/kunserve_manager/{cli,controller}.py` |
| HTTP endpoints | `sglang/.../entrypoints/http_server.py`(`/kunserve/{status,warmup_balloon,prepare_balloon,commit_balloon,restore_from_balloon}`) |
| BALLOON runtime 核心 | `model_executor/model_runner.py`:`register_balloon_global_runtime_bundle / warmup_balloon / commit_balloon / _force_local_bundle_to_standard_dispatcher` |
| MoE dispatcher | `layers/moe/fused_moe_triton/layer.py`、`layers/moe/token_dispatcher/deepep.py` |

### 4.3 EP 拓扑与 layout
```text
replica 0: GPU0,1 → global rank 0,1     replica 1: GPU2,3 → global rank 2,3
manager /init_weights_update_group 建 4-rank cross-replica NCCL group(复用 _model_update_group)
half-split:
  replica0 留 [0..31, 64..95]   replica1 留 [32..63, 96..127]
  → 4 rank 合起来覆盖 128 physical experts(rank0:0-31 rank1:64-95 rank2:32-63 rank3:96-127)
```

### 4.4 控制面状态机
```text
warmup_balloon : 注册 GLOBAL bundle(+ LL 路径 capture graph),不切状态/不释放权重/业务仍 LOCAL
prepare_balloon: local → prepared(快速切换)
commit_balloon : 暂停 replay → 从 MoE VMM borrow donor rows → 映射给 KV VMM → 扩 KV pool →
                 live metadata 切 GLOBAL → 所有 FusedMoE switch_runtime_bundle("global") →
                 _balloon_state="balloon", replay_enabled=True
```

### 4.5 数据面(decode,目标 = LL)
```
1. attention(TP 分片)→ o_proj RowParallel all-reduce → 两 TP 卡各拿到本 replica 完整 hidden
   (注:此 all-reduce 在 attention 里,dense/DeepEP 共用,与 dispatch 后端无关)
2. router gate → topk(logical ids)
3. static remap → GLOBAL physical ids               [DR-4]
4. FusedMoE.forward_impl(global bundle):
     dispatch → run_moe_core(deep_gemm fp8) → combine
   - dispatch:decode 走 buffer.low_latency_dispatch(use_fp8=...);prefill 走 NORMAL buffer.dispatch
   - combine :buffer.low_latency_combine,按 topk_weights 聚合回原 token(reduce_results=False [DR-3])
5. MoE output 回原 replica,后续 residual/norm/attention 仍本地
```
当前 M1 强制 `SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL=1` → 全程 NORMAL、eager(不 capture GLOBAL graph)。

### 4.6 两个实现要点
- **LOCAL 强制 Standard(仅 LL 路径)**:`SGLANG_EXPERIMENTAL_VMM_MOE_WEIGHTS and not SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL` 时,LOCAL bundle 改 StandardDispatcher,避免 NVSHMEM double-init(GLOBAL LL 才初始化一次 NVSHMEM)。NORMAL 路径(=1)不初始化 NVSHMEM,故不强制。
- **DeepEPBuffer 按 (group, NORMAL/LL) 缓存**:`_buffer_cache[group_id][mode]`,防 LOCAL/GLOBAL、NORMAL/LL 互相覆盖;配置不符直接报错。

---

## 5. 里程碑 + 验收

| 里程碑 | 内容 | 传输 | 精度 | graph | 验收 |
|---|---|---|---|---|---|
| M0 ✅ | L1 static remap + RoutingPlan 单测(对拍 dense union 语义) | — | — | — | map 非 None;本 rank 需算 (t,k) 全覆盖 |
| **M1 🔧** | 点亮现有 DeepEP 路径端到端(逐段修 warmup/commit/dispatch/combine/balloon 布局) | DeepEP | fp8/deep_gemm | eager 先 | smoke 不崩;temp=0 输出对拍 dense 一致、无乱码 |
| M2 ⏳ | GLOBAL DeepEP LL 的 CUDA graph capture/replay 稳定 | DeepEP | fp8 | ✅ | `Capturing batches ... variant='global'`;replay 输出一致 |
| M3 ⏳ | profiling 量收益 | DeepEP | fp8 | ✅ | dispatch/combine ms vs dense 下降(注意 DR-1:收益主要在 fp8 payload) |
| M4 ⏳ | bf16:`use_fp8=False` + L3 triton 布局适配 | DeepEP | bf16/triton | ✅ | bf16 对拍 fp8/dense |

> M1 不是写新 dispatcher,而是把已有但没跑通的 4-rank deepep 路径修到能跑。M4 的 bf16 才需实质新代码(L3 grouped↔sorted 布局转换)。

**必须保持的语义(验收清单)**:① attention/KV 不跨 replica、请求不迁移;② dispatch 等价 union token;③ combine 回原 replica、原 token 顺序;④ padding/phantom rows 不影响 logits;⑤ `reduce_results=False`[DR-3];⑥ idle replica keepalive 到对端结束(Phase E;NORMAL 由 buffer setup 兜,LL 下需确认);⑦ graph replay 下 DeepEP buffer 指针 capture 前固定。

---

## 6. 分支状态 + 合并注意

### worktree(开发位置,勿在主 checkout 切分支)
```
/workspace/sglang-deepep   → sglang feat/deepep-comm(基点 kunserve-4-rl HEAD 60b9a947e)
/workspace/verl-deepep     → verl   feat/deepep-comm(基点 main HEAD 266d761d)
主 checkout /workspace/{sglang,verl} 留给 codex 做 H20/VMM/NCCL perf,勿动。
```

### 关键 commit
| 仓库 | commit | 内容 |
|---|---|---|
| sglang | `23f0b53f0` | M0 路由参考 + 5 测 |
| sglang | `259c87878` | PrecisionPolicy(拆 deepep⇒deep_gemm 硬 assert) |
| sglang | `31eac92aa` | `KUNSERVE_DISPATCH_DTYPE` env + M4 骨架 |
| sglang | `54370fef1` | **M1 逐段探针**(layer.py `[M1]` 标) |
| verl | `c1a38530` | `KUNSERVE_DISPATCH_DTYPE` 4 点透传 |

### 文件改动清单
| 文件 | 状态 |
|---|---|
| `model_executor/kunserve_precision.py` | ✅ 新增,8 单测 |
| `model_executor/model_runner.py` | ✅ assert→policy;bf16 暂 NotImplementedError |
| `layers/moe/token_dispatcher/kunserve_routing_ref.py` | ✅ M0 参考,**未接 forward** |
| `layers/moe/token_dispatcher/kunserve_runner_adapter.py` | ✅ M4 骨架,raise NotImplementedError |
| `layers/moe/fused_moe_triton/layer.py` | ✅ M1 探针;(M1/M4 还要接 PrecisionPolicy) |
| `layers/moe/token_dispatcher/deepep.py` | ⏳ M1:LL dispatch/combine 逐段验证修复(尚未动) |

### ⚠️ 合并注意(最重要)
两个 `feat/deepep-comm` 都缺主分支上"未提交"的修复作底(分支从已提交 HEAD 切):
- **verl** 缺 main 的 **NCCL 网卡 forward**(`constants_ppo` passthrough、`async_sglang_server` actor env、脚本 `NCCL_SOCKET_IFNAME` pin)。**M1 上机前必须先合,否则 H20 NCCL 卡。** `KUNSERVE_DISPATCH_DTYPE` 透传与 NCCL forward 代码结构刻意对齐,合并就是并列表。
- **sglang** 缺 kunserve-4-rl 的 **VMM 改动**(`cuda_vmm.py`,codex 在做)。
- 策略:等 codex 把修复在主分支提交后 `git merge`。

### 验证
```bash
cd /workspace/sglang-deepep/python
PYTHONPATH=$PWD python3 -m pytest sglang/test/kunserve/ -q   # 19 passed, 1 xfailed(M4 占位)
```

---

## 7. 怎么在 H20 上跑 M1(当前下一步)

`/workspace` 即 H20 盘,worktree 改动已在 H20 上。sglang 是 PEP660 editable 指向主 checkout,需临时重指到 worktree(只动 sglang,verl 留主 checkout 保住 NCCL forward)。

```bash
# ① sglang 切到 worktree(带 M1 探针 + PrecisionPolicy)
pip install -e /workspace/sglang-deepep/python
python3 -c "import sglang,os; print('sglang =', os.path.dirname(sglang.__file__))"  # 期望 .../sglang-deepep/...
#  ⚠️ 全容器生效,挑 codex 不起新 sglang 进程时做

# ② 跑 deepep NORMAL/eager/fp8 端到端(挑 4 张空闲卡)
CUDA_VISIBLE_DEVICES=2,3,4,7 N_GPUS_PER_NODE=4 \
  bash /workspace/verl/data/run_deepep_normal_eager_fp8_smoke.sh

# ③ 看探针定位卡在哪段(最后一条 [M1] 的 stage = 卡死/崩溃处)
RUN=$(ls -dt /workspace/verl/outputs/deepep_normal_fp8_* | head -1)/kunserve
grep -aE '\[M1\]' "$RUN/kunserve_sglang_detail.log" | tail -15
grep -anE 'Traceback|Error|Assertion|illegal|NVSHMEM|deep_gemm' "$RUN/verl_training.log" | tail -20

# ④ 测完还原
pip install -e /workspace/sglang/python
```

**读探针**:每 GLOBAL forward 在 layer0 打 5 标 `dispatch.enter→dispatch.exit→expert.enter→expert.exit/combine.enter→combine.exit`。某 `.enter` 没对应 `.exit` = 卡在那段(dispatch=跨实例 a2a / expert=deep_gemm / combine)。直接 Traceback = 贴报错定位行。

**预期**:M1 第一次大概率会崩或挂(从没 green 过)——那正是要的信息。流程:跑 → 贴探针最后几条 + Traceback → 修那一段 → 再跑,逐段推进。

---

## 8. 风险与未决

1. M1 是"修没跑通的现有代码",未知点多(warmup/commit/LL handle/balloon static remap 任一段都可能有 bug)——靠 §7 探针二分。
2. 数值:fp8 expert(deep_gemm)在 30B temp=0 历史无漂移,仍需重验。
3. graph capture(M2):DeepEP LL 在 KunServe 自定义 capture 路径下 buffer 固定要确认(从未真正验证)。
4. M4 L3 布局转换(grouped↔sorted)是 bf16 最大不确定性,单测要足。
5. Phase E keepalive 在 LL 下要确认 idle replica 不 hang。
6. 收益预期见 DR-1:4-rank 近稠密,别指望稀疏,主要靠 fp8 payload。
