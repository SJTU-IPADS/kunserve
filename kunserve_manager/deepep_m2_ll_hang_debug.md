# M2 LL 死锁排查文档(DeepEP low_latency_dispatch / Buffer.__init__ hang)

> 分支:`feat/deepep-comm`
> 最后更新:2026-06-11
> 状态:**✅ 已解(2026-06-12)。** 根因 = NVSHMEM(`NVSHMEM_USE_NCCL=ON` 编译)在建 team 时新建 NCCL communicator,与宿主 SGLang 进程已有的 NCCL/CUDA graph/VMM 状态冲突 → `team_internal.cpp:679 'unhandled cuda error'` → `runtime.sync` 挂死。
> **修法:`NVSHMEM_DISABLE_NCCL=1`**(NVSHMEM team 改用自带 ring/recexch,不碰宿主 NCCL)。验证:`[BUF-INIT]` 出现 `POST runtime.sync`,LL dispatch/combine 连续跑 ~25900 个 GLOBAL forward 在实际生成。NVSHMEM_DEBUG=INFO 日志直接读出失败点(非模拟推断)。
> 关键诊断手段:`NVSHMEM_DEBUG=INFO` 暴露 IBGDA/IBRC 失败后 NVSHMEM 走 P2P、然后在 NCCL-backed team setup 上 `unhandled cuda error`。
> 遗留:① CUDA OOM(LOCAL graph 14G + LL NVSHMEM heap + 同卡 FSDP 22G,util=0.83 撑爆)→ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + 降 util;② GLOBAL CUDA graph 仍未捕(`captured_variants=['local']`),LL 目前 eager,graph 是后续提速。
>
> ——以下为排查过程存档——
> 本文是 M2(LL + CUDA graph)调试的专门记录,供接手者快速进入。总览见同目录 `deepep_link.md`。

---

## 0. 一句话现状

把 `deepep_mode=auto`(decode 走 LL)在 KunServe 两实例跨 replica 上跑通的过程中,**第一道坑(两 replica mode 分叉 → `deep_ep.cpp:200 invalid argument` 崩)已用 per-forward is_extend 协商修好**;现在卡在**第二道坑**:decode 第一次用 LL 时,**LL DeepEP Buffer 的 `runtime.sync`(NVSHMEM init)在 4 个 rank 上全部 hang**,无任何报错。

---

## 1. 复现配置

```bash
# H20,4 卡(2 replica × TP=EP=2),Qwen3-30B-A3B
CUDA_VISIBLE_DEVICES=2,3,4,7 N_GPUS_PER_NODE=4 \
  bash /workspace/verl/data/run_deepep_ll_graph_fp8_smoke.sh
```
关键 env:`KUNSERVE_COMM_BACKEND=deepep`、`SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL=0`、`KUNSERVE_DEEPEP_MODE=auto`、`moe_runner_backend=deep_gemm`、`quant=fp8`、`temp=0`。
最近一次卡住的 run:`/workspace/verl/outputs/deepep_ll_graph_fp8_20260611_090023`(及 080853)。

---

## 2. 精确症状(已用探针定位到行)

forward 序列(`[M1]` 逐段探针,layer0 GLOBAL):
- **fwd=1(prefill,协商出 NORMAL)**:`dispatch.enter → expert → combine.exit` **四 rank 全程跑完** ✅
- **fwd=2(decode,协商出 LOW_LATENCY)**:四 rank `dispatch.enter`,**无一 `dispatch.exit`** ❌

往里钻(`[M1-BUF]`/`[LL-CORE]`/`[BUF-INIT]` 探针):
- fwd=2 的 `dispatch_a → _get_buffer → get_deepep_buffer` **lazy 创建 LL buffer**。
- `[M1-BUF] pre-Buffer`(LL,ll=True)四 rank 都打了;`[M1-BUF] POST Buffer()` **没打** → **`Buffer()` 没返回**。
- `[LL-CORE]`(在 `_dispatch_core`,即 `_get_buffer` 之后)**从没打** → 卡在 `_get_buffer` 里,**不是 `low_latency_dispatch`**。
- `[BUF-INIT]`(deep_ep `Buffer.__init__` 分步)四 rank 都走到:
  `ENTER → device_ids gathered → ipc_handles gathered → NVSHMEM block ENTER → nvshmem_unique_id gathered → **PRE runtime.sync** → (无 POST)`

**结论:四 rank 全部卡死在 `self.runtime.sync(device_ids, ipc_handles, root_unique_id)`(deep_ep/buffer.py:124),即 NVSHMEM 的 C++ bootstrap/transport 建立那一步。**

NVSHMEM 自身日志(真实 run,`NVSHMEM_DEBUG=WARN`):
- UID bootstrap(barrier/allgather)**全部 DONE**。
- `transport.cpp:nvshmemi_transport_init:282: init failed for transport: IBGDA`(重复 ×rank)。
- 然后**无后续输出、hang**(约 5 分钟后被 idle 看门狗判死)。

---

## 3. 关键矛盾:standalone repro 怎么都跑通,真实就挂

复现脚本 `/workspace/verl/data/repro_deepep_ll_2replica_buffer.sh`:4 个进程、按真实拓扑各掩 2 卡(device_ids=[0,1,0,1])、建 4-rank LL Buffer + 真跑一次 `low_latency_dispatch`。

**它在以下所有条件下都 PASS(LL buffer 建成 + dispatch ran):**

| 试过的变量 | 结果 |
|---|---|
| 默认拓扑(2 进程 4-rank,掩码) | PASS |
| `NO_MASK`(device_ids=[0,1,2,3]) | PASS |
| `MEM_HOG_GB=85`(显存吃满到剩 10G) | PASS |
| `GROUP_BACKEND=nccl`(NCCL 组,非 gloo) | PASS |
| `NUM_EXPERTS=128` / `num_qps=32`(对齐真实) | PASS |
| `ALLOW_MNNVL=1`(对齐 SGLang 的 allow_mnnvl=True) | PASS |
| `PRE_NORMAL_BUFFER=1`(先在同组建 NORMAL buffer 再建 LL) | PASS |
| `PRE_NCCL_OPS=10`(建 LL 前在组上跑 10 次 all_reduce) | PASS |
| GPU 组 `2,3/4,7` 和真实的 `4,5/6,7` 都试 | PASS |
| `NVSHMEM_DISABLE_CUDA_VMM=1`(对齐真实 env) | PASS |
| IBGDA 失败(repro 里也 `init failed for transport: IBGDA`) | PASS(走 P2P 回退) |

**即:repro 里 IBGDA 同样失败,但它靠 NVLink/P2P 回退把 `runtime.sync` 走通了;真实 run 在同一步 hang。** 差异只剩**真实 SGLang 进程的上下文**,standalone 无法隔离。

---

## 4. 环境事实(已查清)

- **GPU 互联**:`nvidia-smi topo -m` 全是 `NV18` → 8×H20 满 NVLink/NVSwitch,任意两卡 P2P 通。**GPU 拓扑不是问题。**
- **IB 端口**:`ibv_devinfo` 所有 mlx5 口 `state: PORT_INIT`(**不是 PORT_ACTIVE**),link_layer InfiniBand → **没有活的 IB fabric / 没 SM**。所以 **IBGDA 起不来是必然的**(加了 `nvidia_peermem` 也一样)。
- **nvidia_peermem**:已在宿主机 `modprobe`,容器内 `/proc/modules` 可见(`nvidia_peermem` + 依赖 `ib_uverbs`、`nvidia`)。**但 IBGDA 仍 init failed**(因为 IB 口 PORT_INIT)。
- **deep_ep 构建差异**:**H20 的 deep_ep 编译了 NVSHMEM(能跑 LL);本机 A800 的 deep_ep 编译时禁用了 NVSHMEM**(`NVSHMEM is disabled during compilation`,跑不了 LL)。→ **LL 相关只能在 H20 验证,A800 复现不了。**
- `ulimit -l` = unlimited(容器内),`/dev/infiniband` 已映射进容器。

---

## 5. 已排除的假设(逐个被证伪的弯路,别再走)

1. ❌ **2 进程 device_ids=[0,1,0,1] 冲突** → repro 同样 [0,1,0,1] 却 PASS。
2. ❌ **VMM × NVSHMEM cuMem 冲突** → `NVSHMEM_DISABLE_CUDA_VMM=1` 不影响(crash 当时也没修好,后由协商修好;repro 带此 env 仍 PASS)。
3. ❌ **MNNVL 误判** → `NVSHMEM_DISABLE_MNNVL=1` 没改变 buffer hang(它修的是更早一个 sync invalid-argument 的猜测,也未中)。
4. ❌ **deferred recv-hook(TBO)** → `KUNSERVE_DEEPEP_RETURN_RECV_HOOK=0`(inline)仍卡,且卡点在 `runtime.sync`(buffer 创建),根本没到 dispatch。
5. ❌ **TBO 拆批** → `enable_two_batch_overlap` 默认 False,没开。
6. ❌ **先建 NORMAL buffer / 先跑 NCCL collectives** → repro 两者都 PASS。
7. ❌ **GPU 组连通性** → 满 NVLink,换组都 PASS。
8. ❌ **需要 IBGDA** → IB 口 PORT_INIT,IBGDA 本就用不了;repro 没 IBGDA 也跑通(P2P 回退)。

**唯一真实修好的**:**跨实例 mode 分叉**(DR-7,见下),它解决了"一进 GLOBAL 就 `deep_ep.cpp:200` 崩",把进度推进到现在的 buffer hang。

---

## 6. 已落地的修复:跨实例 is_extend 协商(DR-7)

**问题**:`deepep_mode=auto` 按各 replica 自己的 batch 解析 NORMAL(prefill)/LL(decode);两 replica 独立调度,同一个跨实例 collective 上可能一个 NORMAL 一个 LL → 共享的 4-rank buffer/dispatch 参数不一致 → `deep_ep.cpp:200 'invalid argument'` 崩。

**修法**:`model_runner._negotiate_balloon_deepep_is_extend`(commit `ec86399d2`)——每个 GLOBAL forward all-gather 4 rank 的 `is_extend` 取 OR,`set_is_extend_in_batch(any_extend)`;任一 replica 有 extend → 全员 NORMAL,否则全员 LL。
**关键时序**(commit `be2501016`):必须放在 `prepare_mlp_sync_batch`(`forward_batch_info.py:796` 会 `set_is_extend_in_batch(本地值)`)**之后**、`model.forward` 之前,否则被本地值冲掉。
**已验证**:四 rank mode 一致、NORMAL/LL 两 buffer 都一致建成、不再 `deep_ep.cpp:200` 崩。

---

## 7. 当前最佳猜测 + 下一步(未执行)

IB 口 `PORT_INIT`(死链路)。IBGDA 失败后,NVSHMEM 选下一个传输:
- **repro**:走 NVLink/P2P,`runtime.sync` 通。
- **真实**:疑似去试了 **IBRC(CPU-IB)** 在死 IB 口上建 QP → hang(或别的传输选择差异)。

**候选修法(尚未跑验证)**:`NVSHMEM_DISABLE_IB=1` 彻底禁掉 IB 传输,强制 NVLink/P2P-only(repro 证明 P2P 能跑通 LL)。buffer.py 不设这个 env,预先 export 会保留。
> ⚠️ 但注意:repro **不设** `NVSHMEM_DISABLE_IB` 也能走 P2P 通过,所以"真实为何不自动回退 P2P"仍是谜——可能这个 env 也不中。

**另一条诊断路**:真实 run 设 `NVSHMEM_DEBUG=INFO`(比 WARN 详细),抓 IBGDA 失败之后 NVSHMEM **尝试哪个 fallback transport、在哪一步 hang** 的输出。这是最直接的 ground truth(目前 WARN 级别只打到 "IBGDA failed" 就没了)。

**仍未隔离的真实差异(repro 无法复现的)**:
- LL buffer 在 SGLang forward **内部**创建(可能在非默认 CUDA stream / graph 上下文)。
- 进程里已有多个 NCCL communicator(LOCAL TP 组、cross-replica 自定义组 `init_custom_process_group`、model-update 组)+ CUDA graph + VMM。
- cross-replica group 是 `init_custom_process_group`(自定义 TCP store),非 repro 的 `dist.new_group`。

---

## 8. 探针清单(都在 worktree,`pip install -e` 后 H20 自动生效;deep_ep 内的需手动同步)

| 探针 | 位置 | 作用 |
|---|---|---|
| `[M1]` | `fused_moe_triton/layer.py` forward_impl | layer0 GLOBAL 的 dispatch/expert/combine 逐段,采样式 |
| `[M1-BUF]` pre/POST | `token_dispatcher/deepep.py` get_deepep_buffer | buffer 创建前后 + is_extend/mode/CVD/已存在的 modes |
| `[LL-CORE]` PRE/POST | `deepep.py` LL `_dispatch_core` | `low_latency_dispatch` 调用前后 + topk 统计 |
| `[LL-DISP]` | `deepep.py` LL `dispatch_b` | recv-hook 等待前后 |
| `[BUF-INIT]` | **deep_ep/buffer.py `__init__`**(外部安装包!) | runtime.sync 分步:device_ids/ipc/nvshmem_id/PRE-POST sync |

⚠️ `[BUF-INIT]` 在外部 deep_ep 安装包里,**A800 和 H20 的 deep_ep 是不同构建、各自安装**,需在 H20 手动改(锚点:`Buffer.__init__` 里的三处 `all_gather_object` + `self.runtime.sync`)。其余探针在 worktree fork 里,自动同步。

相关 commit:`54370fef1`/`bf6e7f95a`([M1])、`0c0940b0b`/`207c8b52f`/`03ae55d66`([M1-BUF])、`4d0802aec`([LL-CORE]/[LL-DISP])、`ec86399d2`/`be2501016`(协商修复)。

---

## 9. 关键代码位置

- `model_executor/model_runner.py`:`_negotiate_balloon_deepep_is_extend`(协商);`_forward_raw`(在 `prepare_mlp_sync_batch` 后调协商);deepep GLOBAL bundle 注册(`register_balloon_global_runtime_bundle`,`SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL` 分支)。
- `layers/moe/token_dispatcher/deepep.py`:`_DeepEPDispatcherImplLowLatency`(LL dispatch_a/b/_dispatch_core);`DeepEPBuffer.get_deepep_buffer`(buffer 缓存/创建)。
- `layers/moe/fused_moe_triton/layer.py`:`create_moe_dispatcher`(构造 MaybeTboDeepEPDispatcher,`return_recv_hook` 在此);`FusedMoE.forward_impl`(dispatch→core→combine)。
- `deep_ep/buffer.py`(外部包):`Buffer.__init__`(runtime.sync,卡死处);`low_latency_dispatch`。

---

## 10. 给接手者的建议顺序

1. **先确认下一步候选**:H20 真实 run 加 `NVSHMEM_DISABLE_IB=1` 重跑,看 `[BUF-INIT]` 是否越过 `runtime.sync`。
2. 若不中:真实 run 设 `NVSHMEM_DEBUG=INFO`,抓 IBGDA 失败后 NVSHMEM 的 transport fallback 在哪 hang。
3. 若还不行:怀疑 **NVSHMEM init 与进程内已有 NCCL communicator / CUDA graph / 非默认 stream 冲突**——尝试把 LL buffer 的创建挪到 **warmup 阶段、在任何 forward/graph 之前**(干净上下文)预建,看是否绕开。
4. 性能动机:用户实测 **NORMAL vs LL 性能差距很大**,Path A(纯 NORMAL)不可接受,必须打通 LL。

---

## 11. 【2026-06-13 更新】hang 已解,新问题:GLOBAL-LL decode 输出乱码

### 11.1 进展
- **hang 根因已锤定并修复**:`NVSHMEM_DISABLE_NCCL=1`(详见 §0/§7 与 memory `finding_deepep_ll_needs_nvshmem_disable_nccl`)。LL dispatch/combine 已连续跑 ~25900 个 GLOBAL forward。
- **OOM 已解**:降 `GPU_MEMORY_UTILIZATION=0.7` + batch 14(不要用 expandable_segments,与 TorchMemorySaver 冲突,会崩启动)。
- **新问题**:能跑通,但 **balloon 进入 GLOBAL-LL 的 decode 输出乱码**(如 "step. the ..1.1..000000" / "000$-1000$-1000..."),而 M1(NORMAL)输出正确。

### 11.2 乱码的定位证据(已排除项)
- **定位到 GLOBAL-LL decode**:log `deepep_ll_graph_fp8_20260613_021642`。Sample 100(`decode_steps_balloon:0`,全程 LOCAL,finish=stop)输出**连贯正确** → 单实例/LOCAL-LL 没问题;乱码只在 balloon 后的 GLOBAL-LL decode 出现。
- **排除 fp8**:M1 的 NORMAL 路径 `_DeepEPDispatcherImplNormal._dispatch_core` **也走** `sglang_per_token_group_quant_fp8`(NORMAL 同样 fp8 dispatch),NORMAL 正确 → 不是 fp8 量化误差。
- **排除容量溢出**:`num_max_dispatch_tokens=128` > batch,`packed_recv_count` 每个 local expert 计数正常、分布合理。
- **唯一可疑异常**([LL-CMB]/[LL-CORE] 探针):**rank1(replica0 的 TP1,pid 465617)** 的 dispatch topk 持续 `n_neg=8`(某个 token 的 8 个 topk 全是 -1)+ combine `topk_min=-1`;而 **rank0(同 replica、同 token)** `n_neg=0`。**同一 replica 两个 TP rank 的 topk 不一致**——这是 TP-rank 间 topk 不对称,尚不能从 min/max 区分是真发散还是 padding。

### 11.3 数值对拍工具(本次产出)
`/workspace/verl/data/parity_deepep_ll_remap.sh` —— **LL round-trip 正确性对拍**(必须 H20 跑,A800 无 NVSHMEM 不能跑 LL):
- 建真实 4-rank 跨实例组,用真实 KunServe **非连续 ownership**(rank0→{0..31}、rank1→{64..95}、rank2→{32..63}、rank3→{96..127})+ **remap**(`dispatch_id//32==owner_rank`,即 layer0 的 `(0,32,64,96)→(0,64,32,96)`)。
- 每专家 = `f_e(x)=x+e`(distinct、fp8-robust、可直接验算);参考 `out[t]=Σ_k w[t,k]*(x[t]+e_k)`。
- 同输入跑 `low_latency_dispatch`(bf16,清晰)→ 对 group i 加 `inv_remap(r*32+i)` 的 logical id → `low_latency_combine`,逐元素 diff 参考。
- **判读**:
  - **全 PASS** → DeepEP LL + remap 这套 round-trip 正确 → bug 在 **SGLang wiring**(`active_local_expert_mapping` 喂 LL packed 的 row 对应、或 **TP token 复制**那一维——对拍当前是每 rank 自己 token 的简单 EP,没建模 replica 内 2 个 TP rank 共享 token,§11.2 的 rank0/rank1 topk 不对称正指向这维)。下一步把 TP 复制建进对拍。
  - **任一 FAIL** → LL+remap 本身就错 → 非连续布局喂 LL 不成立,需改布局为连续 或 放弃 LL 内部路由。
- 用法:`REPLICA0_GPUS=2,3 REPLICA1_GPUS=4,7 bash /workspace/verl/data/parity_deepep_ll_remap.sh`(语法已本机校验)。

### 11.4 下一步
1. H20 跑 §11.3 对拍,看 PASS/FAIL 二分。
2. 若 PASS:把 **TP token 复制**(replica 内 rank0/rank1 同 token)建进对拍,复现 §11.2 的 rank1 `n_neg=8` 不对称,定位是 SGLang 在 TP+EP 下喂 LL 的 topk/token 分片错位。
3. 若 FAIL:打印 worst token 的 `out_ll vs out_ref + logical/disp topk`,看是 group↔logical 错位还是 combine 回写错位。

### 11.5 【2026-06-14】对拍结果:全 PASS → bug 在 SGLang wiring(TP 维)
H20 实跑 `parity_deepep_ll_remap.sh`,四 rank 全 PASS:
```
rank0: max_abs_diff=0.4403 mean=0.111 ref_mean=65.7 PASS
rank1: max_abs_diff=0.4395 mean=0.107 ref_mean=63.2 PASS
rank2: max_abs_diff=0.4280 mean=0.113 ref_mean=66.3 PASS
rank3: max_abs_diff=0.4833 mean=0.116 ref_mean=66.5 PASS
```
`max_abs_diff≈0.44` 在 `ref_mean≈65` 上是 **bf16 舍入级别**(routing 错位会差几十)→ **DeepEP LL + 非连续 remap + combine round-trip 本身正确**。
**bug 锁定在 SGLang wiring**,且强指向对拍**故意没建模的那一维**:replica 内 2 个 TP rank 共享同一份 token——正对应 §11.2 的 rank0 `n_neg=0` vs rank1 `n_neg=8`(同 replica 同 token,topk 却不一致,本不该发生)。
**下一步**:进真实代码查"为什么同 replica 两 TP rank 喂 LL 的 topk 不一样"(per-rank remap / ExpertLocationDispatch / topk 构造)。

### 11.6 【2026-06-14】缩小到 LL 特有 + 非对称负载(疑 Phase E)
对 M1(NORMAL 正确)vs M2(LL 乱码)逐一排除非 LL 差异:
- **GLOBAL CUDA graph 不是变量**:M2 日志只捕 `variant='local'`,**GLOBAL LL 跑 eager**(没捕 GLOBAL 图)。排除 graph。
- **forward 路径相同**:`moe_a2a_backend=deepep` → `QwenMoeSparseMoeBlock.forward` 走 `forward_deepep`(qwen3_moe.py:285),**不做末尾 `tensor_model_parallel_all_reduce`**(combine 已跨 EP world 聚合,对拍已证 LL combine = 全 topk 求和),且给 `self.topk` 传 `num_token_non_padded`。NORMAL/LL 共用此路 → "双重 reduce""padding 未 mask" 都不是 LL 特有差异,排除。
- **per-rank `logical_to_rank_dispatch_physical_map[ep_rank]`**:确为 per-rank 切片,但 KunServe 每 logical 仅 1 个物理副本 → `_find_nearest_expert` 候选唯一 → 各 rank map 相同;且 `assert all != -1` → remap 不产出 -1。故 rank1 的 `n_neg=8`(整 token 全 -1)**只能来自 remap 之前**(padding / num_token_non_padded mask)。
- **剩下的唯一谜点**:rank0(replica0 TP0)`n_neg=0` vs rank1(replica0 TP1)`n_neg=8`。TP 伙伴 token 相同、router logits all-reduce 后相同 → 后续 topk 本应**逐元素相同**。出现非对称,只能是**两 rank 的 `num_token_non_padded` / 实际 token 数不同** → 强烈指向 **replica 间非对称负载 / idle-keepalive(Phase E 未做)**:一侧有真 decode token、另一侧 idle 用 dummy 凑数,LL 的**固定容量 `num_max_dispatch_tokens_per_rank` 打包 + 跨实例 combine 不容忍这种非对称**,而 NORMAL 的动态 layout 能容忍 → 只有 LL 乱。

**下一步二选一**:
1. **真实代码探针(H20)**:在第一个 GLOBAL-LL decode forward,rank0 与 rank1 各 dump `num_token_non_padded` + 前若干**真实** token 的 post-remap topk + `forward_mode`/是否 idle-keepalive。若两 rank 的 num_token_non_padded 不同 → 实锤 Phase E 非对称。
2. **对拍升级**:在 `parity_deepep_ll_remap.sh` 上加 (a) TP 复制(replica 内两 rank 同 token)+ (b) 一侧 idle(real token 数不同),看是否复现乱码。

### 11.7 新探针 [LL-ANOM](已加,worktree 自动同步 H20)
`deepep.py _DeepEPDispatcherImplLowLatency._dispatch_core`:在 `low_latency_dispatch` 前,**只要 topk 有整行 -1(n_neg>0)就触发**(独立计数 ≤80,跨过 [LL-CORE] 的前 30 次限制,能抓到 decode 乱码时刻),dump `n_tokens / n_neg / fully_masked_rows / num_max`。`KUNSERVE_DETAIL_LOG` 门控。
**判读**:跑 M2 后比对同一 decode step 下 rank0 与 rank1(同 replica 两 pid)的 [LL-ANOM]:`n_tokens` 或 `fully_masked_rows` 不同 → 实锤非对称负载/idle-keepalive(Phase E)在腐蚀 LL 固定容量打包。日志:`grep -aE '\[LL-ANOM\]' kunserve_sglang_detail.log`。

### 11.8 【2026-06-14 log 20260614_030306】修正:Phase-E 是红鲱鱼,真凶疑 LL fp8 scale 路径
- **乱码确认仍在**:37 个经 balloon 的请求 **34 个 finish=length**(顶满 32K 不停,balloon_steps≈21600+),3 个 finish=stop 的 balloon_steps 很少(2420/9656/10287,大部分 decode 在 LOCAL)。完全吻合"finish=length=乱码"签名。
- **[LL-ANOM] 是正常 padding,排除**:4 rank 都 27 token;两个 replica-B rank(816685/816686)decode 期**一致**地 row26 整行 -1(=26 真实+1 padding),replica-A(816330/816332)27 真实无 mask。replica 内两 rank 一致 → 不是 TP 非对称;dispatch/combine 对 -1 按 API 正常处理(combine `topk_min=-1` 也是 padding 所致)。**Phase-E 非对称假设推翻**。
- **重新聚焦**:对拍 PASS 用的是 **bf16 + 合成专家 + 喂对的 mapping**;真实 run 是 **fp8 + 真实权重**。`_dispatch_core`:`elif not SGLANG_DEEPEP_BF16_DISPATCH: use_fp8=True` → **LL 走 fp8**,返回 `(packed_recv_x_fp8, packed_recv_x_scales[.., hidden//128])`,per-128-channel scale、列主序布局——**与 NORMAL fp8 是不同代码路径**。之前"NORMAL 也 fp8 故排除 fp8"的推理**不成立**(两条 fp8 路径不同)。
- **专家核约束**:`layer.py:1743 local_expert_offset = moe_ep_rank * num_local_experts` → group i 必须 = 物理 `ep_rank*num_local+i`(连续),靠 remap 伪装非连续 ownership——对拍已证这套正确。
- **对拍已升级 fp8**(`parity_deepep_ll_remap.sh` 加 `USE_FP8=1`):dispatch `use_fp8=True` → 用返回 scales 反量化 `(fp8.float().view(nl,M,H//128,128)*scales[...,None]).view(nl,M,H)` → +e → combine,阈值放宽到 2.0。
  - **FAIL** → LL fp8 dispatch/scale 路径就是真凶(反量化/scale 布局错)。
  - **PASS** → fp8 也没问题,真凶在真实**专家权重行 wiring**(active_local_expert_mapping ↔ masked-GEMM group 顺序),需在真实 forward 探针对比 NORMAL/LL 喂给专家核的权重行。

### 11.9 【2026-06-14】fp8 对拍也 PASS → comm 全清,补 TP 一致性测试
- **fp8 对拍 PASS**:`USE_FP8=1` 四 rank `max_abs_diff≈0.53`(仅比 bf16 的 0.44 多 0.09 = 纯 fp8 量化噪声,远低于阈值 2.0)。**LL fp8 dispatch+scale 反量化+combine round-trip 正确,fp8-LL 不是 bug**。
- 至此 comm 侧(dispatch/combine/remap/非连续布局/bf16/fp8)**全部被对拍洗清**;NORMAL 正确又证明权重排列+contiguous active_local_expert_mapping(model_runner.py:1153 只支持 contiguous narrow)对。
- **LL vs NORMAL 唯一未双重验证的差异 = 专家核**:LL 用 deep_gemm **masked** grouped GEMM,NORMAL 用 **contiguous** grouped GEMM。外加对拍**没建模 TP 复制**(真实里 replica 内两 TP rank token 相同、都往 EP=4 dispatch,产生 duplication;且 combine 后两者输出必须逐元素相等,否则下一层 TP attention 发散→乱码)。
- **对拍补 TP 一致性**(`TP_REPLICATE=1`):按 replica(rank//2)播种,使 rank0/rank1 同 token、都 dispatch(模拟 duplication);两 partner 都对**同一份**参考,若 LL 有任何 rank 相关发散,其中一个会 FAIL。
  - **TP_REPLICATE FAIL** → TP 复制/duplication 下 LL combine 产生 rank 间发散 = 真凶。
  - **TP_REPLICATE PASS** → comm+TP 全清,真凶只剩**真实 deep_gemm masked grouped GEMM 路径**(kernel 本身 / 权重 scale `w13_weight_scale_inv` 喂 masked 核的方式),需在真实 forward 探针对比单专家 LL 输出 vs 手算参考。

### 11.10 【2026-06-14】TP_REPLICATE 也 PASS → 锁定 masked grouped GEMM 数值路径
- **TP_REPLICATE PASS(bf16+fp8)**:按 replica 播种使 rank0/rank1 同 token、都 dispatch(真实 duplication)。结果 **rank0 与 rank1 的 max_abs_diff/ref_mean 完全相同**(0.4403/65.695),rank2/rank3 相同(0.4395/63.195)→ **TP 伙伴输出逐元素一致,TP 复制/duplication 排除**。
- **scale 切分一致**:`build_dense_expert_runtime_tensors`(layer.py:546)对 w13_weight 和 w13_weight_scale(_inv)用**同一 `narrow(0,start,length)`** → 权重/scale 切得一致,"scale 切错"排除。
- **adapter 不在路径上**:`kunserve_runner_adapter.py` 是未实现骨架,且只服务 bf16/triton;fp8 路径不用它。
- **排除法定论**:comm(dispatch/combine/remap/非连续/bf16/fp8)+ TP + 权重排列/scale切分 全清。LL vs NORMAL 唯一未验证差异 = **deep_gemm masked grouped GEMM**(`moe_runner/deep_gemm.py:_run_masked_gemm`,LL 专属;NORMAL 走 contiguous,M1 从没碰过 masked)。M2 里这个 masked 运行器**只有 GLOBAL LL 会触发**(LOCAL 已是 StandardDispatcher)。
- **新探针 [MASKED-GEMM]**(`moe_runner/deep_gemm.py` down_output 后,KUNSERVE_DETAIL_LOG 门控,≤40):dump `num_groups/m/masked_m/w13+scale 形状/down_mean_abs/down_max_abs/down_has_nan`。
  - down NaN 或 max 爆炸 → kernel blow-up(疑 scale 布局:UE8M0 / `get_mn_major_tma_aligned_tensor` 对 GLOBAL bundle narrowed 权重不匹配)。
  - norms 正常但仍乱 → 更隐蔽,下一步加 in-situ 参考对拍(反量化权重+激活做 bf16 分组 matmul,逐 group diff masked 输出,定位 GEMM-0 vs GEMM-1)。
- 另跑 `KUNSERVE_WEIGHT_PROBE=1`(layer.py:556 已有)验证四 rank 权重字节溯源(replica1 是否读错半)。

### 11.11 【2026-06-14 log 140752】masked GEMM 输出含 NaN — 待分清有效行 vs padding
- `[MASKED-GEMM]` 四 rank、ct=1 起**全部 `down_mean_abs=nan down_max_abs=nan down_has_nan=True`**。w13=(32,1536,2048) w13_scale=(32,12,16)、w2=(32,2048,768) w2_scale=(32,16,6) 形状都对;masked_m_sum≈128-328 合理。
- **但这是整 tensor 判的**:masked GEMM 每专家 m=512 槽、只有 masked_m[g] 有效(sum≈210/16384),`down_output=torch.empty` 的 **padding 行未初始化→NaN 属预期且无害**(combine 只读有效行)。所以**还不能定罪**。
- **探针已细化**(`_run_masked_gemm`):只统计**有效行**([0,masked_m[g]))的 NaN,分 `gateup_valid_nan`(GEMM-0)/`down_valid_nan`(GEMM-1)+ `down_valid_mean_abs`,并保留 `down_all_nan`。
  - `gateup_valid_nan=True` → NaN 起于 **GEMM-0**(`(hs_fp8,hs_scale)@(w13,w13_scale)`),疑激活/权重 scale 布局(UE8M0/tma-align)。
  - `gateup_valid_nan=False & down_valid_nan=True` → 起于 act 或 **GEMM-1**。
  - **两个 valid_nan 都 False** → 有效行无 NaN,bug 是"值错不爆"→ 上 in-situ 反量化参考对拍定位。

### 11.12 【2026-06-14 log 143349】NaN 红鲱鱼,真凶=masked GEMM 输出值错并发散
- **有效行无 NaN**(`gateup_valid_nan=False down_valid_nan=False` 全 160 次)→ 整 tensor NaN 只是无害 padding;且 `down_valid_nan=False` 反证 GEMM-1 正确掩码忽略了输入 padding 的 NaN。**NaN 排除**。
- **真凶 = 值错且发散**:`down_valid_mean_abs` 单 pid 单调上升 **0.013→0.195(40 步 ~15×,近指数)**。GLOBAL-LL 专家输出值不爆但错,每步注入小误差→残差流累积发散→乱码不停。解释了"balloon 步少的请求仍连贯(3 个 finish=stop 都步少)、步多的崩"。
- **形状全对**:w13=(32,1536,2048)/scale=(32,12,16)、w2=(32,2048,768)/scale=(32,16,6),masked_m 合理。所以是**数值/scale 消费**错,非形状/计数。
- **新探针 [MASKED-REF]**(`_run_masked_gemm` GEMM-0 后,`KUNSERVE_MASKED_REF=1` 门控,ct≤2):反量化本 rank fp8 激活(用 `runner_input.hidden_states_scale` 原始逻辑 scale,非 line254 TMA 重排后的)+ w13(per-128x128 block scale),手算 bf16 参考,diff 单个有效 (g,t) 的 `gateup_output`。
  - `gemm0_max_abs_diff` 大(ref 与 act 量级接近但内容差,或量级都差) → **GEMM-0 的 fp8 scale 消费错**(疑 KunServe 喂 masked 核的激活 scale 布局/或 scale vs scale_inv)。
  - diff 小 → GEMM-0 没问题,转查激活量化(silu_and_mul_masked_post_quant)或 GEMM-1(w2)。
- 透传已加 `KUNSERVE_MASKED_REF`/`KUNSERVE_WEIGHT_PROBE`(constants_ppo.py)。

### 11.13 【2026-06-14 log 153614】GEMM-0 算术正确 + "发散"是误判 → 转查权重 provenance
- **GEMM-0 算术正确**:in-situ [MASKED-REF](反量化 fp8 激活+w13 手算 bf16 参考)`gemm0_max_abs_diff=0.002~0.004`、mean=2e-5、`ref_mean_abs==act_mean_abs`、ref[:4]≈act[:4]。`hsc_shape=(32,512,16)` 反量化布局假设正确。w13 的 fp8 scale 消费没问题。
- **撤销"发散"判断**:`_kun_mg_ct` 是 class 级计数器,跨所有 MoE 层共享。核实时间戳:连续 ct 间隔 ~3ms = **同一 forward 内逐层**(非 decode step)。`down_valid_mean` 0.013→0.195 是**残差流随层深正常增长**,不是发散。§11.12 的发散结论作废。
- **in-situ ref 的盲区**:它用 kernel 自己的 `w13[g]` 同时算 ref 和 act,只验证"算得对",**无法验证 `w13[g]` 是否正确专家的权重(provenance)**。这是非连续 balloon 布局最易错处(layer.py:548 注释疑 replica1 读错半区)。
- **下一步 = 权重 provenance**:`KUNSERVE_WEIGHT_PROBE`(已设进 smoke 默认 + constants_ppo 透传)。`build_dense_expert_runtime_tensors` 打每 rank 的 `start`/`length`/`sl_abs_sum`(所用行校验和)/`alt_abs_sum`(另一半)。判读:某 rank 的 `sl_abs_sum`==另一 rank 的、或==自己 `alt_abs_sum` → 读错半区 = 应用错专家。
- 探针扩展:[MASKED-REF] 现同时测 t=0 与 t=last(masked_m[g]-1),查 packed buffer 的 token 索引/scale 错位。

### 11.14 【2026-06-14 log 160340】权重 provenance 正确(用 ep_rank 重判) → 锁定 NaN-padding 经 combine 传染
- **pid→rank(ep_rank 实证)**:1806552=rank0(phys0-31), 1807880=rank1(phys64-95), 1806557=rank2(phys32-63), 1807881=rank3(phys96-127)。**replica 分组是 ep 奇偶(0,1)/(2,3),pid 是交错的**(之前按 pid 前缀分组判错了)。
- **权重 provenance 全对**:ranks0/1 留 VMM 前缀(start=0)、ranks2/3 留后缀(start=32),逐一对应各自拥有的物理专家(build_complementary:replica0 留前缀、replica1 留后缀)。跨 rank 校验和相同 = 两 replica 是同模型副本、同物理专家权重当然相同。**weight bug 排除**。
- 至此 dispatch/combine(对拍)、GEMM 算术(MASKED-REF)、provenance、层增长 全部验证正确,但仍乱(本 run 105 length)。
- **对拍盲区 = NaN padding**:真实 `down_output=torch.empty` → padding 行 NaN(`down_all_nan=True`,`down_valid_nan=False`)。对拍的 padding 是有限垃圾(非 NaN),故无法发现 **low_latency_combine 若触碰 padding 行 → NaN 传染真实 token → 乱码**。这是 LL 特有、对拍未覆盖、且能解释乱码的唯一剩余点。
- **A/B 修复实验 `KUNSERVE_ZERO_PAD=1`**(已设进 smoke 默认 + 透传):combine 前把 down_output padding 行清零。
  - 乱码消失(finish=length→stop) → **实锤:NaN-padding 经 combine 传染**,正式修复=zero/mask masked-GEMM 输出 padding。
  - 仍乱 → combine 正确忽略 padding,NaN 无害,bug 在别处(回头查 combine 的 topk_weights 应用 / 残差加法)。

---

## 12. ✅ 已解(2026-06-15):GLOBAL-LL 乱码根因 = masked-GEMM padding NaN 经 combine 传染

### 根因
deep_gemm masked 运行器(`moe_runner/deep_gemm.py:_run_masked_gemm`)的 `down_output = torch.empty(num_groups, m, n)`,**padding 行([masked_m[g], m))未初始化 = NaN**(探针 `down_all_nan=True` 而 `down_valid_nan=False`)。`low_latency_combine` 对每个专家的全部 m 个槽做 `sum(weight_i * x_i)`;padding 槽 combine 权重=0 但 x=NaN,**`0 * NaN = NaN`** → NaN 传染进真实 token 的合并输出 → decode 乱码(finish=length 不停)。

### 为何对拍没抓到
`parity_deepep_ll_remap.sh` 的 padding 是有限垃圾(dispatch 写的),不是 NaN;NaN 是真实 run 里 masked-GEMM `torch.empty` 引入的。所以对拍把 dispatch/combine/remap/fp8/非连续布局/权重 provenance/GEMM 算术全判对——都对,唯独差这个 NaN-padding。

### 修复
GEMM-1 之后、combine 之前,把 `down_output` 的 padding 行清零(`masked_fill_`,in-place)。`0 * 0 = 0`,NaN 不再产生。默认开启,`KUNSERVE_ZERO_PAD=0` 可 A/B 回退。

### 验证(deepep_ll_graph_fp8_20260615_014628)
经 balloon 的请求 finish_type **从 100% length 翻转为 100% stop**(3/3,GLOBAL-LL 解码 3513/3353/7202 步后正确吐 EOS=151645)。M2 LL GLOBAL 推理首次产出连贯正确输出。

### 排错全链(供复盘)
对拍逐步排除:dispatch/combine(§11.3-11.4)→ fp8-LL(§11.9)→ TP 复制(§11.10)→ 权重 provenance(§11.14)→ 全对;真实 run 探针:NaN 仅在 padding(§11.11)→ "发散"是层深增长误判(§11.13)→ GEMM 算术正确(§11.13)→ 最终定位 NaN-padding×combine(§12)。教训:对拍要复现真实的**未初始化/NaN** 边界,不能只用有限随机值。
