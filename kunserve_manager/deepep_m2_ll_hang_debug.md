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
