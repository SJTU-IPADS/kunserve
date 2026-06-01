# KunServe 综合分析与优化文档

> 本文档合并了原先散落的 6 份文档（`kunserve_implementation_detail.md`、
> `cross_replica_standard_like_dispatcher_方案综述.md`、`global_cuda_graph链路说明.md`、
> `phase_e_negotiate策略.md`、`lane_combine_tp_allreduce_fusion_准备文档.md`、
> `moe_comm_compute_overlap_准备文档.md`），并补充了截至 2026-05-28 的全部优化进展、
> 实测数据和瓶颈分析。全文以 **30B、两个 TP=2 实例** 为固定例子讲解。
>
> 顶层入口 `/workspace/kunserve_architecture.md` 仍保留作为简短总览，细节以本文为准。

最后更新：2026-05-29

---

> ## ✅ 2026-05-31 最终结论（已验证，读正文前必看；覆盖之前所有诊断）
>
> 本文档经历过 4 轮诊断，前 3 轮都有测量/方法错误，**以本节为准**。最终结论建立在
> **(a) 验证过的插桩代码 + (b) temperature=0 确定性对照 + (c) 真实 prompt 文本配对**之上。
>
> **插桩代码验证**（用户加的，已逐行核对 `scheduler.py:_log_decode_step_timing` + `run_batch`）：
> `total_kv_tokens = Σ(len(prompt)+len(output))` = 真实 KV；`gap_ms` = run_batch 间隔（含 `copy_done.sync` 等 GPU）；
> `iter_ms = step_ms + gap_ms` = **每步真实墙钟，Σiter = decode 总墙钟，正确**。**只有 per-stage 的
> `KUNSERVE_STAGE_PROFILE` CUDA-event 是坏的（capture 时被冻结），`iter_ms`/`KV` 数据可信。**
>
> **核心事实（temp=0 run `ab_20260530_053452`，确定性 + prompt 文本配对）：**
> 1. **输出完全一致**：kunserve vs baseline 逐 prompt 配对（0/176 不匹配）长度比中位 **1.000**，
>    **91% 最终答案相同**，**无乱码**。→ 之前"balloon 漂移 → 输出变长 → 变慢"**已证伪**
>    （那是 `req_lifecycle` 里 baseline 17 个 abort 残缺记录 + temperature=1.0 采样方差造成的假象）。
> 2. **端到端 decode 墙钟**：baseline `Σiter=2567s`，kunserve `2660s` = **+3.6%**（不是之前说的 +13.5%）。
> 3. **【2026-06-01 Kineto 逐阶段修订】同 bs 每步差不是恒定 +5.5ms，而是 +4~+39ms 的变量**，分四块（详见 §11.3）：
>    - **纯跨实例传输 ~5ms 恒定** = dispatch all-gather ~2.4 + combine reduce-scatter ~2.7 + remap ~0.1（9 个均衡 bucket 稳定，和 bs78 nsys 一致）。这是旧版说的"+5.5ms"，只是其中一块。
>    - **跨实例 lockstep 同步等待（旧版漏了的大头）** = replica-0 的 ncclAllGather kernel **在 GPU 上空转等负载更重的 replica-1**，最高飙到 +24ms（bs54 AG=26ms）。lockstep 强制两 replica 步调一致，谁慢另一个就在 AG 里等；低 bs / rollout 后期（序列陆续结束→失衡）更频繁。
>    - **attention** 同 bs 下 kunserve KV 更高（并发更高）→ +0~+16ms；但**同 KV 下两者相等**（bs74: KV 431 vs 474、attn 25.5 vs 27.9），attention 本身非差异点，只是"同 bs 长度差异"通过 KV 真实贡献时间。
>    - **expert** 高 bs kunserve **更便宜**（每卡 32 vs 64 expert，−3ms），低 bs ~中性。
> 4. **交叉验证（证为真非 profiler 伪影）**：用**非 profiled** 的 batch_timing `iter_ms` 中位对每个 bs，与 profiled 窗口 kernel 总和对比，**15/15 bucket PROFtot≈REAL**（bs54 PROF72.6≈REAL72.7）。⇒ sync 等待是真 GPU 墙钟。真实同 bs 差 kun−base = **+3.7~+38.8ms**（无 profiler）。
> 5. **端到端仍只 +3.6%**：kunserve 步数更少（balloon 高并发免排队、0 abort vs baseline 17 abort）抵消了每步更慢。**attention** 是单步最大组成，但同 KV 下两者一样、非差异点。
>
> **一句话**：30B 上 KunServe 与 baseline **产出相同**，decode 端到端慢 **+3.6%**；同 bs 每步差是 **+4~+39ms 的变量**，由 **纯跨实例传输 ~5ms + 新发现的跨实例 lockstep 同步等待（all-gather 空转等对端，最高 +24ms） + attention(KV 驱动) + expert(更便宜)** 组成（Kineto 逐阶段 + 非 profiled iter 交叉验证）。**优化方向修正：不只是砍传输，更要解 lockstep 同步等待 —— Phase E + 跨实例负载均衡（对齐两 replica 序列长度/数量减少 AG 空转），或合并 TP=4 彻底消跨实例。**
>
> **作废的旧结论（仅供对照，正文标注「❌ 旧」）**：① "9.4ms NCCL wire-bound"（INTERNAL_TIMING graph-event
> 扭曲）；② "+13.5% 慢 / 输出长 9%"（abort 残缺 + 采样方差）；③ "跨实例只 ~1ms/~2.8ms、非瓶颈、瓶颈是
> attention、微优化点错方向"（前者只取高 bs 残差/截距偏高而低估，后者 GLOBAL-vs-LOCAL bs 不重叠是外推；
> 正确做法是 baseline-vs-GLOBAL matched bs+KV = **+5.5ms**，**跨实例 EP 就是瓶颈，方向没点错**）。

## 目录

1. [KunServe 是什么 / 解决什么问题](#1)
2. [整体架构图](#2)
3. [固定例子：30B 两个 TP=2 实例，128 experts](#3)
4. [数据面：一层 decode 的完整 forward 链路](#4)
5. [控制面：balloon 状态机与三个进程组](#5)
6. [GLOBAL CUDA graph 链路（为什么 post-commit capture）](#6)
7. [Phase E negotiate 策略（原理 + 为什么不能更激进）](#7)
8. [已落地的性能优化逐项详解（P1/P2/P3/RouteA/B.1/B.2）](#8)
9. [评测流程：指标、plot 脚本、一个完整 iter、vs 单实例 TP=4](#9)
10. [compare 脚本命令行参数 → 优化开关映射](#10)
11. [核心瓶颈：为什么 KunServe 比 baseline 慢，且现有手段无法补上](#11)
12. [后续可探索的方向](#12)

---

<a name="1"></a>
## 1. KunServe 是什么 / 解决什么问题

KunServe 是在 **verl GRPO rollout + SGLang 后端** 上做的一套 "MoE expert balloon" 机制。

**核心动机**：rollout 解码时 KV cache 显存吃紧，请求会排队甚至被 retract。KunServe 的想法是——当 KV 紧张时，**把每个 replica 一半的 MoE expert 权重显存临时"借"给 KV cache**，从而扩大 KV 容量、容纳更多并发请求。被借走的 expert 不再本地计算，而是通过 **跨 replica expert sharing（cross-replica EP）** 由对端 replica 远程算出来。

**核心约束**（决定了整个架构）：

- Attention / KV cache **永远归属原 replica**，不跨 replica 迁移。
- 每个 replica 内部仍是原生的 `TP=2` attention + local TP all-reduce。
- **只有 MoE expert 计算跨 replica 补齐**。
- balloon 后输出不能乱码；某个 replica 先 drain 完请求时不能 hang。

这套机制有两个运行态：

| 运行态 | 何时 | MoE 怎么算 | CUDA graph |
|---|---|---|---|
| **LOCAL** | balloon 前（KV 充裕） | 本实例 local bundle，等价原生单实例 TP=2 | 原生 local graph |
| **GLOBAL (BALLOON)** | KV 吃紧、`expand_requested` 触发后 | 跨 replica EP，每卡只算 1/4 experts | post-commit GLOBAL graph |

> **重要现实**（2026-05-28 实测）：在当前 30B workload 上，balloon 一旦 commit 就 **不会 restore**（restore 条件是所有 replica 完全 drain，连续 workload 下几乎不发生），所以 ~86% 的时间都在 GLOBAL 模式付 cross-replica EP 的通信代价。这是 KunServe 比 baseline 慢的根本原因，详见 [第 11 节](#11)。

---

<a name="2"></a>
## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│ verl / Ray  (trainer + rollout)                                          │
│                                                                           │
│   AgentLoopManager                                                        │
│     └── KunServeController  (kunserve_manager/controller.py)             │
│           ├─ 初始化 runtime_group + lane_group[0] + lane_group[1]        │
│           ├─ warmup_balloon : 注册 GLOBAL bundle（不 pre-capture）       │
│           ├─ poll /kunserve/status  (默认 poll_interval=2s)              │
│           ├─ 看到 expand_requested=True → prepare_balloon → commit_balloon│
│           └─ balloon 后只观测；restore 仅在全 replica drain 时触发        │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 │ HTTP /kunserve/*
        ┌────────────────────────┴────────────────────────┐
        ▼                                                   ▼
┌───────────────────────────┐                ┌───────────────────────────┐
│ SGLang replica 0  (TP=2)  │                │ SGLang replica 1  (TP=2)  │
│  ┌─────────┐ ┌─────────┐  │                │  ┌─────────┐ ┌─────────┐  │
│  │ rank 0  │ │ rank 1  │  │                │  │ rank 2  │ │ rank 3  │  │
│  │ lane 0  │ │ lane 1  │  │                │  │ lane 0  │ │ lane 1  │  │
│  │ GPU 0   │ │ GPU 1   │  │                │  │ GPU 2   │ │ GPU 3   │  │
│  └────┬────┘ └────┬────┘  │                │  └────┬────┘ └────┬────┘  │
│       │ local TP  │       │                │       │ local TP  │       │
│       │ all-reduce│       │                │       │ all-reduce│       │
│  scheduler(Phase E)       │                │  scheduler(Phase E)       │
│  model_runner             │                │  model_runner             │
│  local_bundle/global_bundle                │  local_bundle/global_bundle│
└───────┼────────────┼──────┘                └───────┼────────────┼──────┘
        │            │                               │            │
        │            └───────── lane_group[1] ───────┼────────────┘
        │              (rank1 ↔ rank3, experts 64-127)
        └────────────────────── lane_group[0] ───────┘
                       (rank0 ↔ rank2, experts 0-63)

        runtime_group = {rank0,1,2,3}  (Phase E negotiate 用)
```

三类通信组（控制面在 `warmup_balloon` 阶段建立，都走 `kunserve_pynccl` registered collective）：

| 组 | 成员 | 用途 |
|---|---|---|
| `runtime_group` | rank 0,1,2,3 | Phase E 跨 replica 协商 graph bucket |
| `lane_group[0]` | rank 0 ↔ rank 2 | lane 0 的 dispatch all-gather + combine reduce-scatter |
| `lane_group[1]` | rank 1 ↔ rank 3 | lane 1 的 dispatch all-gather + combine reduce-scatter |

---

<a name="3"></a>
## 3. 固定例子：30B 两个 TP=2 实例，128 experts

模型 Qwen3-30B-A3B：`hidden=2048`, `层数 L=48`, `experts=128`, `topk=8`, attention heads=32。

两个 SGLang 实例各 `TP=2/EP=2`，共 4 GPU。原生状态下每个实例内：

- TP lane 0 负责本实例 expert **0..63** 的前半 expert 空间；
- TP lane 1 负责本实例 expert **64..127** 的后半 expert 空间；
- attention `o_proj` 后做本实例 `tp_group.all_reduce`，两张卡都拿到本实例完整 hidden。

**进入 BALLOON 后**，每张卡只保留 32 个 physical experts（offload 掉一半）：

| global rank | replica | TP lane | 保留的 physical experts |
|---|---|---|---|
| rank 0 | replica 0 | lane 0 | **0..31** |
| rank 1 | replica 0 | lane 1 | **64..95** |
| rank 2 | replica 1 | lane 0 | **32..63** |
| rank 3 | replica 1 | lane 1 | **96..127** |

这样划分的逻辑：

- 同一个 lane 的两个 rank 合起来覆盖该 lane 原本负责的整段 expert 空间：
  - lane 0：rank0(0..31) + rank2(32..63) = **0..63** ✓
  - lane 1：rank1(64..95) + rank3(96..127) = **64..127** ✓
- 每张卡省下的另外 32 个 expert 的 VMM physical page 被 borrow 给 KV cache。
- 因此 global 的 128 个 expert 仍然完整存在，只是分散在 4 张卡上，需要跨 replica 协作才能算全。

> 历史 bug：曾经把所有 replica 的 `active_local_expert_mapping` 都映射成 0..31，导致 expert 32..63 / 96..127 永远不被计算，输出尾部乱码。现在 `layout.py` + `physical_to_logical_map` 保持全局一致语义。

---

<a name="4"></a>
## 4. 数据面：一层 decode 的完整 forward 链路

以 GLOBAL（balloon 后）一层 decode 为例，**每张卡**上发生的事：

```
  [本 replica 内, 原生 TP=2]
  1. prepare_attn  (input layernorm + residual)
  2. attention QKV / core / o_proj
  3. attention 后 local TP all-reduce        ← 两张卡都拿到本 replica 完整 hidden
  4. prepare_mlp   (post-attn layernorm + residual)
  5. router gate → topk  (每 token 选 8 个 global expert id)

  [跨 replica, lane_group 内]  ← KunServe 独有，baseline 没有
  6. dispatch: lane all-gather
       lane0: rank0 ↔ rank2 交换 hidden/topk_ids/topk_weights
       → 每个 rank 拿到 union = [本replica tokens, 对端replica tokens]
  7. dispatch remap: global expert id → 本 rank 的 local compact id（不在本卡的 → -1）
  8. expert kernel (Triton): 只算本卡保留的 32 个 expert 对 union 所有 token 的贡献
  9. combine: lane reduce-scatter
       lane 内对各 expert shard 的 partial 求和，只取回本 replica chunk → [M, H]

  [本 replica 内, 原生 TP=2]
  10. local TP all-reduce  ← 合并 lane0/lane1 的 partial，得到本 replica 完整 MoE 输出
  11. post-layer (residual + norm)
```

**关键数学**（对 replica `r` 的 token `t`，目标输出）：

```
Y[r,t] = Σ_{lane∈{0,1}} Σ_{shard∈{0,1}} partial[shard, lane, r, t]
       = local_tp_all_reduce( lane_reduce_scatter( partial[:, lane] ) )
```

即 combine 分两步求和：先 lane 内（跨 replica shard）reduce-scatter，再 replica 内（跨 lane）TP all-reduce。

**为什么必须 lane 通信、不能 4-rank 全互联**：attention 后每张卡已经有本实例完整 hidden，跨 replica 只需要同 lane 之间交换。如果改成 4-rank all-gather，会多搬不需要的另一个 lane 的数据，还会打破 "lane0 管 0..63 / lane1 管 64..127" 的 expert 空间划分。

**dispatcher 静态 buffer 形状**（30B 例子，capture_max_m = 单 replica 最大 padded batch，记 M）：

```
_buf_padded_hidden        [M, 2048]         本卡的输入（pad 到 M）
_buf_gathered_hidden      [2M, 2048]        lane all-gather 后 = union（aliased 为 union buffer）
_buf_union_topk_ids       [2M, 8]           union 的 global expert id
_buf_union_topk_ids_remapped [2M, 8]        remap 后的 local compact id
_buf_combine_local_slice  [M, 2048]         combine reduce-scatter 输出（本 replica chunk）
```

所有 `_buf_*` 在 `__init__` / `_allocate_static_buffers` 中一次性分配，**指针在所有 (variant, bs) graph capture 间保持稳定**——这是 CUDA graph 能正确 replay 的前提，运行期只能 `copy_/zero_/fill_` in-place，绝不能重新赋值 `self._buf_*`。

关键代码：`sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py` 的 `_dispatch_static` / `_combine_static`。

---

<a name="5"></a>
## 5. 控制面：balloon 状态机与三个进程组

状态机（`model_runner.py` + `scheduler.py` + `controller.py`）：

```
  LOCAL ──warmup_balloon──> LOCAL(bundle registered)
    │  (注册 GLOBAL bundle + process group，但不 capture GLOBAL graph)
    │
    │  scheduler 检测到 KV 不足 retract → expand_requested=True
    ▼
  PREPARED ──prepare_balloon──> (确认 bundle/PG/layout，graph replay 暂停)
    │
    ▼
  BALLOON ──commit_balloon──>
    │  1. 暂停 graph replay
    │  2. 从 offloaded experts 的 VMM page borrow donor
    │  3. donor remap 给 KV cache，扩容 max_total_num_tokens
    │  4. 切 MoE live expert metadata 到 GLOBAL physical mapping
    │  5. 预留 dummy_kv_slot + phantom_req_idx（Phase E keepalive 用）
    │  6. post-commit capture GLOBAL CUDA graph
    │  7. _balloon_graph_replay_enabled = True
    ▼
  BALLOON steady decode (GLOBAL graph replay + Phase E negotiate)
    │
    │  restore 条件：所有 replica running==0 AND waiting==0  ← 连续 workload 几乎不触发
    ▼
  (rarely) LOCAL
```

`expand_requested` 触发点（`scheduler.py:3019`）：当 `retract_decode` 因 KV 不足回退请求时置位，控制器 poll 到后启动 balloon。

> **关于 restore 策略**（`controller.py:_should_restore_balloon`）：当前只有当 **所有 replica 完全 drain**（running=0 且 waiting=0）才 restore。用户确认这是 **有意为之** 的策略：如果 balloon 后还有大量排队，restore 回去只会重新排队，没有意义。所以连续高负载下 balloon 是 "一旦进入就长期保持"。

---

<a name="6"></a>
## 6. GLOBAL CUDA graph 链路

### 6.1 为什么是 post-commit capture，不能 pre-capture

CUDA graph 记录的是 capture 当时的 **kernel DAG + 指针拓扑**。但 `commit_balloon` 会改变最终内存布局：

- expert VMM 权重的 donor page 被 borrow 给 KV cache；
- KV cache VMM 容量扩展；
- MoE live expert metadata 切到 GLOBAL physical mapping；
- dummy KV slot + phantom req_pool row 被预留；
- GLOBAL runtime bundle 在最终状态重新绑定。

如果在 commit 前 capture，replay 时这些指针已经失效 → balloon 后乱码 + 异步 CUDA illegal memory access（历史 bug）。所以现在严格 **先 commit 完成所有内存重排，再在最终布局上 capture**。实测 post-commit capture 很快（十几秒），可接受。

### 6.2 fixed_padded replay

- graph bucket 用固定 batch size：`1,2,4,8,...,256`（共 36 个）。
- replay 时真实 `raw_bs=7` → 选 `graph_bs=8`，多出的 padding 行写入 `dummy_kv_slot`，用 `phantom_req_idx` 作 req_pool row，`topk_ids=-1` 让 expert kernel 跳过。
- padding 行参与 graph shape 和 collective，但不产生真实输出；replay 后只截取 `raw_bs` 行的 logits。

关键代码：`cuda_graph_runner.py:capture_one_batch_size` / `replay_prepare`。

---

<a name="7"></a>
## 7. Phase E negotiate 策略

### 7.1 要解决什么

GLOBAL graph replay 有两个 **硬约束**：

1. 所有 lane rank 必须按 **同样顺序** 进入 collective；
2. 同一 collective 的 tensor shape 必须 **一致**。

所以每个 decode step 都要让所有 replica 对 "本 step 用哪个 graph bucket" 达成一致。最朴素做法是每步做一次 `runtime_group` collective 协商，但实测这会把 scheduler gap 放大到每步 ~20ms 量级，吞吐崩掉。

### 7.2 当前策略：低频 lockstep refresh + 窗口内 cache 复用

默认 `KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=256`：

- **refresh step**（每 256 步一次）：用 `runtime_group` collective 交换每个 rank 的 `(local_padded_bs, local_force_eager)`，算出 `max_bs / min_bs / any_force_eager`，缓存。
- **cache step**（其余 255 步）：不做 collective，复用缓存的决策。所有 rank replay `cached_max_bs` 对应的 bucket，小 batch pad 到该 bucket；idle replica 发 `ForwardMode.IDLE` keepalive batch（target_bs = cached_max_bs）参与 collective。

决策表：

| 场景 | 行为 |
|---|---|
| 所有 replica bucket 相同 | replay 该 bucket |
| busy/busy 但 bucket 不同 | 所有 rank replay `max_bs` bucket，小 batch pad |
| idle/busy | idle rank 发 IDLE keepalive，target = max_bs |
| 任一 rank EXTEND/mixed prefill | 所有 rank force eager（跳过 graph） |
| 所有 rank idle | max_bs=0，停 keepalive，不 run batch |
| cache step 本地增长超 cached max | reset cache，下一步 collective refresh |
| cache step 本地有 prefill 等待 | defer prefill admission 到 refresh step |

### 7.3 为什么不能更激进

- **不能 "只在本地 batch 变化时 negotiate"**：本 rank 知道自己变了，但不知道 peer 变没变。如果本 rank 单独发起 collective，peer 还在复用 cache 不进 collective → **死锁**。所有 negotiate 必须是 lockstep（要么都进，要么都不进）。
- **不能永久固定到最大 bucket**：batch 慢慢变小时会浪费大量 padding 计算和通信。
- **代价**：peer 的 batch shrink 在 cache 窗口内（最多 256 步）观察不到，会多跑少量 padding/keepalive。这是性能与正确性的折中。

### 7.4 Keepalive

当 `max_bs>0` 但本地无真实请求：构造 `ForwardMode.IDLE` batch（`reqs=[]`，但 input_ids/seq_lens/out_cache_loc/req_pool_indices 按 target bucket 构造，KV 写 dummy_kv_slot）。它参与 graph shape 和 lane collective，不产生真实 token。

关键代码：`scheduler.py` 的 `_phase_e_get_step_decision` / `_build_balloon_keepalive_batch` / `_phase_e_try_reuse_cached_decision`。

### 7.5 进展与状态

Phase E **已落地且稳定**。已解决：last-request hang、mismatched-decode 走 eager（现在用 graph bucket override）、每步 negotiate 过慢、release 收尾断言（drain result_queue + reset cache）。

**未做（Phase E 形式化 idle keepalive 的进一步优化）**：peer shrink 的实时观测需要 manager heartbeat 低成本广播 peer batch bucket，目前没做。这不是当前性能瓶颈的大头。

---

<a name="8"></a>
## 8. 已落地的性能优化逐项详解

> 下面每一项都标注了：**它在优化什么 / 原理 / 进展 / 收益 / 为什么停在这里**。
> 所有开关默认值见 [第 10 节](#10) 和 `CLAUDE.md`。

> ❌ **旧错误结论（已作废，保留供对照）**：早期版本在这里放了一张 "每层 stage × 48" 的表，
> 算出 "KunServe 独有开销 9.4ms/replay，其中 dispatch AG 4.3ms + combine RS 3.1ms = 7.4ms
> wire-bound NCCL"，并据此判断 "瓶颈是 NCCL 通信"。
>
> **这是错的**（见顶部勘误 + [9.0 节](#90)）：那些 `graph_kunserve_*` 单层毫秒数来自
> `INTERNAL_TIMING=1` 的 run，被 graph 内 CUDA event 注入严重放大。**真实的整个 transformer
> CPU launch（step_ms）只有 ~3ms**，dispatch+combine 在 step_ms 上的增量只有 ~0.3ms。
> 真实墙钟差距（+13ms/step）在 scheduler gap 里，归属待 [11.4 节](#11) 的诊断 run 定性。
>
> **下面 P1/P2/P3 的"单 stage 减幅"仍是真的**（它们确实减少了对应 stage 的 GPU 时间/launch 数），
> 但因为整个 GPU forward 在墙钟里占比极小（这个 workload 是 scheduler-bound），所以**端到端
> 吞吐收益都在 noise 内**——这恰好和 "瓶颈不在 GPU" 的结论一致。

### 8.1 P1 — dispatch 3 个 all-gather 合并成 1 个 ncclGroup ✅ 已落地

- **开关**：`KUNSERVE_DISPATCH_NCCL_GROUP=1`（默认开）
- **优化什么**：`_dispatch_static` 原本对 hidden / topk_ids / topk_weights 各发一次 `all_gather_into_tensor`（3 次 NCCL kernel launch）。
- **原理**：NCCL 在 `ncclGroupStart/End` 内会把同 communicator 的 collective 合并，省掉 `3-1=2` 次 kernel launch。新增 `KunServePyNcclGroup.grouped_all_gather_into_tensor`，绕开 `register_custom_op`（那层只为 torch.compile 可见性，graph capture 不依赖），直接走 `pynccl.group_start() → 3× all_gather → group_end()`。
- **进展**：完成，preheat 加了 group bracket 预热。
- **收益**：dispatch_AG 单层 0.082→0.072 ms，省 ~0.77 ms/replay，端到端 +1.6% 吞吐。
- **为什么停**：launch overhead 本来就小，这是热身性质的低风险优化，到顶了。

### 8.2 P2 — dispatch topk_ids remap 融合成单个 Triton kernel ✅ 已落地

- **开关**：`KUNSERVE_REMAP_FUSED=1`（默认开）
- **优化什么**：remap 把 union 的 global expert id 转成本 rank 的 local compact id。原本是 5–7 个 torch op：`(ids>=0)&(ids<N)` → `clamp` → `to(long)` → `mapping[safe]` → `where(...)` → `copy_`。
- **原理**：写一个 Triton kernel `out[i,k] = mapping[ids[i,k]] if 0≤ids<N else -1` 一次搞定。文件 `kunserve_remap.py`，在 `_allocate_static_buffers` 末尾用真实 buffer preheat 一次（强制 JIT 编译发生在 capture 外）。
- **进展**：完成，单测验证与原 torch chain bit-equivalent（4096 元素混合 valid/-1/越界，mismatch=0）。
- **收益**：remap 单层 0.024→0.008 ms（-67%），省 ~0.8 ms/replay GPU + 少 240 个 kernel launch。
- **为什么停**：remap 本身已经压到极限（~0.008 ms 几乎是 1 个 kernel 的下限）。

### 8.3 P3 — dispatch pad 在满 batch 时跳过 zero/fill ✅ 已落地

- **开关**：`KUNSERVE_PAD_SKIP_WHEN_FULL=1`（默认开）
- **优化什么**：`_dispatch_static` 先 `zero_()` 整个 padded buffer 再 `[:local_m].copy_(hidden)`。当 `local_m == capture_max_m`（满 batch graph）时，copy 全量覆盖，zero 是浪费。
- **原理**：`local_m == M` 时跳过三个 zero/fill。`local_m < M` 时保留（padding 区必须是 0/-1）。每个 (variant,bs) graph 在 capture 时定分支，replay 时按各自录的走。
- **进展**：完成。
- **收益**：满 batch graph 省 ~1MB GPU 写 + 1 kernel launch / 层 ≈ 0.4 ms/replay。
- **为什么停**：只对满 batch graph 有效，且量很小。

> **P1+P2+P3 累计 ~1.6 ms/replay 节省**，但端到端吞吐只 +0.3%（noise 内）——因为这 1.6 ms 占总 wall-clock 太小，被 jitter 吃掉。**结论：本地小 op / launch overhead 已经基本榨干**。

### 8.4 Route A — lane reduce-scatter + TP all-reduce 融合成 composite op ✅ 已落地但默认关，无收益

- **开关**：`KUNSERVE_STATIC_COMBINE_TP_ALLREDUCE_FUSION=1` 或 `KUNSERVE_STATIC_COMBINE_MODE=reduce_scatter_tp_all_reduce`（默认关）
- **优化什么**：combine 的 lane reduce-scatter 和后面 qwen3_moe 的 local TP all-reduce 是两次独立的 Python 调用 + 两条 NCCL。想把它们包成一个 `register_custom_op`。
- **原理**：`parallel_state.py:kunserve_lane_reduce_scatter_then_tp_all_reduce` 内部顺序调 `lane_group._reduce_scatter_tensor` + `tp_group._all_reduce_in_place`，并通过 `_kunserve_tp_allreduce_done` 标志让上层跳过重复的 TP all-reduce（避免双 reduce 导致数值爆炸）。
- **进展**：完成，env-gated 默认关，作为对照基线保留。
- **收益**：**0**。CUDA graph 下 host launch overhead 已被摊平，composite op 不减少 NCCL wire 也不减少 device kernel 数，纯属把两条 NCCL 换个地方调。实测 combine 单层 0.090→0.110 ms（甚至略差）。
- **为什么停**：在 graph 模式下 ncclGroup / composite 都无法把 **不同 communicator** 的 collective fuse 成一个 device kernel，所以零收益。保留代码仅供未来非 graph 场景或对照。

### 8.5 B.1 — combine NCCL 移到 alt stream（为 overlap 做基础设施）✅ 已落地但默认关

- **开关**：`KUNSERVE_COMBINE_ALT_STREAM=1`（默认关）
- **优化什么**：把 combine 的 lane reduce-scatter fork 到一条 side CUDA stream 上，为 B.2 的 "expert 计算与 combine 通信 overlap" 铺路。
- **原理**：单例 alt stream（`get_kunserve_combine_alt_stream`），用 `apply_qk_norm` 验证过的 `alt.wait_stream(main) → with stream(alt): NCCL → main.wait_stream(alt)` fork/join 模式。preheat 阶段在 alt stream 上预热 NCCL（避免 graph capture 内 lazy init）。
- **进展**：完成，验证 alt-stream NCCL 在 fixed_padded graph capture 内 graph-safe、数值等价。
- **收益**：B.1 自身 **0**（主路径上 expert 之后没有别的 main stream 工作可以 overlap，只是把 NCCL 挪到 alt stream，main stream 在 join 处空等）。它只是 B.2 的前置基础设施。
- **为什么停**：见 B.2。

### 8.6 B.2 — chunked expert + chunked combine overlap ✅ 已落地但默认关，**净亏损**

- **开关**：`KUNSERVE_COMBINE_CHUNKED=1`（默认关，需 `KUNSERVE_COMBINE_ALT_STREAM=1`）
- **优化什么**：把 expert kernel 按 replica 切成 2 个 chunk（union[0:M] 和 union[M:2M]），让 chunk-0 的 combine NCCL（在 alt stream）与 chunk-1 的 expert kernel（在 main stream）overlap。
- **原理**：`combine_with_chunked_expert` 编排：main 跑 expert_0 → fork alt 做 ncclReduce(chunk0) → main 跑 expert_1 → main 做 ncclReduce(chunk1) → join。新增 `KunServePyNcclGroup._reduce_into_tensor`（直接 `ncclReduce`，每 chunk root=lane_pos）。
- **进展**：完成，数值正确（tail dump 连贯）。
- **收益**：**负的**。实测：
  - Triton expert kernel M=256 比 M=512 只快 12%（不是 50%）——**小 batch 下 launch overhead 主导**。
  - combine NCCL M=256 比 M=512 只快 9%。
  - chunked 把 expert 和 NCCL 各调一倍，多出的开销（~5 ms/replay）远超 overlap 省下的（~2 ms/replay）。
  - 加上 CUDA graph node 数翻倍，host `cuda_graph_replay_launch` +4.5 ms。
  - **端到端吞吐 -8%**。
- **为什么停**：**结构性失败，工程上救不了**。decode 阶段 batch（M=256）太小，kernel/NCCL 的固定 launch 开销远大于按数据量线性的部分，chunking 必然亏。只有 M≥1024 量级才可能翻盘，而 decode 不会有那么大 batch。代码保留作为 "未来更大 batch / 更新硬件" 的预留。

### 8.7 B1（SM-side reduce kernel）— ❌ 未实现，评估后放弃

- **想法**：用 CUDA IPC + 自写 SM-side reduce kernel 替代 NCCL，削 dispatch/combine 的 NCCL launch overhead。模板是 `sgl-kernel/csrc/allreduce/custom_all_reduce.cuh`。
- **为什么放弃（结论变了，但仍不做）**：早期放弃理由（"GPU 36ms→33ms 仍慢"）基于被扭曲的数字，已作废。**新的真实情况**：整个 GPU forward 的 CPU launch（step_ms）才 ~3ms，dispatch+combine 的增量只有 ~0.3ms——**B1 最多能省的就是这 0.3ms 里的一部分**，对 46ms 的 iter_ms 完全无意义。真正的 +13ms 在 scheduler gap（见 [11.3](#11)），B1 碰都碰不到。**所以 B1 更不该做了**——它优化的根本不是瓶颈。

---

<a name="9"></a>
## 9. 评测流程：指标、plot 脚本、一个完整 iter、vs 单实例 TP=4

<a name="90"></a>
### 9.0 ⭐ 关键：timing 字段的真实代码含义（不能只看字面）

分析任何 timing 前必须先懂插桩点。`sglang_batch_timing.jsonl` 由 `scheduler.py` 的
`run_batch` (3087-3259) + `_log_decode_step_timing` (5011-5089) 写出。代码事实：

**`step_ms`**（`scheduler.py:5026` `= perf_counter() - step_start`，step_start 在 run_batch 入口 3098）：
- overlap 模式下（`enable_overlap=True`，默认开），forward 在 `forward_stream` 上 **异步 enqueue**
  （3146-3158：`forward_batch_generation` 只 launch kernel，`copy_to_cpu` 用 `non_blocking=True`
  + `copy_done.record()`），**CPU 不等 GPU**。
- 所以 **`step_ms` = CPU 把整个 transformer 的 kernel enqueue（含 cuda graph replay launch）
  出去的时间，不是 GPU 执行时间**。

**`gap_ms`**（`scheduler.py:3097` `= now - self._last_run_batch_end_ts`，end_ts 在上一次 run_batch
末尾 3253 设置）：
- 是上一次 `run_batch` 返回 → 这一次 `run_batch` 进入 之间的 CPU 时间。
- 中间 event loop（`event_loop_overlap` 1413-1619）做：`result_queue.append` → `recv_requests`
  → `process_input` → `get_next_batch`(+Phase E negotiate/decision) → **`pop_and_process`(处理上一批结果)**。
- **关键**：`pop_and_process` → `process_batch_result_decode`
  （`scheduler_output_processor_mixin.py:478-481`）会调 **`result.copy_done.synchronize()`**
  —— 阻塞 CPU 直到 **上一步 GPU forward+sample+copy 完成**。

**`iter_ms = step_ms + gap_ms`**：因为 CPU event loop 串行（run_batch 之间无空隙），iter_ms 是每步
真实墙钟。`throughput ≈ batch_size / iter_ms × 1000`。

**最重要的推论**：**`step_ms` 和 `gap_ms` 无法直接分离 "CPU 调度" 和 "GPU forward"**：
- `step_ms` 只是 CPU launch（异步，完全不含 GPU 执行）。
- 真实 GPU forward 时间藏在 `gap_ms` 的 `copy_done.synchronize()` 里（如果 GPU 比 gap 里的
  纯 CPU 工作慢，这里阻塞；否则不阻塞，GPU 时间被完全 overlap 隐藏）。
- 要把 GPU 时间从 gap 里挑出来，**唯一办法**是开 `DETAIL=1`（写 `scheduler_decode_result_copy_done_sync`
  stage，`mixin:481`）**且** `INTERNAL_TIMING=0`（否则 graph 内插的 event 会污染 GPU 时间）。

> 这就是为什么早期用 `graph_*` stage 得出的 "9.4ms NCCL" 结论是错的：那些 stage 在
> `INTERNAL_TIMING=1` 下被 CUDA event 注入扭曲，而真正的墙钟分解应该看 `step_ms`/`gap_ms`/
> `copy_done_sync`。

### 9.1 一个完整 rollout iter 包含什么

verl rollout 一次 iter（`compare_kunserve_vs_baseline.sh` 跑一次）：

1. **加载 + 权重同步**（FSDP2，被 `skip_initial_rollout_weight_sync` 可跳过）。
2. **prefill 阶段**：处理 prompt（1024 token），填 KV。
3. **decode 阶段**：逐 token 生成（最多 20000-32768 token）。这是主体，KunServe 的 balloon 在这里触发。
4. **结束**：`release_memory_occupation`。

### 9.2 关键产物文件（每个 run 一个目录）

| 文件 | 内容 |
|---|---|
| `verl_training.log` | 全部 stdout，含 `[KUNSERVE-MS]` 里程碑 |
| `kunserve_milestones.log` | 过滤后的 balloon 里程碑（prepare/commit/restore） |
| `kunserve_sglang_detail.log` | `[KUNSERVE-DBG]` 详细日志（probe_init / *_active / static_combine_mode 等） |
| `kunserve_forward_timing.jsonl` | **逐 stage timing**（plot 脚本的输入） |
| `bw_throughput.jsonl` | 每隔几秒采样：sm 利用率 / KV token_usage / running / waiting / throughput |
| `rollout_metrics.jsonl` | prometheus 指标（含 `sglang_gen_throughput`） |
| `prompt_answer_streaming_r{0,1}.txt` | 流式输出（**查正确性看这里的 tail**） |

### 9.3 关键指标及其含义

| 指标 | 来源 | 含义 |
|---|---|---|
| 总时长 | bw_throughput 首尾 ts 差 | 整个 rollout wall-clock |
| `sglang_gen_throughput` | rollout_metrics | 每实例 tokens/sec（稳态取 trim 10% 后均值） |
| `running_sglangN` | bw_throughput | 并发 decode 请求数 |
| `waiting_sglangN` | bw_throughput | 排队请求数 |
| `token_usage_sglangN` | bw_throughput | KV 占用率（0-1） |
| `graph_qwen3_moe_transformer` | forward_timing | 单次 graph replay 的 GPU 端总耗时 |
| `graph_kunserve_dispatch_static_all_gather` | forward_timing | 单层 dispatch all-gather 耗时 |
| `graph_kunserve_combine_static_lane_reduce_scatter` | forward_timing | 单层 combine 耗时 |
| `cuda_graph_replay_launch` | forward_timing | host 端 graph enqueue 耗时 |

### 9.4 plot 脚本怎么用

`/workspace/verl/data/analyze/plot_kunserve_forward_timing.py`：

```bash
python /workspace/verl/data/analyze/plot_kunserve_forward_timing.py \
  --input  <run>/kunserve_forward_timing.jsonl \
  --output /tmp/timing.png \
  --csv-output /tmp/timing.csv \
  --include-regex "graph_kunserve|graph_qwen3_moe|cuda_graph_replay_launch" \
  --metric mean          # mean / median / p95 / max
```

- 它对每个 stage（事件名去掉 `_end` 后缀）聚合 mean/median/p95/max，输出柱状图 + CSV。
- `--balloon-ts <unix_ts>`：按 balloon 时间切 pre/post 两段对比。
- `--include-regex`：只看关心的 stage（强烈建议加，否则 stage 太多）。
- `--top N`：只保留最大的 N 个 stage。
- **看瓶颈的常用姿势**：`awk -F',' '$2=="all"{print $4, $1}' timing.csv | sort -rn | head -25`。

### 9.5 是否需要测 "不开 log 的真实性能" —— **强烈建议**

当前所有对比都开了 `KUNSERVE_FORWARD_TIMING_DETAIL=1` + `KUNSERVE_GRAPH_INTERNAL_TIMING=1`。**graph internal timing 会在 graph 内插入 CUDA event record/elapsed_time**，这本身有开销（之前观察到 `cuda_graph_replay_launch` 从 11ms 涨到 22ms 的异常，部分就来自 timing 注入）。

**建议做一组干净对照**：

```bash
ONLY_RUN=kunserve KUNSERVE_PHASE_G=0 KUNSERVE_CAPTURE_POLICY=fixed_padded \
  KUNSERVE_ROLLOUT_QUANTIZATION=none KUNSERVE_MOE_A2A_BACKEND=none \
  KUNSERVE_MOE_RUNNER_BACKEND=triton \
  KUNSERVE_FORWARD_TIMING_DETAIL=0 \
  KUNSERVE_GRAPH_INTERNAL_TIMING=0 \
  bash /workspace/verl/data/compare_kunserve_vs_baseline.sh
```

只看总时长 + `sglang_gen_throughput`（这俩不依赖 detail timing）。这能告诉我们 **去掉 timing 注入后，KunServe 与 baseline 的真实 gap 到底是多少**——很可能比开 timing 时小。这是评估 "现有优化是否真有用" 最干净的方法。

### 9.6 vs 单实例 TP=4 的区别（重要概念澄清）

注意有 **两个不同的对比基线**，别混淆：

- **实际 AB 对比的 baseline** = `run_smoke_test.sh` = **两个独立 TP=2 实例**（无 balloon、无 cross-replica EP）。这是我们 compare 脚本里真正跑的对照。
- **用户心中的理想** = 单实例 **TP=4**（业界常用配置）。

| | 2×TP=2 baseline | 单实例 TP=4 | KunServe GLOBAL |
|---|---|---|---|
| MoE 算法 | 每实例独立，本地全 expert | 4 卡 EP，每卡 1/4 expert | 跨 replica EP，每卡 1/4 expert |
| dispatch all-gather | 无 | **无**（hidden 已全有，本地索引） | **有**（lane all-gather）← 独有开销 |
| combine | 无 | TP all-reduce | lane reduce-scatter + TP all-reduce |
| KV 容量 | 固定 | 固定 | **balloon 可扩** ← 独有优势 |

**关键洞察**：KunServe GLOBAL 在 MoE 通信结构上其实最接近 "两个半的 TP=4"，但比单实例 TP=4 **多了一道 lane all-gather**——因为 TP=4 单实例的 hidden 在 attention 后就已经全卡可见，不需要再 gather；而 KunServe 的 hidden 只在本 replica 内可见，必须跨 replica gather 一次才能让对端帮算 expert。这道 all-gather（4.3ms/replay）就是 KunServe 相对 TP=4 的纯增量代价。

---

<a name="10"></a>
## 10. compare 脚本命令行参数 → 优化开关映射

`compare_kunserve_vs_baseline.sh` 跑两个子脚本：
- kunserve：`run_smoke_test_kunserve_tp2_dual_replica.sh`
- baseline：`run_smoke_test.sh`
（80B 用 `compare_kunserve_vs_baseline_80B.sh` + `*_80b_next.sh` 两个子脚本。）

### 10.0 推荐命令 + 产物对照（先看这个）

**正常跑（出干净数据 + dashboard，30B）**——只需后端那几个，**不要**开 profiler：
```bash
KUNSERVE_CAPTURE_POLICY=fixed_padded KUNSERVE_ROLLOUT_QUANTIZATION=none \
  KUNSERVE_MOE_A2A_BACKEND=none KUNSERVE_MOE_RUNNER_BACKEND=triton \
  CUDA_VISIBLE_DEVICES=2,3,4,6 N_GPUS_PER_NODE=4 \
  bash /workspace/verl/data/compare_kunserve_vs_baseline.sh
```

**哪个产物由什么决定**（回答"为什么没生成 XX 图"）：

| 产物 | 谁生成 | 需要什么 |
|---|---|---|
| `<run>/{kunserve,baseline}/rollout_report.png`（dashboard，含吞吐/SM 柱状子图）| compare 脚本**自动** | 无需额外参数；吞吐已修对（scraper fix） |
| `<run>/{kunserve,baseline}/sglang_batch_timing.jsonl`（每步 iter/gap/kv）| sglang 调度器**自动直写** | 无需参数；瓶颈分析就用它 |
| **`kv_attention.png`**（瓶颈验证图，给老师用）| **手动**跑 `plot_kv_attention.py` | 无需 run 参数，读 batch_timing |
| `stage_profile.png`（per-stage **堆叠柱状图**）| **手动**跑 `plot_stage_profile.py` | 必须 run 时带 `KUNSERVE_STAGE_PROFILE=1`（否则无数据）+ 手动出图。**⚠️ 这个 profiler 数据不可信且扰动 run（见 [11.4](#11)），别用** |
| `prompt-answer.txt`（rollout 输出，权威）| verl rollout dump **自动** | 无需参数 |
| ~~`prompt_answer_streaming_r*.txt`~~ | **已失效** | compare 脚本仍 set `SGLANG_STREAMING_PROMPT_ANSWER_LOG`，但 sglang 里**写它的代码已被移除**（全工程无消费者）。这是 debug 产物，不影响正确性/分析；要恢复得重新加 writer。|

一句话：**"柱状图没出" = 没带 `KUNSERVE_STAGE_PROFILE=1` 且没手动出图，但那张图本来就坏、不该用；"streaming 没出" = writer 已从 sglang 移除。** 真正要的瓶颈图是 `kv_attention.png`（手动一行）。

### 10.1 顶层 workload 参数（两个 run 共享）

| 参数 | 默认 | 含义 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 用哪些 GPU |
| `N_GPUS_PER_NODE` | `4` | 总卡数（4 = 2 replica × TP2；80B 是 8 = 2×TP4） |
| `TRAIN_BATCH_SIZE` | `22` | rollout prompt 数 |
| `MAX_PROMPT_LENGTH` | `1024` | 最大 prompt 长度 |
| `MAX_RESPONSE_LENGTH` | `32768` | 最大生成长度 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | sglang 显存占用率 |
| `ONLY_RUN` | `both` | 只跑 `kunserve` / `baseline` / `both` |
| `OUT_ROOT` | `outputs/ab_<ts>` | 产物根目录 |
| `COOLDOWN_SEC` | `60` | 两个 run 之间冷却秒数 |
| `KUNSERVE_SCRIPT` / `BASELINE_SCRIPT` | 两个子脚本名 | 覆盖默认 train 子脚本 |

### 10.2 KunServe 后端选择（决定走哪条数据面）

| 参数 | 调优主线值 | 含义 |
|---|---|---|
| `KUNSERVE_COMM_BACKEND` | `sglang` | GLOBAL 通信用 sglang PyNccl（另一选项 `deepep` 是 legacy） |
| `KUNSERVE_CAPTURE_POLICY` | `fixed_padded` | 启用 GLOBAL CUDA graph（`disabled` = 不 capture，eager） |
| `KUNSERVE_ROLLOUT_QUANTIZATION` | `none` | bf16（`fp8` 给 deepep 用） |
| `KUNSERVE_MOE_A2A_BACKEND` | `none` | 不用内置 DeepEP a2a |
| `KUNSERVE_MOE_RUNNER_BACKEND` | `triton` | expert 用 Triton runner |
| `KUNSERVE_PHASE_G` | `0` | 不用 Phase G token-level a2a（legacy 实验路径） |

### 10.3 优化开关（对应第 8 节）

| 参数 | 默认 | 对应优化 | 状态 |
|---|---|---|---|
| `KUNSERVE_DISPATCH_NCCL_GROUP` | `1` | **P1** dispatch 3×AG 合并 | ✅ 有正收益 |
| `KUNSERVE_REMAP_FUSED` | `1` | **P2** remap Triton kernel | ✅ 有正收益 |
| `KUNSERVE_PAD_SKIP_WHEN_FULL` | `1` | **P3** 满 batch 跳 zero | ✅ 微正收益 |
| `KUNSERVE_STATIC_COMBINE_TP_ALLREDUCE_FUSION` | `0` | **Route A** composite op | ⚪ 默认关，无收益 |
| `KUNSERVE_COMBINE_ALT_STREAM` | `0` | **B.1** combine 移 alt stream | ⚪ 默认关，基础设施 |
| `KUNSERVE_COMBINE_CHUNKED` | `0` | **B.2** chunked overlap | 🔴 默认关，净亏损 |
| `KUNSERVE_STATIC_COMBINE_MODE` | `reduce_scatter` | combine 模式（`all_reduce` 回退） | — |

### 10.4 timing / profiling / 调试开关（正常跑别开）

| 参数 | 默认 | 含义 / 注意 |
|---|---|---|
| `KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL` | `256` | Phase E refresh 间隔（见第 7 节） |
| `KUNSERVE_FORWARD_TIMING` | `1` | 是否写 `kunserve_forward_timing.jsonl`。compare 脚本总会 set `..._LOG` 路径，所以**默认就在写**（基础 per-step 事件，能到 **~1GB**）。不需要就 `=0` 省盘。 |
| `KUNSERVE_FORWARD_TIMING_DETAIL` | `0` | 开 per-stage host detail timing + gap 子阶段。**有 host 开销**，只在专门诊断时设 1。 |
| `KUNSERVE_GRAPH_INTERNAL_TIMING` | 跟随 detail | graph 内插 CUDA event。**有开销 + 扰动**，正常别开。 |
| `KUNSERVE_GRAPH_INTERNAL_TIMING_INTERVAL` | `128`（stage_profile 时=1） | 每 N 步采一次 graph internal timing。 |
| `KUNSERVE_STAGE_PROFILE` | (off) | per-stage GPU 堆叠柱状图的数据采集（出 `kunserve_stage_profile` 摘要行）。**⚠️ 实测不可信**（GLOBAL 全 0、attention 严重低估、67-99% 落 "other"）**且每 replay 读 event 扰动 overlap 调度**（会让 `rollout_report.png` 震荡、timing 失真）。**报告别用**，要 per-kernel 用 nsys。详见 [11.4](#11)。 |
| `KUNSERVE_STAGE_PROFILE_FLUSH` | `500` | stage_profile 每 N 步落一行摘要。 |
| `KUNSERVE_STAGE_PROFILE_SAMPLE` | `1` | stage_profile 每 N 个 replay 采一次。 |
| `KUNSERVE_DISPATCH_PROBE` | (off) | 诊断 expert 路由正确性：milestone 步打 `dispatch_probe ... routing valid=X/total routed_global_experts=[...]`。**用来区分"乱码=数值精度 vs expert mapping bug"**（80B 乱码排查用）。写进 `kunserve_sglang_detail.log`。 |

> 一般跑 compare **只设 [10.2](#10) 那一组后端参数**（capture_policy / quantization / moe_a2a / moe_runner）+ GPU。上面这些 timing/profile 开关默认全关，开了反而污染数据。

### 10.5 VMM 必需开关（在子脚本里 export，不在 compare 顶层）

| 参数 | 含义 |
|---|---|
| `SGLANG_EXPERIMENTAL_VMM_MOE_WEIGHTS=1` | MoE 权重放 VMM（balloon borrow 的前提） |
| `SGLANG_EXPERIMENTAL_VMM_KV_CACHE=1` | KV cache 放可扩展 VMM |
| `SGLANG_EXPERIMENTAL_VMM_KV_RESERVE_SLOTS` | 预留 KV slot 数 |

---

<a name="11"></a>
## 11. 核心瓶颈：为什么 KunServe 比 baseline 慢，且现有手段无法补上

### 11.1 方法论：哪些数据可信（先看）

| 数据 | 可信？ | 依据 |
|---|---|---|
| `iter_ms`/`gap_ms`/`step_ms`/`total_kv_tokens`（`sglang_batch_timing.jsonl`）| ✅ | 已逐行核对插桩代码（下）|
| temp=0 + 真实 prompt 文本配对的输出对照 | ✅ | verl rollout dump（原生，非自加）|
| per-stage `KUNSERVE_STAGE_PROFILE` CUDA-event | ❌ 坏 | capture 时冻结，GLOBAL 读 0、attention 低估 |

**插桩核对**（`scheduler.py` `_log_decode_step_timing` + `run_batch`）：
`total_kv_tokens = Σ(len(prompt)+len(output))` = 真实 KV；`gap_ms` = 相邻 run_batch 间隔（含
`copy_done.sync` 等上一步 GPU）；`step_ms` = run_batch 体耗时；`iter_ms = step_ms + gap_ms` =
**每步真实墙钟**，Σiter 电报式累加 = decode 总墙钟。**正确，可用于端到端与 per-step 分析。**

### 11.2 端到端对照（temp=0 确定性 run `ab_20260530_053452`）

temp=0 贪心 + `shuffle=False` 同一批 prompt → 逐 prompt 文本可配对，消除采样噪声：

| | baseline | kunserve | 差异 |
|---|---|---|---|
| 逐 prompt 配对长度比（中位）| — | — | **1.000（输出长度一致）** |
| 最终答案一致率 | — | — | **91%**（其余多为难题/LaTeX 格式）|
| **decode 总墙钟 `Σiter`** | **2567 s** | **2660 s** | **+3.6%** |
| decode 步数 | 78089 | 60296 | kunserve 更少（balloon 并发更高）|
| 总 attention 读取 `ΣKV` | 25.22 B | 24.09 B | kunserve 略少 |

→ **kunserve 与 baseline 产出相同，只慢 +3.6%。** 旧的 "+13.5% / 输出长 9%" 来自 temp=1.0 采样方差 +
`req_lifecycle` 里 baseline 17 个 abort 残缺记录，**已作废**。

### 11.3 +3.6% 的唯一来源 = 跨实例 EP（baseline vs GLOBAL，matched bs+KV，已验证 +5.5ms）

**正确的对照是 baseline vs kunserve-GLOBAL**（不是 GLOBAL-vs-LOCAL——后者 bs 几乎不重叠，外推不可靠）：
- **先验证 kunserve-LOCAL（balloon 关）= baseline**：matched bs 逐档对照 Δ≈0（bs=88 +0.2、bs=76 −0.5）→
  关 balloon 时 kunserve 和 baseline 一样，没有 VMM/dispatcher 额外开销。差异全部出在 GLOBAL 路径。
- **baseline vs GLOBAL，matched bs 且 KV 也匹配（KV差≈0 的档）**：bs=18 +5.8、bs=21 +6.8、bs=9 +5.4、
  bs=6 +5.2、bs=5 +5.8 →

> **跨实例 EP（GLOBAL 路径）净开销 = ~+5.5 ms/步**（= dispatch all-gather + combine reduce-scatter
> + union expert 处理 2× 行）。**这就是 kunserve 比 baseline 慢的来源。**

**端到端账本（为什么毛开销大、净只 +3.6%）：**

| 项 | 值 |
|---|---|
| 跨实例 EP 毛开销 | +5.5ms × 48667 GLOBAL步 = **+268 s** |
| balloon 高并发省（kunserve 少跑 17793 步 × ~9ms 固定开销） | **−175 s** |
| **净** | **+93 s = +3.6%** |

→ **balloon 的免排队/高并发本可让 kunserve 快 175s，但跨实例 EP 的 +268s 吃掉还倒亏 93s。**
**break-even：跨实例 EP ≤ 3.6ms/步（268→175s）追平 baseline，再低反超。现在 5.5ms，要砍 ~2ms。**

> ❌ 旧（上一版）"+2.8ms"是 GLOBAL-vs-kunserve-LOCAL matched-bs，但二者 bs 范围几乎不重叠（LOCAL 均值 85、
> GLOBAL 均值 32），是外推、低估。本节 baseline-vs-GLOBAL 大量重叠，+5.5ms 可信。

### 11.3.5 【2026-06-01 修订】Kineto 逐阶段：同 bs 每步差是 +4~+39ms 变量，不是恒定 +5.5ms

§11.3 的"+5.5ms"是在 **matched bs + matched KV + 两 replica 均衡** 三个条件都满足时测的，那是"纯跨实例传输"。
但真实 rollout 大部分步不满足后两个条件。用 sglang `/start_profile`（Kineto/CUPTI，能看进 CUDA graph replay 内部，
不像坏掉的 stage-event）做了两次完整干净 run：`outputs/prof_base4`（baseline 17 bucket bs7-88）、
`outputs/prof_kun3`（kunserve-GLOBAL 15 bucket bs6-79，gpu_mem_util=0.80 防 profiler OOM）。
脚本 `profile_orch.py`（后台每 5 个 bs-bucket 触发，`variant=global` 只抓 balloon-on）+ `parse_stage_traces.py`
（按 manifest trigger_ts 配 trace、kernel 名→~10 阶段、FlashAttn 计数/48 估步数）。产物 `stage_compare_full.csv` + `stage_compare_full_bar.png`。

**同 bs 每步差 = +4~+39ms，四块组成：**

| 块 | 大小 | 性质 |
|---|---|---|
| ① 纯跨实例传输（AG~2.4 + RS~2.7 + remap~0.1） | **~5ms 恒定** | 9 个均衡 bucket 都是这个值，= §11.3 的 +5.5ms |
| ② **跨实例 lockstep 同步等待（旧版漏了）** | **+13~26ms，6/15 bucket** | replica-0 的 ncclAllGather **在 GPU 上空转等负载更重的 replica-1**；bs54 AG=26ms。lockstep 强制两 replica 步调一致，谁慢另一个就在 AG 里等。低 bs/rollout 后期（序列陆续结束→失衡）更频繁 |
| ③ attention | +0~+16ms | 同 bs 下 kunserve KV 更高→更大；**同 KV 下相等**（bs74: 25.5 vs 27.9）→ 非差异点，只是"同 bs 长度差异"经 KV 真实贡献 |
| ④ expert | −3~+2ms | 高 bs kunserve 更便宜（每卡 32 vs 64 expert），低 bs ~中性 |

**交叉验证（关键，排除 profiler 伪影）**：用**非 profiled** 的 batch_timing `iter_ms` 中位（整个 GLOBAL run）对每个 bs，
与 profiled 窗口 kernel 总和对比，**15/15 bucket PROFtot≈REAL**（bs54 PROF72.6≈REAL72.7、bs64 64.3≈64.6、bs57 70.3≈74.4）。
⇒ ② 的 sync 等待是**真 GPU 墙钟**，不是 profiler 放大。真实同 bs 差 kun−base = **+3.7~+38.8ms**（无 profiler）。

**对优化的含义（修正方向）**：旧版"break-even 砍传输到 3.6ms"只针对 ①。真正的大头是 ② **lockstep 同步等待** + ③ attention-KV。
所以优化应转向：**Phase E（idle keepalive / lockstep 形式化）+ 跨实例负载均衡**（让两 replica 序列长度/数量对齐，
减少 AG 空转等待）；或**合并 TP=4** 彻底消跨实例（同时吃到并发红利）。单纯 overlap/砍传输（P1/B.2）只动 ①，收益有限。

### 11.4 per-step 组成（attention 是最大块，但 **不是** kunserve 慢的原因）

`plot_stage_bars.py`（temp=0 run，LOCAL vs GLOBAL；下表的跨实例值用 §11.3 的 baseline-vs-GLOBAL +5.5ms 为准）：

| 阶段组 | LOCAL (kv≈296K) | GLOBAL (kv≈430K) | 说明 |
|---|---|---|---|
| attention（读 KV，0.055ms/1K）| 16.3 ms (46%) | 23.6 ms (50%) | **baseline 完全一样**；GLOBAL 高仅因 KV 更高（并发），非 balloon |
| 跨实例 EP [KunServe 独有] | — | **~5.5 ms** | ← **唯一差异点**（baseline vs GLOBAL matched bs+KV）|
| MoE+其它（expert/GEMM/norm/lm_head）| 19.2 ms | 20.8 ms | 一块，未能可靠细分（需 nsys）|

**要点**：attention 是单步**最大组成**，但 baseline 和 kunserve **同 KV 下一模一样** → 它**不是**两者差异的来源。
（⚠️ 本表的"跨实例 EP ~+5.5ms"只是 §11.3.5 的 ① 纯传输那块；真实同 bs 差是 +4~+39ms 变量，大头是 ② lockstep 同步等待，见 §11.3.5。）
报告用图：`kv_attention.png`（iter vs KV 散点+拟合）+ `stage_bars.png`。**注意 stage_bars.py 里那条
"cross-replica" 用的是 GLOBAL-vs-LOCAL 残差（偏低），真实值以 §11.3 的 +5.5ms 为准。**

### 11.5 不可信的工具（保留警告）

`KUNSERVE_STAGE_PROFILE` 的 per-stage CUDA-event 在 sglang 真实 graph 里 **capture 时冻结**
（GLOBAL 读 ~0、LOCAL attention 只读到真值的 1/5、67-99% 落 "other"），且 `interval=1` 读 event 扰动 overlap 调度。
**别用它出报告。** 要真·per-kernel 细分（那 +5.5ms 里 NCCL（AG/RS）vs union expert 各占多少）→ 用 **nsys profile**。

---

<a name="12"></a>
## 12. 后续可探索的方向

已验证结论：30B 上 KunServe 与 baseline **产出相同、decode 慢 +3.6%**，**唯一来源是跨实例 EP 的
per-step ~+5.5ms**（毛开销 +268s，被 balloon 高并发省的 −175s 抵掉大半，净 +93s）。
**所以缩小跨实例通信 = 该投的方向，且有明确目标：把 5.5ms 砍到 ≤3.6ms 就追平 baseline。** 排序：

1. **缩小跨实例 EP（5.5ms → 目标 <3.6ms）= 唯一能动的 decode 旋钮**
   P1（dispatch 3×AG 合一 ncclGroup）已落地。先用 **nsys** 看 5.5ms 里 NCCL(AG/RS) vs union-expert(2×行) 哪个大头，
   再决定做 combine/expert overlap（B.1/B.2）还是减 union 行数。
   **评估方法（重要）**：开关前后各跑一次 **temp=0**，比 **baseline-vs-GLOBAL matched bs+KV 残差**（11.3 的方法），
   **不要看端到端**（采样/并发噪声大，会误导，见历史教训）。

2. **nsys 细分那 +5.5ms**：NCCL（all-gather/reduce-scatter）vs union expert 处理各占多少，定优化重点。
   这是唯一可靠的 per-kernel 分解（per-stage event profiler 已证不可信）。

3. **架构层：合并单实例 TP=4** —— 直接消除跨实例 EP 的 5.5ms，理论上不仅追平、还能拿到 balloon 的并发红利
   （那 −175s）→ 比 baseline 快。代价是失去 balloon 借对端 expert 显存扩 KV 的能力。收益最确定。

4. **别忘了报告 balloon 的收益面**：temp=0 下 baseline 有 **17 次 abort**（KV 压力被迫中断），kunserve **0 次**。
   balloon 避免了排队/中断——KunServe 用 +3.6% 的 decode 开销换来了无排队。报告应成本/收益一起说。

### 开关总表

| 开关 | 默认 | 作用 |
|---|---|---|
| `KUNSERVE_DISPATCH_NCCL_GROUP` (P1) | 1 | dispatch 3×AG 合一，减跨实例 launch |
| `KUNSERVE_REMAP_FUSED` (P2) | 1 | remap Triton kernel |
| `KUNSERVE_PAD_SKIP_WHEN_FULL` (P3) | 1 | 满 batch 跳 zero |
| `KUNSERVE_STATIC_COMBINE_TP_ALLREDUCE_FUSION` (Route A) | 0 | combine+TP 融合（需 temp=0 matched-bs 重测）|
| `KUNSERVE_COMBINE_ALT_STREAM` / `_CHUNKED` (B.1/B.2) | 0 | combine overlap（需 temp=0 matched-bs 重测）|
| `KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL` | 256 | Phase E 协商间隔 |

> P1/P2/P3 之前标的"+1.6%/收益"是 temp=1.0 端到端测的，**噪声内不可信**；要论收益必须用 temp=0 + matched-bs 残差重测。

**一句话总结（2026-05-31 验证版）**：30B 上 **KunServe 与 baseline 产出相同**（temp=0 逐 prompt 配对：
长度比 1.000、91% 答案一致、无乱码、无变长），**decode 慢 +3.6%**（`Σiter` 2660 vs 2567s），**唯一来源是
跨实例 EP 的 per-step ~+5.5ms**（dispatch all-gather + combine reduce-scatter + union expert 2×行；baseline-vs-GLOBAL
matched bs+KV 测得；kunserve-LOCAL 已验证 = baseline）。账本：毛开销 +268s（5.5ms×48667 GLOBAL步），balloon 高并发
省 −175s（少跑 17793 步），净 +93s。**break-even：跨实例 EP ≤3.6ms 追平、再低反超 baseline。** attention 是单步最大
组成但 baseline 一模一样、**非差异点**。**跨实例通信优化或合并 TP=4 是该投方向。** 插桩 `iter_ms`/`KV` 已验证正确；
`stage_profile` per-stage event 坏、勿用；细分那 5.5ms 用 nsys。**作废历史**：① "9.4ms NCCL"（INTERNAL_TIMING 扭曲）；
② "+13.5%/输出长 9%"（采样方差+abort 残缺）；③ "跨实例 ~1ms/~2.8ms 非瓶颈、瓶颈是 attention、点错方向"
（残差外推/截距偏高低估了；正确是 baseline-vs-GLOBAL +5.5ms，跨实例 EP 就是瓶颈，方向没点错）。
