# KunServe Global CUDA Graph 链路说明

更新时间：2026-05-27

本文只保留当前有效链路：`sglang` GLOBAL MoE backend、`fixed_padded` capture policy、KunServe PyNccl registered collective、post-commit capture、Phase E keepalive 和 cached negotiate。

## 1. 当前运行配置

当前性能调优主线使用：

```bash
ONLY_RUN=kunserve
KUNSERVE_PHASE_G=0
KUNSERVE_CAPTURE_POLICY=fixed_padded
KUNSERVE_ROLLOUT_QUANTIZATION=none
KUNSERVE_MOE_A2A_BACKEND=none
KUNSERVE_MOE_RUNNER_BACKEND=triton
KUNSERVE_FORWARD_TIMING_DETAIL=1
```

控制面建立三类通信组：

- `runtime_group`：4 个 rank 的 GLOBAL group，用于全局状态协商和部分 runtime collective。
- `lane_group[0]`：两个 replica 的 TP lane 0。
- `lane_group[1]`：两个 replica 的 TP lane 1。

这些 group 当前走 `kunserve_pynccl`，即 KunServe 自己注册到 SGLang 通信层的 PyNccl/registered collective。目标是让 GLOBAL graph 内的 lane all-gather / all-reduce 与 SGLang 原生 TP all-reduce 一样可 capture、可 replay。

## 2. 固定例子：两个 TP=2 replica，128 experts

假设有两个 SGLang 实例，每个实例 `TP=2`，共 4 张 GPU，模型有 128 个 routed experts。

原生单实例内：

- TP lane 0 负责本实例 local expert 0..63 的前半空间。
- TP lane 1 负责本实例 local expert 64..127 的后半空间。
- Attention 的 `o_proj` 后，本实例内 `tp_group.all_reduce` 让两张 TP 卡都拿到本实例完整 hidden state。

进入 BALLOON 后，每张卡只保留 32 个 physical experts：

| rank | replica | TP lane | 保留 physical experts |
| --- | --- | --- | --- |
| rank 0 | replica 0 | lane 0 | 0..31 |
| rank 1 | replica 0 | lane 1 | 64..95 |
| rank 2 | replica 1 | lane 0 | 32..63 |
| rank 3 | replica 1 | lane 1 | 96..127 |

一层 decode 的 GLOBAL MoE 数据流：

1. 本实例 attention/o_proj 完成后，SGLang 原生 TP all-reduce 已经发生。
2. 每个 TP lane 都有本实例所有请求的 hidden state。
3. KunServe dispatcher 在 lane group 内交换 hidden/topk：
   - lane 0：replica 0 lane 0 与 replica 1 lane 0 互换。
   - lane 1：replica 0 lane 1 与 replica 1 lane 1 互换。
4. 每张卡只计算自己保留的 32 个 experts。
5. lane 内 combine 后，只取回属于本 replica 请求的输出。
6. FusedMoE 层外侧继续走 SGLang 原生本实例 TP all-reduce，把 lane 0 和 lane 1 的 partial expert 输出合成该实例的完整 MoE 输出。

这里 lane 通信是必要的。它不是额外设计负担，而是保持 “attention/KV cache 仍归属原 replica，expert 计算跨 replica 补齐” 的核心约束。

## 3. 为什么不用 pre-capture

旧方案尝试在 `warmup_balloon` 阶段预先 capture GLOBAL CUDA graph。这个方向已删除为主线，因为 `commit_balloon` 会改变 graph 依赖的最终内存拓扑：

- expert VMM 权重 donor pages 被 borrow 给 KV cache；
- KV cache VMM 容量扩展；
- MoE live expert metadata 切到 GLOBAL physical mapping；
- keepalive dummy KV slot 和 phantom req_pool row 被预留；
- GLOBAL runtime bundle 在最终状态下重新绑定。

SGLang 原生 CUDA graph 的前提是 capture 后权重/KV/buffer pointer 稳定。pre-commit capture 违反这个前提，曾导致 BALLOON 后乱码和异步 CUDA illegal memory access。

当前策略：

1. `warmup_balloon` 只注册 process group、GLOBAL bundle、capture 配置，不启用最终 GLOBAL graph。
2. `commit_balloon` 先暂停 graph replay。
3. 完成 expert borrow、KV expand、runtime switch、dummy/phantom padding 资源预留。
4. 丢弃旧 GLOBAL graph。
5. 在最终 post-balloon 内存布局上 capture GLOBAL graph。
6. capture 成功后设置 `_balloon_graph_replay_enabled=True`。

实测 post-commit capture 很快，因此目前优先保持这个策略。

## 4. fixed_padded graph replay

`fixed_padded` 的含义：

- CUDA graph bucket 使用固定 batch size，例如 `1,2,4,8,...`。
- replay 时如果真实 `raw_bs=7`，选择 `graph_bs=8`。
- 多出来的 padding 行写入 dedicated dummy KV slot，并使用 phantom req_pool row，避免污染真实请求。

`CudaGraphRunner.replay_prepare()` 会把真实 batch 复制到 graph input buffer；KunServe 对 padding 行做额外 patch：

- `input_ids=0`
- `seq_lens=1`
- `positions=0`
- `out_cache_loc=dummy_kv_slot`
- `req_pool_indices=phantom_req_idx`
- `topk_ids=-1` 或对应 dispatcher 的无效 expert 标记

这样 padding 行会参与 graph shape 和 collective，但不会贡献真实输出。

## 5. Phase E keepalive

GLOBAL graph/replay 的硬约束：所有 lane rank 必须在同一步进入同样 shape 的 collective。

当 replica 0 还在 decode、replica 1 已经无真实请求时，replica 1 仍必须发一个 `ForwardMode.IDLE` keepalive batch。否则 replica 0 的 lane collective 会等待不存在的 peer，轻则 hang，重则后续 CUDA API 报错。

当前 keepalive 资源：

- `dummy_kv_slot`：`commit_balloon` 预留，IDLE batch 的 KV 写入这里。
- `phantom_req_idx`：`commit_balloon` 预留，`req_to_token[phantom_req_idx, :]` 指向 dummy KV slot。
- `ForwardMode.IDLE` batch：`reqs=[]`，但 `input_ids/seq_lens/out_cache_loc/req_pool_indices` 按 target graph bucket 构造。

## 6. Phase E cached negotiate

逐 token 都做跨 replica negotiate 的开销很高，timing 里曾看到 scheduler gap 被 negotiate 放大到 20ms 量级。因此当前策略不是每步 collective，而是缓存最近一次决策：

- 默认 `KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=16`。
- refresh step：用 `runtime_group` collective 交换 `(local_padded_bs, local_force_eager)`。
- cache step：复用上次 `cached_max_bs/cached_min_bs/cached_any_force_eager`。
- cache step 上所有 rank 都 replay `cached_max_bs` 对应 graph bucket，小 batch padding 到该 bucket。

安全规则：

- 本地 batch 增长超过 `cached_max_bs`：立即 reset cache，下一步 collective negotiate。
- 本地出现 EXTEND/mixed prefill：reset cache，并 force eager。
- cache 未到期时若 waiting queue 有 prefill：暂缓 prefill 到下一次 refresh，避免一边 replay decode graph、一边另一个 rank 需要 eager。
- cache 到期后必须 collective refresh；如果所有 rank idle，`max_bs=0`，停止 keepalive。

这不是 “永久固定 batch=8”。它是 bounded cached decision：在一个短窗口内固定 graph bucket，窗口结束或本地 shape 不安全时重新协商。

## 7. 2026-05-27 release bug

失败日志：`/workspace/verl/outputs/ab_20260527_020500/kunserve/verl_training.log`

直接错误：

```text
AssertionError: release_memory_occupation should be called only when no ongoing request.
```

timing 文件显示最后真实请求已经结束：

- `running=0`
- `waiting=0`
- `last_batch_mode=ForwardMode.IDLE`
- `result_queue_len=1`
- Phase E cache 还在复用 `cached_max_bs=1`

原因：`release_memory_occupation` 是 control request，在 overlap scheduler loop 顶部处理。上一轮 IDLE keepalive 的结果还在 `result_queue` 中没有 pop/process，原生 `_is_no_request()` 因 `result_queue_len > 0` 认为仍有 ongoing request。

当前修复：

- `release_memory_occupation` 断言前调用 `_kunserve_prepare_for_memory_release()`。
- cleanup 会 drain overlap `result_queue`，处理已完成的 IDLE/上一轮结果。
- 停止 keepalive，reset Phase E cache。
- 清理 per-step force eager / graph bucket override。
- 只在 cleanup 后继续使用原生 `_is_no_request()` 断言；如果仍有真实 running request，断言仍然会失败。

新增日志：

- `KunServeScheduler memory release cleanup`
- `kunserve_memory_release_cleanup` timing event

## 8. 当前剩余性能问题

1. GLOBAL graph replay 已能正确输出，但 throughput 仍低于原生单实例 TP=4。
2. lane combine 当前保守使用 all-reduce + slice，带宽比 reduce-scatter 大。
3. Phase E cache 只能减少 negotiate 频率；peer batch shrink 在 cache 窗口内不会被立即观察到，会多跑少量 padding/keepalive。
4. scheduler Python gap 仍然需要继续细分，尤其是 control polling、result processing、prefill admission 与 graph replay 之间的空隙。
5. RESTORE 仍不是当前主线，`enable_restore=False`。
