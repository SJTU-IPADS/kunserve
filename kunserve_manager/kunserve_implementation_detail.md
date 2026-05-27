# KunServe Implementation Detail

更新时间：2026-05-27

本文记录当前代码实现，不保留旧 patch 日志。当前主线是 `sglang` GLOBAL backend、KunServe PyNccl registered collective、post-commit fixed-padded CUDA graph、Phase E keepalive/cached negotiate。

## 1. 关键目录

```text
/workspace/sglang
  python/sglang/srt/model_executor/
    model_runner.py
    cuda_graph_runner.py
    forward_batch_info.py
  python/sglang/srt/managers/
    scheduler.py
    scheduler_update_weights_mixin.py
    scheduler_dp_attn_mixin.py
    schedule_batch.py
  python/sglang/srt/layers/moe/
    ...
  python/sglang/srt/kunserve_forward_timing.py

/workspace/sglang/kunserve_manager
  kunserve_manager/
    controller.py
    client.py
    layout.py

/workspace/verl
  data/compare_kunserve_vs_baseline.sh
  data/train/run_smoke_test_kunserve_tp2_dual_replica.sh
```

## 2. 控制面链路

`KunServeController` 负责：

1. 发现两个 SGLang replica。
2. 建立 `runtime_group` 和 `lane_group`。
3. 调用 `warmup_balloon` 注册 GLOBAL runtime bundle。
4. 轮询 `/kunserve/status`。
5. 看到 `expand_requested=True` 后执行：
   - `prepare_balloon`
   - `commit_balloon`
6. BALLOON 后继续轮询状态，只做观测和日志。

当前 `controller.py` 中 asymmetric drain 日志已更新：不再说 “还没有 dummy participation”，而是明确提示该路径依赖 scheduler Phase E keepalive。

## 3. Runtime group 初始化

`kunserve_pynccl` 进程组用于 graph-safe registered collective。

当前 group：

- `kunserve_global_ep_*`：4 ranks。
- `kunserve_lane0_*`：两个 replica 的 TP lane 0。
- `kunserve_lane1_*`：两个 replica 的 TP lane 1。

lane group 只让对应 TP lane 参加。非参与 TP rank 会跳过该 lane group 初始化。

## 4. GLOBAL expert layout

固定例子：两个 replica，每个 `TP=2`，128 experts。

| rank | replica | TP lane | retained experts |
| --- | --- | --- | --- |
| 0 | r0 | 0 | 0..31 |
| 1 | r0 | 1 | 64..95 |
| 2 | r1 | 0 | 32..63 |
| 3 | r1 | 1 | 96..127 |

实现要点：

- `layout.py` 生成 retained/offloaded layout。
- `active_local_expert_mapping` 不能把所有 replica 都强制映射到 0..31。
- `physical_to_logical_map` 保持 128 expert 的全局一致语义。
- `model_runner.register_balloon_global_runtime_bundle()` 将 layout 下发给每个 MoE layer。

之前出现过 “replica 1 也读 0..31” 的错误，会导致 experts 32..63/96..127 永远不算，输出尾部乱码。当前文档和代码都按上表理解。

## 5. VMM commit

`model_runner.commit_balloon()` 当前负责：

1. 校验 prepared GLOBAL runtime。
2. 关闭当前 step graph replay。
3. 从 offloaded local experts 对应 VMM pages borrow donor。
4. 将 donor pages remap 给 KV cache。
5. 同步 `max_total_num_tokens`。
6. 更新 MoE live expert metadata。
7. 预留 `dummy_kv_slot`。
8. 预留 `phantom_req_idx`，并把 `req_to_token[phantom_req_idx, :]` 指向 dummy KV slot。
9. 切换到 GLOBAL runtime。
10. 在 post-commit 最终内存拓扑上 capture GLOBAL CUDA graph。

`scheduler.commit_balloon()` 会同步 scheduler 侧容量缓存，并在容量增长时清掉 `running_batch.batch_is_full`，让之前因 KV 不足 retract 的请求有机会重新进入 prefill。

## 6. CUDA graph capture/replay

当前策略：

- `warmup_balloon`：注册 bundle 和通信信息，不启用最终 graph。
- `commit_balloon`：完成 VMM/KV/phantom 后 post-commit capture。
- steady decode：`CudaGraphRunner` 选择 fixed padded bucket replay。

`cuda_graph_runner.py` 中 replay 需要支持：

- `graph_bs_override`：Phase E 让小 batch 强制 replay cached/negotiated max bucket。
- padding 行 patch：使用 dummy KV slot 和 phantom req_pool row。
- raw logits 截断：只返回真实 `raw_bs` 行对应输出。

## 7. Scheduler Phase E

代码位置：`python/sglang/srt/managers/scheduler.py`

Phase E 只在以下条件同时满足时启用：

- model_runner state 是 `balloon`；
- backend 是 `sglang`；
- runtime variant 是 `global`；
- process group 已存在。

核心函数：

- `_kunserve_phase_e_active()`
- `_phase_e_get_step_decision()`
- `_phase_e_apply_step_decision()`
- `_build_balloon_keepalive_batch()`
- `_phase_e_reset_cached_decision()`
- `_phase_e_try_reuse_cached_decision()`
- `_phase_e_update_cached_decision()`
- `_phase_e_may_defer_prefill_for_cache()`

Phase E 每步给出：

```text
max_bs
min_bs
any_force_eager
from_cache
```

含义：

- `max_bs`：本 step 所有 replica 应 replay/keepalive 的最大 graph bucket。
- `min_bs`：用于判断是否 idle/busy 或 busy/busy mismatch。
- `any_force_eager`：有任一 rank 是 EXTEND/mixed prefill 时，所有 rank skip graph。
- `from_cache`：本 step 是否复用 cached decision。

## 8. cached negotiate

环境变量：

```bash
KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=256
```

默认每 256 步做一次 lockstep collective refresh，其余步骤复用 cache。这个 refresh 是安全边界：单个 rank 不能只因为本地 batch 变化就独自进入 negotiate，否则 peer rank 可能仍复用 cache。

cache 可用条件：

- GLOBAL graph replay active。
- negotiated `max_bs > 0`。
- `any_force_eager=False`。
- `max_bs` 是已 capture 的 bucket。

cache 失效条件：

- 本地 padded bs 大于 cached max bs。
- 本地需要 force eager。
- interval 到期。
- prepare/commit/restore balloon。
- memory release cleanup。

prefill 处理：

- cache 未到期且 waiting queue 有 prefill 时，暂缓 admission。
- 下一次 refresh step 统一 negotiate，如果出现 EXTEND/mixed，所有 rank force eager。

## 9. Keepalive batch

代码位置：

- `scheduler_dp_attn_mixin.py:get_idle_batch()`
- `schedule_batch.py:prepare_for_idle()`
- `scheduler.py:_build_balloon_keepalive_batch()`

当 `target_bs > 0`：

- `forward_mode=ForwardMode.IDLE`
- `reqs=[]`
- `input_ids=zeros(target_bs)`
- `seq_lens=ones(target_bs)`
- `out_cache_loc=full(target_bs, dummy_kv_slot)`
- `req_pool_indices=full(target_bs, phantom_req_idx)`

它的作用是参与 graph shape 和 lane collective，不产生真实输出。

## 10. release_memory_occupation bug 修复

失败 run：

```text
/workspace/verl/outputs/ab_20260527_020500/kunserve/verl_training.log
```

根因：

- rollout 结束后 `free_cache_engine=True` 触发 `release_memory_occupation`。
- scheduler overlap loop 顶部先处理 control request。
- 上一轮 Phase E IDLE keepalive result 还在 `result_queue`。
- `_is_no_request()` 因 `result_queue_len=1` 返回 false。
- 触发断言，随后 Ray 关闭导致二次 CUDA `invalid argument`。

修复文件：

- `scheduler_update_weights_mixin.py`
- `scheduler.py`

新增行为：

1. `release_memory_occupation()` 断言前查找并调用 `_kunserve_prepare_for_memory_release()`。
2. cleanup drain overlap `result_queue`。
3. stop balloon keepalive。
4. reset Phase E cached decision。
5. 清理 model_runner step force-eager 和 graph bucket override。
6. 清理空的 `last_batch/cur_batch`。
7. 再执行原生 `_is_no_request()`。

这样只处理 harmless overlap residue；如果真实请求仍在 running，原生断言仍会阻止 release。

## 11. Timing

`kunserve_forward_timing.py` 提供 JSONL timing。常用开关：

```bash
KUNSERVE_FORWARD_TIMING_DETAIL=1
```

关键事件：

- `scheduler_gap_recv_requests_end`
- `scheduler_gap_process_input_end`
- `scheduler_gap_get_next_batch_end`
- `scheduler_gap_phase_e_negotiate_end`
- `phase_e_negotiated`
- `phase_e_cache_update`
- `phase_e_cache_reuse`
- `phase_e_step_decision`
- `scheduler_gap_build_keepalive_end`
- `model_runner_forward_select`
- `cuda_graph_replay_selected`
- `scheduler_run_batch_begin/end`
- `kunserve_memory_release_cleanup`

分析脚本应按 balloon timestamp 切 pre/post，再聚合 p50/p90/mean。

## 12. 当前已解决的问题

1. BALLOON 后乱码：pre-commit graph capture 和错误 expert mapping 已从主线移除/修复。
2. last request hang：Phase E IDLE keepalive 让 idle replica 继续参与 collective。
3. mismatched decode batch 走 eager：现在可用 `graph_bs_override=max_bs`，小 batch pad 到 max bucket replay。
4. negotiate 每步过慢：引入 interval cached negotiate。
5. release 收尾断言：新增 memory release cleanup。

## 13. 当前未解决的问题

1. GLOBAL throughput 还没有达到原生单实例 TP=4。
2. lane combine 仍用 all-reduce + slice，带宽不是最优。
3. cached negotiate 对 peer shrink 只能在 refresh step 观察，窗口内会多 padding。
4. prefill admission 在 cache window 内会被延迟到 refresh step。
5. restore path 不是当前主线。

## 14. 推荐回归

先跑不带 GLOBAL graph 的正确性，再跑 graph：

```bash
ONLY_RUN=kunserve \
KUNSERVE_PHASE_G=0 \
KUNSERVE_ROLLOUT_QUANTIZATION=none \
KUNSERVE_MOE_A2A_BACKEND=none \
KUNSERVE_MOE_RUNNER_BACKEND=triton \
bash /workspace/verl/data/compare_kunserve_vs_baseline.sh
```

```bash
ONLY_RUN=kunserve \
KUNSERVE_PHASE_G=0 \
KUNSERVE_CAPTURE_POLICY=fixed_padded \
KUNSERVE_ROLLOUT_QUANTIZATION=none \
KUNSERVE_MOE_A2A_BACKEND=none \
KUNSERVE_MOE_RUNNER_BACKEND=triton \
KUNSERVE_FORWARD_TIMING_DETAIL=1 \
bash /workspace/verl/data/compare_kunserve_vs_baseline.sh
```

重点检查：

- prompt answer streaming 是否在 BALLOON 后 coherent。
- `phase_e_negotiated` 中 cache/collective 比例。
- `cuda_graph_replay_selected` 是否出现 global graph replay。
- rollout 结束是否出现 `kunserve_memory_release_cleanup`，且没有 release assertion。
