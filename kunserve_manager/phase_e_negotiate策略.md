# Phase E Negotiate 策略

更新时间：2026-05-27

本文单独说明当前 Phase E negotiate 策略。目标是在 GLOBAL CUDA graph replay 下减少逐 token 跨 replica 协商，同时保持 shape 和 collective participation 正确。

## 1. 背景

GLOBAL graph replay 有两个硬约束：

1. 所有 lane rank 必须进入同样顺序的 collective。
2. 同一 collective 的 tensor shape 必须一致。

因此每个 decode step 都需要知道所有 replica 的有效 graph bucket。最朴素做法是每步 collective negotiate，但 timing 显示这会造成很大的 scheduler gap，吞吐下降明显。

## 2. 当前策略

当前策略是 “低频 lockstep collective refresh + 本地安全失效 + 窗口内复用 cache”。它不是让某个 rank 因为本地 batch 变化而单独进入 collective；这种纯本地事件触发会和 peer 的 cache reuse 形成死锁。

默认窗口：

```bash
KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=256
```

每个 refresh step：

1. 本 rank 计算 `local_bs`。
2. 根据 fixed padded capture buckets 算 `local_padded_bs`。
3. 判断本 rank 是否需要 eager：EXTEND/mixed prefill 会设置 `local_force_eager=True`。
4. 通过 runtime_group collective 交换所有 rank 的 `(local_padded_bs, local_force_eager)`。
5. 得到：
   - `max_bs`
   - `min_bs`
   - `any_force_eager`
6. 如果可 cache，则缓存这组三元组，剩余步数设为 `interval - 1`。

每个 cache step：

1. 不做 collective。
2. 本 rank 检查自己的 `local_padded_bs` 是否仍小于等于 `cached_max_bs`。
3. 如果安全，复用 cached decision。
4. 小 batch 通过 `graph_bs_override=cached_max_bs` pad 到 cached bucket replay。
5. idle rank 构造 target bs 为 `cached_max_bs` 的 keepalive batch。

## 3. 状态变量

代码位置：`scheduler.py`

```text
_phase_e_negotiate_interval
_phase_e_cache_valid
_phase_e_cached_max_bs
_phase_e_cached_min_bs
_phase_e_cached_any_force_eager
_phase_e_cached_steps_left
```

关键 helper：

```text
_phase_e_cache_enabled()
_phase_e_cache_due()
_phase_e_try_reuse_cached_decision()
_phase_e_update_cached_decision()
_phase_e_reset_cached_decision()
_phase_e_may_defer_prefill_for_cache()
```

## 4. Decision 表

| 场景 | 行为 |
| --- | --- |
| 所有 replica decode bucket 一样 | replay 该 bucket |
| busy/busy bucket 不同 | 所有 rank replay `max_bs` bucket，小 batch pad |
| idle/busy | idle rank 发 `ForwardMode.IDLE` keepalive，target bs 为 `max_bs` |
| 任一 rank EXTEND/mixed | 所有 rank force eager |
| 所有 rank idle | `max_bs=0`，停止 keepalive，不 run batch |
| cache step 本地增长超过 cached max | reset cache，下一步 collective refresh |
| cache step 本地有 prefill 等待 | 暂缓 prefill admission 到 refresh step |

## 5. 为什么不是永远固定最大 bucket

永远固定到最大 bucket，例如一直 pad 到 8 或更大，确实能避免 negotiate，但会在 batch 慢慢变小时浪费大量计算和通信。

当前策略只在短窗口内固定 bucket：

- batch 稳定时，跳过多数 negotiate。
- 本地 batch 增长时，立即失效。
- interval 到期后，观察 peer 状态。
- peer shrink 不能无通信实时观察，因此最多滞后一个 interval。

这就是当前在性能和正确性之间的折中。

## 6. 为什么不能纯 “只有 batch 变化时 negotiate”

本 rank 可以知道自己的 batch 是否变化，但不知道 peer replica 是否变化。若完全不通信，peer 从 8 降到 4 时，本 rank 无法立即知道。

因此当前实现是：

- 本地变化导致不安全时立刻 reset，并等待 lockstep refresh。
- peer 变化通过低频 refresh 观察。
- interval 控制最坏滞后和通信开销；默认从 16 放大到 256，以减少 steady decode 中重复的 `max=8/min=8` negotiation。

如果未来 manager heartbeat 能低成本广播 peer batch bucket，可以进一步接近真正的 “变化时 negotiate”。

## 7. Prefill 规则

cache step 只对 decode graph 安全。若本地突然 admission 一个 prefill/extend，而 peer 仍 replay decode graph，会出现 graph/eager 不一致。

因此：

- cache 未到期时，如果 waiting queue 有 prefill，先 defer。
- 到 refresh step 时，collective 会看到 `local_force_eager=True`。
- 所有 rank 该 step force eager。
- eager step 后重新判断能否 cache。

## 8. Keepalive 规则

当 `max_bs > 0` 但本地 `batch is None`：

- 构造 `ForwardMode.IDLE` batch。
- `target_bs=max_bs`。
- 使用 dummy KV slot 和 phantom req_pool row。
- 参与 graph replay 和 lane collective。
- 不产生真实 token。

当 refresh 得到 `max_bs=0`：

- 所有 replica idle。
- stop keepalive。
- 不 run batch。

cache 窗口内如果所有 replica 已经 idle，但 cached `max_bs > 0`，会多跑至多 `cached_steps_left` 次 keepalive；下一次 refresh 会停掉。

## 9. release cleanup

rollout 结束时可能在 cache 窗口尾部触发 `release_memory_occupation`。此时真实请求已经结束，但 overlap `result_queue` 可能还有一个 IDLE keepalive result。

当前 release 前 cleanup 会：

1. drain `result_queue`。
2. stop keepalive。
3. reset Phase E cache。
4. 清理 graph bucket override。
5. 再执行 SGLang 原生 no-request 断言。

这样避免 harmless keepalive residue 被误判为 ongoing request。

## 10. Timing 观测

看 `kunserve_forward_timing.jsonl`：

```text
phase_e_negotiated source=collective
phase_e_negotiated source=cache
phase_e_cache_update
phase_e_cache_reuse
phase_e_cache_reset
phase_e_step_decision
scheduler_gap_phase_e_negotiate_end
kunserve_memory_release_cleanup
```

期望现象：

- steady decode 中 `source=cache` 明显多于 `source=collective`。
- `scheduler_gap_phase_e_negotiate_end` 不再每步出现。
- mismatched decode 时 `graph_bs_override=max_bs`，不是直接 eager。
- release 前若有 idle residue，会看到 `kunserve_memory_release_cleanup`。

## 11. 调参建议

- `KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=1`：退回每步 negotiate，适合 debug。
- `KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=8` 或 `16`：更保守，peer shrink/prefill admission 滞后较小，但 negotiation 开销更高。
- `KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=256`：当前默认，用于降低 steady decode 中重复协商。
- 更大 interval：进一步降低协商开销，但增加 padding/keepalive 浪费和 prefill admission 滞后。

性能实验应同时看：

- throughput；
- `scheduler_gap_*`；
- cache/collective 比例；
- padding 后 graph bucket 分布；
- prompt answer streaming 正确性。
