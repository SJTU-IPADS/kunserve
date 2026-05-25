# Phase E Idle Keepalive — Dilemma Doc (KunServe 2026-05-24)

> **Audience**: future KunServe maintainers picking up Phase E.
> **Status**: dispatcher correctness verified ✅, Phase E hang root-caused but unfixed ⏳.
> **Source session**: cff76992-a4f8-49fb-96b3-086aa8fa6c6f (2026-05-24).

---

## TL;DR

1. `CrossReplicaStandardDispatcher` (Phase F eager path) is **numerically correct**.
   Probe data (`KUNSERVE_DISPATCH_PROBE=1`, call=1/5/20/100/500) shows no NaN/Inf,
   stable magnitudes, cross-rank consistency. Model output reaches natural EOS
   (`finish=stop`) in BALLOON mode (multiple confirmed instances).

2. The remaining symptom is **asymmetric-workload hang**: when one replica
   finishes its requests before the other, the busy replica's
   `lane_group.all_gather()` has no participant on the idle replica and
   deadlocks. The previous "garbled output" hypothesis was a misread — the
   slow throughput (5-10x) plus high `max_new_tokens` causes most requests
   to hit `length` before they can reach EOS, but the tokens themselves are
   coherent text.

3. The Phase E scaffolding in `scheduler.py` (`_kunserve_phase_e_active`,
   `negotiate_balloon_step_bs`, `_build_balloon_keepalive_batch`,
   `prepare_for_idle(target_bs > 0)`) exists but is partially broken:
   - **Outer bug** (control flow): in `event_loop_overlap`, keepalive is built
     in the `elif batch is None:` branch at the bottom, which only fires when
     `last_batch` is **also** None. That leaves a one-step gap on the
     transition step (last real batch still in `result_queue`, new queue
     empty) where the idle replica issues no collective.
   - **Inner latent bug** (kernel safety): `prepare_for_idle(target_bs > 0)`
     constructs an IDLE batch with `req_pool_indices = zeros(n)`, which
     points every dummy token at req-pool entry 0. After the prior real
     requests are freed, the kv slots they referenced may no longer be
     accessible (especially under KunServe's VMM mapping), and the attention
     kernel's read triggers `cudaErrorIllegalAddress`. This path was never
     actually exercised in any prior session because the outer bug always
     suppressed it.

4. A naive fix that only addresses the outer bug exposes the inner bug.
   Fixing both requires a small design change. **Recommended approach**:
   pre-allocate a permanent "phantom" req_pool entry at `commit_balloon`
   time, populate `req_to_token[phantom]` with `dummy_kv_slot`, and have
   keepalive batches use `req_pool_indices = phantom_idx` so attention always
   reads valid memory.

---

## Evidence

### Dispatcher correctness probes

Run `/workspace/verl/outputs/probe3_20260524_052111/kunserve` with
`KUNSERVE_DISPATCH_PROBE=1`:

```
call=1  rank=0  hidden_in=absmax=2.69 nan=0 inf=0  union_hidden=absmax=2.69 nan=0 inf=0
call=5  rank=0  hidden_in=absmax=2.75 nan=0 inf=0  ...
call=20 rank=0  hidden_in=absmax=2.98 nan=0 inf=0  ...
call=100 rank=0 hidden_in=absmax=3.98 nan=0 inf=0  ...
call=500 rank=0 hidden_in=absmax=4.19 nan=0 inf=0  ...
```

Same numerical sanity on all four ranks across all 48 layers, both
`dispatch_probe` (after all_gather + remap) and `combine_probe` (after
reduce_scatter + slice). Cross-rank invariants hold:

- `hidden_in` matches across TP peer ranks (rank0=rank1, rank2=rank3)
- `union_hidden` matches across all 4 ranks (lane0/lane1 both produce the
  same union of replica0_tokens + replica1_tokens)
- `remapped_topk` is in `[-1, 31]` (i.e., -1 for non-local experts and 0..31
  for the rank's local expert rows after the sglang-backend identity
  override)

### Natural EOS in BALLOON mode

- `ab_20260524_022932/kunserve`: 1 `finish=stop` per replica after balloon
  commit. Random tokens never trigger EOS — these requests produced
  coherent text long enough to converge.
- `phaseE_20260524_113608/kunserve`: replica 0 `rid=791ae8fdd83e...`
  finished at 11:45:40 with `finish=stop` after 291s of post-balloon
  decoding.

### Pre-balloon LOCAL bundle text dump

`probe_20260524_040951` → `prompt-answer.txt` shows perfectly coherent math
reasoning (Qwen3-Thinking style: "Let's confirm with our parameter t...
let's denote for clarity...") truncated mid-sentence by `max_new_tokens=5000`.
The model is producing real reasoning, just slow.

---

## Hang anatomy

Default scheduler loop is `event_loop_overlap`. Two replicas, each
TP=EP=2, total 4 GPU ranks, lane subgroup membership:

- lane 0 = [global rank 0, global rank 2]  (cross-replica, same lane index)
- lane 1 = [global rank 1, global rank 3]

In BALLOON state, every MoE layer issues two cross-replica NCCL collectives
on the lane subgroup (one `all_gather_into_tensor` in dispatch, one
`reduce_scatter_tensor` in combine). 48 MoE layers × 2 collectives × every
decode step = ~96 cross-replica collectives per decode token.

```
Step N-1 (replica 1's last real batch)
  scheduler.event_loop_overlap iter N-1:
    batch = real_batch_N-1
    Phase E negotiate (runtime_group all_gather, all 4 ranks reach this)  → max=bs, min=bs (both busy)
    run_batch(real_batch_N-1) → forward, issues lane.all_gather (matches peer)
    result_queue.append(...)
    pop_and_process(last_batch=real_N-2)
    self.last_batch = real_batch_N-1

Step N (replica 1's first idle step)
  iter N on replica 1:
    batch = None  (queue empty)
    Phase E negotiate            → peer (replica 0) advertises bs=8, self advertises 0; max=8, min=0
    (currently: nothing built; batch stays None)
    No run_batch (batch is None)
    pop_and_process(last_batch=real_N-1)
    self.last_batch = None       (because batch is None)
  iter N on replica 0:
    batch = real_batch_N
    Phase E negotiate            → max=8, min=0
    run_batch(real_batch_N)
      → forward
      → MoE layer 0 dispatch.all_gather on lane_group  ← BLOCKED, peer didn't issue matching call
```

`negotiate_balloon_step_bs` succeeds at step N because it uses
`runtime_group` (4-rank), and all 4 ranks reach it. After that the busy
replica issues the lane collective and waits forever. The idle replica
proceeds to step N+1 *if it can* — but its Phase E negotiate at step N+1
needs all 4 ranks too, and the busy replica is stuck on a lane collective
from step N. So step N+1 negotiate also deadlocks. Final outcome: every
rank stuck on a different collective. Eventually killed by either NCCL
timeout (30 min default) or `SILENCE_THRESHOLD_SEC` from `run_and_scrape.sh`
(default 200 s; bump to 600 s+ for diagnostics).

---

## What a naive outer-bug fix does

Patch tried in this session: insert the keepalive build immediately after
the Phase E negotiation in `event_loop_overlap`, *before* `run_batch`, so
that on the transition step the idle replica still issues a lane collective
matching the busy replica's. The patch is structurally simple:

```python
real_batch_this_step = batch is not None
if (
    batch is None
    and phase_e_negotiated_max is not None
    and phase_e_negotiated_max > 0
    and phase_e_status is not None
):
    batch = self._build_balloon_keepalive_batch(
        int(phase_e_negotiated_max), phase_e_status
    )
elif batch is None and phase_e_negotiated_max == 0:
    self._stop_balloon_keepalive("all replicas idle")

if real_batch_this_step:
    self._stop_balloon_keepalive("scheduled real batch")
```

Result: the outer deadlock disappears (keepalive runs on the transition
step), but the keepalive forward itself crashes with
`cudaErrorIllegalAddress` on the very first invocation:

```
File ".../qwen3_moe.py", line 305, in forward_normal
    ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
File ".../expert_location_dispatch.py", line 121, in init_new
    partial_dispatch[:8].tolist()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

`partial_dispatch[:8].tolist()` is a CUDA→CPU sync that surfaces an *earlier*
asynchronous illegal access — the offending kernel ran before this point in
the same forward pass. The most likely culprit is the attention kernel
reading `kv_cache[req_to_token[req_pool_indices[i]][token_offset]]` where
`req_pool_indices = zeros(n)` from `prepare_for_idle(target_bs > 0)` and
the prior req at req_pool entry 0 has been freed (or is still in
`result_queue` but its KV slots are being released by the overlap
processing).

Run: `/workspace/verl/outputs/phaseE_20260524_113608/kunserve` (patch since
reverted).

---

## Why the existing elif at line 1333 hides this

The existing elif `_build_balloon_keepalive_batch` call only fires when
`last_batch is None AND batch is None`, i.e., the **second** idle step.
By then `pop_and_process` has cleared `last_batch`, but the underlying
`req_to_token` table may still be in the same problematic state (no live
real reqs). In practice no prior session reached this code path because
real workloads were either:

- both replicas drained in lockstep (no asymmetric workload),
- or one replica drained first and the outer bug deadlocked the run before
  the second idle step ever happened.

So this is the first session to exercise the path, and it crashes.

---

## Recommended fix path

### Phantom req_pool entry (highest confidence)

Reserve one permanent req_pool slot at `commit_balloon` time, populate its
`req_to_token` entries with `dummy_kv_slot`, and use it for all keepalive
batches:

```python
# in ModelRunner.commit_balloon (sglang backend, after dummy_kv_slot is reserved)
self._kunserve_keepalive_phantom_req_idx = self.req_to_token_pool.alloc(1)
# populate every token position of req_to_token[phantom] with dummy_kv_slot
self.req_to_token_pool.req_to_token[
    self._kunserve_keepalive_phantom_req_idx, :
].fill_(self._kunserve_keepalive_dummy_kv_slot)

# in prepare_for_idle when target_bs > 0:
self.req_pool_indices = torch.full(
    (n,), phantom_req_idx, dtype=torch.int32, device=self.device
)
```

Now attention's read pulls from a known-valid kv slot. After balloon
restore, the phantom slot is freed in the symmetric `restore_from_balloon`
path.

Apply this **and** the outer-bug fix (keepalive built before `run_batch`,
not in the bottom elif). Both pieces are needed.

### Alternative: skip the MoE collective on the transition step

Have Phase E negotiation also broadcast a "skip this step" bit so the busy
replica's forward does *not* issue lane collectives this step. Requires
both replicas to drop a decode step in lockstep. Functionally correct but
penalises the busy replica's throughput. Simpler than the phantom but uglier.

### Alternative: defer keepalive build

Process the prior `last_batch` (pop_and_process) before building the
keepalive in the same iteration. Then `req_pool` is in a known-empty
state when keepalive runs. The pop has to happen *before* `run_batch`
which contradicts the overlap design — would force eager mode on the
transition step.

---

## Open verification needed

Before declaring Phase E done, even after fixing the kernel bug:

1. Probe milestone at call=2000 should also show clean stats (not reached
   in current runs because requests finish before count=2000).
2. Run a workload where one replica explicitly idles for thousands of
   steps while the other generates — confirm `balloon_keepalive_step_ct`
   grows monotonically and lane collectives stay matched.
3. Run the existing AB compare (with `MAX_RESPONSE_LENGTH=32768`,
   `TRAIN_BATCH_SIZE=2`) and verify all 16 requests finish (most with
   `length`, a couple with `stop`), and `prompt_answer_streaming.txt`
   shows coherent text on both replicas all the way to end-of-context.

---

## Streaming dump (introduced in this session)

`async_sglang_server.py` now writes prompt + decoded answer to
`${SGLANG_STREAMING_PROMPT_ANSWER_LOG}_r${replica}.txt` immediately when
each request finishes — regardless of whether the rollout later hangs or
crashes. `compare_kunserve_vs_baseline.sh` wires this to
`${rundir}/prompt_answer_streaming.txt` so a partial AB run still leaves
behind the completed-request text for inspection.

Use this for future correctness validation: do not rely on the
post-rollout `prompt-answer.txt` dump (which only fires if
`exit_after_rollout` completes), since the Phase E hang prevents that.

---

## File pointers

- Dispatcher: `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`
- Phase E scaffolding: `/workspace/sglang/python/sglang/srt/managers/scheduler.py` (search `_kunserve_phase_e_active`, `_build_balloon_keepalive_batch`)
- Keepalive batch prep: `/workspace/sglang/python/sglang/srt/managers/schedule_batch.py:prepare_for_idle` (target_bs > 0 branch is the bug)
- Dummy KV slot reservation: `/workspace/sglang/python/sglang/srt/model_executor/model_runner.py` (search `keepalive_dummy_kv_slot`)
- Streaming dump: `/workspace/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py` (search `SGLANG_STREAMING_PROMPT_ANSWER_LOG`)
- Architecture overview: `/workspace/sglang/kunserve_manager/cross_replica_standard_like_dispatcher_方案综述.md` (see § 8 idle keepalive, § Phase E)
