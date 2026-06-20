# KunServe GLOBAL CUDA Graph / Phase E Negotiate 问题分析

本文记录 2026-06-09 在 `nsys_h20_wd_20260609_122703` 一类运行中暴露出的 GLOBAL CUDA graph 崩溃问题。重点不是给出一个临时阈值规避，而是说明当前同步协议、CUDA graph replay 条件、日志证据、为什么会崩、以及系统性修复应该从哪里下手。

相关日志：

- `/workspace/verl/outputs/nsys_h20_wd_20260609_122703/kunserve/verl_training.log`
- `/workspace/verl/outputs/nsys_h20_wd_20260609_122703/kunserve/bw_throughput.jsonl`
- `/workspace/verl/outputs/nsys_h20_wd_20260609_122703/kunserve/kunserve_sglang_detail.log`
- `/workspace/verl/outputs/nsys_h20_wd_20260609_122703/kunserve/sglang_batch_timing.jsonl`

相关代码：

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/distributed/kunserve_pynccl.py`
- `python/sglang/srt/distributed/device_communicators/pynccl.py`

## 1. 现象

这次不是普通的 Python exception，也不是 manager 的 `/kunserve/status` 首先卡住。

`verl_training.log` 中第一现场是：

```text
SGLangHttpServer pid=3697723
!!!!!!! Segfault encountered !!!!!!!
File "<unknown>", line 0, in cuGraphLaunch
File "<unknown>", line 0, in cudaGraphLaunch
File "<unknown>", line 0, in at::cuda::CUDAGraph::replay()
```

随后才出现：

```text
RuntimeError: gloo/transport/tcp/pair.cc:547 Connection closed by peer
```

这说明 Gloo 报错不是根因，而是 SGLang server 进程已经因为 CUDA graph replay segfault 死掉之后，其他进程继续通信时看到 peer closed connection。

`bw_throughput.jsonl` 最后有效状态大致是：

```json
{"running_sglang1": 13, "running_sglang2": 15, ...}
```

下一条开始 token usage / throughput 变成 0，running 字段缺失，说明监控端已经拿不到正常 server 状态。

## 2. 最关键的日志证据

`kunserve_sglang_detail.log` 最后阶段反复出现：

```text
replica0:
phase_e_decide local_bs=13 padded=16 fe=0 loop=overlap
neg_call bs=16 fe=0 state=balloon backend=sglang

replica1:
phase_e_decide local_bs=15 padded=16 fe=0 loop=overlap
neg_call bs=16 fe=0 state=balloon backend=sglang
```

这说明当前 Phase E 看到的真实情况是：

- replica 0 真实 decode batch 是 13。
- replica 1 真实 decode batch 是 15。
- 二者都被 fixed-padded CUDA graph 规则 padding 到 graph bucket 16。
- negotiate 传入的是 16，而不是 13/15。
- 因此全局协商结果会把这一步看成 `min_bs=max_bs=16`。

从系统角度看，这就是一个“真实 batch 状态不一致，但 graph bucket 一致”的场景。

当前代码把它当成 uniform graph replay 处理。

## 3. 当前 Phase E 的目的

KunServe 进入 balloon 之后，两个 SGLang replica 共享 expert。每个 replica 内部仍是 TP=2，每张卡有完整 attention 后 hidden state，然后需要跨 replica 的 lane 通信把 hidden/topk 信息送到拥有对应 expert shard 的 peer 上。

GLOBAL CUDA graph replay 下，每个 rank 必须在同一 decode step 进入相同结构的 graph。否则会出现：

- 某些 rank 进入跨 lane collective，另一些 rank 没进入；
- 某些 rank 用 graph bucket 16，另一些 rank 用 graph bucket 8；
- idle replica 没有真实 batch，但 peer replica 正在 decode；
- EXTEND/prefill 与 decode 混在同一个 GLOBAL graph step 中。

Phase E 的设计目标就是在每个 graph step 前，让所有参与 GLOBAL MoE 的 rank 对当前 step 的形状达成一致。

当前主要处理三类问题：

1. busy / idle：
   - 有的 replica 没有请求，但 peer 还有请求。
   - idle replica 要构造 keepalive dummy batch，避免 peer 在跨 replica collective 中没有 partner。

2. graph bucket mismatch：
   - 一个 rank 需要 graph bucket 8，另一个 rank 需要 graph bucket 16。
   - 所有 rank 应该统一到更大的 bucket，或者统一 eager。

3. EXTEND / mixed batch：
   - 有 rank 要做 prefill/extend。
   - GLOBAL decode graph 不适合 replay，所有 rank 应该统一 eager。

这些目标本身是正确的，但当前实现有一个关键缺口：它同步的是 padded graph bucket，不是真实 raw batch 状态。

## 4. 当前代码链路

### 4.1 scheduler 先拿本地 batch

在 `Scheduler.event_loop_overlap()` 或普通 event loop 中：

```python
batch = self.get_next_batch_to_run()
```

如果处于 KunServe balloon + sglang backend，会进入 Phase E：

```python
self._phase_e_get_step_decision(batch=batch, local_status=..., loop=...)
```

### 4.2 Phase E 计算本地 shape

在 `scheduler.py` 的 `_phase_e_get_step_decision()` 中：

```python
local_bs = batch.batch_size() if batch is not None else 0
local_force_eager = self._kunserve_batch_requires_eager_for_phase_e(batch)
local_padded = self._padded_capture_bs(local_bs)
```

`local_bs` 是真实 batch size。

`local_padded` 是根据当前 CUDA graph capture bucket 算出的 replay bucket。例如：

```text
local_bs=13 -> local_padded=16
local_bs=15 -> local_padded=16
local_bs=63 -> local_padded=64
local_bs=64 -> local_padded=64
```

### 4.3 当前传给 negotiate 的是 padded bs

当前代码调用：

```python
self.negotiate_balloon_step_state(
    local_padded,
    local_force_eager=local_force_eager,
    local_state_signature=local_signature,
)
```

这一步很关键：函数参数名叫 `local_bs`，但调用方实际传进去的是 `local_padded`。

因此 `model_runner.negotiate_balloon_step_state()` 收到的第一列不是 raw batch size，而是 graph bucket。

### 4.4 model_runner 做 all-gather

`model_runner.py` 中：

```python
local_payload = [int(local_bs), 1 if local_force_eager else 0]
if local_state_signature is not None:
    local_payload.extend(...)

all_gather(local_payload)

return (
    max(bs_values),
    min(bs_values),
    any(eager_values),
    state_fingerprint,
)
```

因为传入的是 `local_padded`，所以这次 13/15 的真实状态会变成：

```text
rank payload:
  r0tp0: [16, 0, signature(... raw 13 ...)]
  r0tp1: [16, 0, signature(... raw 13 ...)]
  r1tp0: [16, 0, signature(... raw 15 ...)]
  r1tp1: [16, 0, signature(... raw 15 ...)]

returned:
  negotiated_max_bs = 16
  negotiated_min_bs = 16
  any_force_eager = False
```

`state_fingerprint` 会包含 signature 差异，但当前它主要用于判断 cached decision 是否可以复用，不用于最终 graph/eager policy。

### 4.5 apply decision 判断是否 graph replay

`scheduler.py` 中 `_phase_e_apply_step_decision()`：

```python
force_eager = bool(negotiated_any_force_eager)
mismatch_decode = negotiated_min_bs > 0 and negotiated_min_bs != negotiated_max_bs

if graph_bs_override_hint is not None and not force_eager:
    graph_bs_override = graph_bs_override_hint
elif mismatch_decode and not force_eager:
    if capture_bs_supported(negotiated_max_bs):
        graph_bs_override = negotiated_max_bs
    else:
        force_eager = True
```

由于当前协商得到的是：

```text
min_bs = 16
max_bs = 16
any_force_eager = False
```

所以：

```text
mismatch_decode = False
force_eager = False
graph_bs_override = None
expected_graph_bs = 16
```

这一步把 raw 13/15 的真实不一致完全隐藏了。

### 4.6 cuda_graph_runner 最后选择 graph

`cuda_graph_runner.py` 中：

```python
cuda_graph_bs = forward_batch.batch_size
index = bisect_left(capture_bs, cuda_graph_bs)
padded_bs = capture_bs[index]
```

因此：

```text
raw_bs=13 -> selected graph bs=16
raw_bs=15 -> selected graph bs=16
```

graph guard 会检查 selected graph bs 是否等于 expected graph bs。这里二者都是 16，所以 guard 放行。

最终进入：

```python
self.graphs[graph_key].replay()
```

崩溃就发生在这个 replay 里。

## 5. 为什么这不是简单的 tail batch 问题

这次崩在 `13/15 -> 16/16`，看起来像 tail batch 才有的问题。但本质不是 batch 小，而是 raw batch state 被 padded bucket 掩盖。

同样的问题可以发生在大 batch：

```text
raw 63 / 64 -> padded 64 / 64
raw 127 / 128 -> padded 128 / 128
raw 95 / 96 -> padded 96 / 96
```

只要 graph bucket 一样，当前协议就会认为它们是 uniform decode step。

如果 GLOBAL graph replay 的所有输入、padding rows、dummy KV、global_num_tokens、DP buffer length、lane collective shape 都真的能严格支持 raw mismatch，那么这种做法可以成立。但目前代码没有把这个语义显式建模，只是在 graph runner 内部补了一部分 padding rows，因此它不够系统。

## 6. 当前 padding 补丁解决了什么，没有解决什么

`cuda_graph_runner.py` 中 `_kunserve_patch_global_graph_padding()` 会处理：

```text
raw_bs < graph_bs
```

它会把 padding rows 写到：

- phantom req_pool row
- dummy KV slot
- seq_lens=1
- out_cache_loc=dummy_kv_slot
- input_ids/positions/mrope_positions 清零

这解决的是一个具体问题：

> padding rows 不应该复用 stale req_pool_indices，否则 request 结束后这些 stale row 可能指向已经释放的 KV slot，导致 illegal address 或污染 KV。

但这个补丁没有解决更高层的协议问题：

1. Phase E 不知道 raw_bs mismatch。
2. graph/eager policy 不知道 raw_bs mismatch。
3. state_fingerprint 只用于 cache invalidation，不用于 graph replay 许可。
4. graph guard 只检查 graph bucket，不检查 raw state。
5. overlap loop 中 `last_batch/result_queue/current_batch` 的状态也没有成为 graph replay 的强约束。

因此 padding 补丁是必要条件，但不是充分条件。

## 7. 为什么 `neg_EXIT` 不等价于同步已经完成

当前 `KunServePyNcclGroup.all_gather_into_tensor()` 在非 CUDA graph capture 时走：

```python
with pynccl_comm.change_state(enable=True, stream=get_current_device_stream_fast()):
    pynccl_comm.all_gather(output, input)
```

`PyNcclCommunicator.all_gather()` 直接调用：

```python
ncclAllGather(..., cudaStream_t(stream.cuda_stream))
```

NCCL 调用通常是把 collective enqueue 到 CUDA stream。host 侧函数返回不代表 GPU 上这个 collective 已经执行完。

在 `negotiate_balloon_step_state()` 中，后面确实有：

```python
bs_values.max().item()
eager_values.max().item()
all_t.detach().cpu().tolist()
```

这些 `.item()` / `.cpu()` 一般会触发 stream 同步，因此函数 return 时应该拿到了结果。

但目前 WD 日志是：

```text
neg_ENTER
all_gather enqueue
neg_EXIT
CPU read results
return
```

`neg_EXIT` 打在 all_gather 调用之后、CPU read 之前。因此 `neg_EXIT` 只能说明 Python 已经提交了 NCCL 调用，不能说明 CPU 已经拿到最终协商结果。

这解释了为什么日志上“看起来都 neg_EXIT 了”，但仍不能把它当成严格的 step barrier。

更好的日志应该区分：

```text
neg_ENTER
neg_ENQUEUE_DONE
neg_RESULT_READY max=... min=... raw=...
neg_RETURN
```

这样才能知道是 enqueue 阶段卡住、device 阶段卡住，还是结果同步后才进入 graph replay。

## 8. 当前 cached negotiate 的风险

当前 Phase E 有 cache：

```text
KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL 默认 256
```

设计目标是减少每步 negotiate 的开销。cache guard 会把本地状态签名放进 collective：

```text
local_bs
local_padded
force_eager
batch mode
running len
waiting len
result_queue len
last_batch size
...
```

如果 fingerprint 变化，会 reset cache 并跑一个 eager step。

这个方向是合理的，但当前仍有两个问题：

1. cache guard 只影响“是否复用 cache”，不影响“当前 collective 之后是否允许 graph replay”。
2. 即使重新 collective，只要返回给 apply decision 的仍是 padded min/max，raw mismatch 还是会被隐藏。

所以 cached negotiate 不是根因，但它会让问题更难观察：部分 step 是 cache reuse，部分 step 是 collective refresh，日志上很容易误以为已经同步充分。

## 9. 为什么我不建议用简单阈值修

一种临时方案是：

```text
如果 local_bs < 16，强制 eager。
```

这能覆盖本次 `13/15` 的 crash，但它不是系统修复：

- `63/64 -> 64/64` 仍可能出问题。
- H20 大 batch 下也可能遇到 raw mismatch。
- 它没有改变协议不完整的问题。
- 它会把“是否安全”误绑定到 batch 大小，而不是绑定到 graph replay 语义是否完整。

阈值可以作为 emergency rollback 开关，但不应该作为主线设计。

## 10. 我认为真正的问题定义

当前 GLOBAL CUDA graph replay 的安全条件没有被完整表达。

现在代码实际表达的是：

```text
只要所有 rank 的 padded graph bucket 一样，并且没有 EXTEND/mixed，就可以 replay GLOBAL graph。
```

但更严格的安全条件应该至少包括：

```text
1. 所有 rank 对本 step 的 graph bucket 达成一致。
2. 所有 rank 对本 step 的 raw batch size 达成一致，或者系统显式支持 raw mismatch padding。
3. 所有 rank 对 idle/real/extend/decode mode 达成一致。
4. 所有 rank 对 keepalive/dummy row 的数量和语义达成一致。
5. 所有 rank 的 global_num_tokens / DP buffer len / lane collective tensor shape 一致。
6. graph input buffers 中 raw rows 和 padding rows 的 req_pool/out_cache_loc 都有效。
7. overlap loop 的 result_queue/last_batch/current_batch 不会让某个 rank 提前进入下一步 graph。
```

当前代码只强约束了第 1 条的一部分。

## 11. 推荐的系统修复方案

### 方案 A：协议先变完整，raw mismatch 先统一 eager

这是最稳的第一步。

扩展 negotiate payload，从：

```text
[padded_bs, force_eager, state_signature...]
```

改成：

```text
[
  raw_bs,
  padded_bs,
  mode_code,
  force_eager,
  is_idle,
  is_keepalive,
  state_signature...
]
```

返回值从：

```text
max_bs, min_bs, any_force_eager, state_fingerprint
```

改成：

```text
raw_max_bs,
raw_min_bs,
padded_max_bs,
padded_min_bs,
any_force_eager,
mode_mask,
state_fingerprint
```

然后 graph/eager policy 先保守定义：

```text
if any_force_eager:
    force eager
elif mode 不全是 decode/idle:
    force eager
elif padded_max_bs 不支持:
    force eager
elif raw_min_bs != raw_max_bs:
    force eager
else:
    replay graph bucket padded_max_bs
```

这不是阈值策略，而是语义策略：只要 raw batch 不一致，就先不 replay graph。

优点：

- 快速解决 native segfault。
- 不依赖 batch 大小。
- 错误边界清楚。
- 后续可以逐步放开 raw mismatch graph。

缺点：

- tail 阶段或 rebalancer 导致的不均衡阶段会多走 eager。
- 性能可能退一些，但只在 raw mismatch step 上退，不是全部退。

### 方案 B：显式支持 raw mismatch graph

这是更高性能但更复杂的方案。

核心思想是：如果要让 `13/15 -> graph 16` 成为合法 graph replay，那就不能只在 `cuda_graph_runner` 内部偷偷补 padding，而要在 scheduler / ForwardBatch / Phase E 级别把本 step 的 effective shape 变成一等概念。

需要做：

1. Phase E negotiate 得到：

```text
raw_bs_per_rank
padded_bs_per_rank
global_effective_bs = padded_max
```

2. 每个 rank 在进入 ForwardBatch 前就知道：

```text
real_rows = raw_bs
dummy_rows = global_effective_bs - raw_bs
effective_bs = global_effective_bs
```

3. ForwardBatch 明确携带：

```text
kunserve_real_bs
kunserve_effective_bs
kunserve_dummy_row_start
kunserve_dummy_row_count
kunserve_graph_bucket
```

4. `prepare_mlp_sync_batch()` / `global_num_tokens` / DP padding / lane dispatcher 使用 `effective_bs`，但采样和 request 状态更新只使用 `real_bs`。

5. graph runner 不再自己猜 padding，而是验证 ForwardBatch 已经 materialize 到 effective shape。

6. graph guard 检查：

```text
raw_bs == negotiated_raw_bs_for_this_rank
effective_bs == negotiated_padded_max
dummy rows point to phantom req/dummy KV
```

优点：

- 可以保留大部分 graph replay 性能。
- raw mismatch 不必统一 eager。
- 语义清楚，后续更接近 native TP=4 的固定 bucket 行为。

缺点：

- 改动范围大。
- 需要仔细验证采样、req_pool、KV 写入、result processing 只处理真实 row。
- 需要和 overlap loop 的 batch copy/result_queue 兼容。

### 方案 C：短期 fail-fast guard

无论选 A 还是 B，都应该先加 fail-fast guard。

当前最糟糕的是错误以 native segfault 形式出现。我们应该在 `CUDAGraph.replay()` 前检查：

```text
current raw_bs
selected graph bs
negotiated raw_min/raw_max
negotiated padded_min/padded_max
current mode
state fingerprint
```

如果发现不符合当前 policy，直接抛 Python `RuntimeError`，不要进入 `cuGraphLaunch`。

这样下一次错误会变成：

```text
RuntimeError: KunServe GLOBAL graph unsafe replay:
  raw_bs=13
  raw_min=13 raw_max=15
  padded_min=16 padded_max=16
  policy=raw_mismatch_requires_eager
```

这比 `CUDAGraph::replay()` segfault 可调试得多。

## 12. 推荐落地顺序

我建议按下面顺序做，而不是直接大改：

1. 扩展日志和 guard：
   - negotiate payload 中加入 raw/padded 分离。
   - 记录 `neg_RESULT_READY`。
   - 在 graph replay 前 fail-fast。

2. 先实现方案 A：
   - raw mismatch 统一 eager。
   - 不依赖 batch 阈值。
   - 跑 A800/H20 验证稳定性。

3. 用日志统计 raw mismatch 占比：
   - 如果 raw mismatch 很少，方案 A 可能已经足够。
   - 如果 raw mismatch 很频繁，再实现方案 B。

4. 实现方案 B：
   - 将 effective_bs/dummy rows 提前到 scheduler/ForwardBatch 层。
   - graph runner 只验证，不临时修补。

## 13. 本次日志对应的具体错误链路

本次最后阶段可以还原成：

```text
replica0 real batch = 13
replica1 real batch = 15

_padded_capture_bs:
  13 -> 16
  15 -> 16

negotiate receives:
  16, 16, 16, 16

negotiate returns:
  min=16
  max=16
  any_force_eager=False

_phase_e_apply_step_decision:
  mismatch_decode=False
  force_eager=False
  expected_graph_bs=16

cuda_graph_runner:
  raw_bs 13 selects graph 16
  raw_bs 15 selects graph 16
  graph guard passes because selected graph bs == expected graph bs

CUDAGraph.replay:
  native segfault in cuGraphLaunch
```

换句话说，这次 crash 不是因为没有 negotiate，而是 negotiate 的变量不够表达真实安全条件。

## 14. 一句话总结

当前 KunServe GLOBAL CUDA graph 的 Phase E negotiate 把“graph bucket 一致”误当成了“本 step 语义一致”。在 fixed-padded graph 下，raw batch 不同但 padded bucket 相同是常见状态；如果没有把 raw/effective/dummy row 语义显式同步并验证，就可能在 graph replay 中触发 native segfault。系统修复应该扩展 negotiate 协议和 graph guard，而不是用某个 batch 阈值临时 eager。


## 15. 对当前 Phase E negotiate 优化的代码核对

这里逐条核对当前代码中已经存在的优化，以及它们和原始设计描述的差异。

### 15.1 低频 refresh + cache 复用：代码存在，但默认并非“255 步无 collective”

当前代码确实有：

```python
self._phase_e_negotiate_interval = get_int_env_var(
    "KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL", 256
)
```

相关位置：

- `scheduler.py:init_running_status`
- `_phase_e_cache_enabled`
- `_phase_e_update_cached_decision`
- `_phase_e_try_reuse_cached_decision`

实现逻辑：

1. refresh/collective step：
   - 调用 `negotiate_balloon_step_state(...)`。
   - 得到 `negotiated_max_bs / negotiated_min_bs / any_force_eager / state_fingerprint`。
   - 如果可 cache，则写入：
     - `_phase_e_cached_max_bs`
     - `_phase_e_cached_min_bs`
     - `_phase_e_cached_any_force_eager`
     - `_phase_e_cached_steps_left = interval - 1`
     - `_phase_e_cached_state_fingerprint`

2. cache step：
   - 调用 `_phase_e_try_reuse_cached_decision(...)`。
   - 如果 cache 可复用，返回 cached decision。
   - 每复用一次，`_phase_e_cached_steps_left -= 1`。

但是当前默认还有：

```python
self._phase_e_cache_guard_enabled = os.environ.get(
    "KUNSERVE_PHASE_E_CACHE_GUARD", "1"
) not in ("0", "false", "False", "no", "NO")
```

只要 cache guard 开启，`_phase_e_try_reuse_cached_decision()` 内部仍会调用：

```python
self.negotiate_balloon_step_state(
    int(local_padded),
    local_force_eager=bool(local_force_eager),
    local_state_signature=local_signature,
)
```

也就是说，默认配置下，cache step 并不是“完全不做 runtime_group collective”。它仍然做一个小 payload 的 collective，用来比较 fingerprint 是否变化。

所以更准确的当前状态是：

| 描述 | 当前代码是否符合 |
|---|---|
| 有 `KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=256` | 是 |
| 有 cached decision | 是 |
| cache step 默认完全不 collective | 否 |
| cache guard 关闭时可做到窗口内不 collective | 是，但风险更大 |
| 当前日志中每 step 都有 `neg_call` | 是，说明 guard 实际在生效 |

这点很重要：本次 crash 不能简单归因于“256 步没有 negotiate，cache 过旧”。因为实际日志显示最后阶段每步都有 `neg_call / neg_ENTER / neg_EXIT`。

### 15.2 cache guard：代码存在，但它只保护 cache 复用，不保护 graph replay 语义

当前 `_phase_e_local_state_signature()` 会把一些本地状态放进 signature：

```text
local_bs
local_padded
local_force_eager
batch mode
running req 数
waiting queue 长度
batch_is_full
chunked_req 是否存在
result_queue 长度
last_batch_size
```

这比只同步 `(local_padded_bs, local_force_eager)` 更强。

cache guard 的意图是：

- 如果窗口内本地状态变了，所有 rank 通过 guard collective 看到 fingerprint 变化；
- 然后 reset cache；
- 当前 step 返回 `any_force_eager=True`，让所有 rank 跑一个 eager step；
- 下一步重新 collective 建立新 cache。

这解决的是“cache 是否还能复用”的问题。

但它没有解决“collective 之后当前 graph replay 是否安全”的问题。原因是：

- fingerprint 差异只用于 cache invalidation；
- `_phase_e_apply_step_decision()` 仍然只看 `negotiated_min_bs / negotiated_max_bs / any_force_eager`；
- 而这两个 bs 仍然是 padded bucket，不是 raw batch。

因此即使 guard 能观察到 raw batch 从 16 变成 13，它也只能让 cache reset 或某一步 eager；一旦重新建立 cache，系统仍可能把稳定的 `raw 13/15 -> padded 16/16` 当成可 replay 的 graph state。

### 15.3 graph bucket override：代码存在，但只处理 padded bucket mismatch

当前 `_phase_e_apply_step_decision()` 有：

```python
mismatch_decode = (
    negotiated_min_bs > 0 and negotiated_min_bs != negotiated_max_bs
)

elif mismatch_decode and not force_eager:
    if self._capture_bs_supported(negotiated_max_bs):
        graph_bs_override = negotiated_max_bs
    else:
        force_eager = True
```

对应 `model_runner` / `cuda_graph_runner`：

- `set_balloon_step_graph_bs_override(...)`
- `get_balloon_step_graph_bs_override()`
- `cuda_graph_runner._kunserve_graph_bs_override(...)`

这意味着：

```text
如果一个 rank padded 到 8，另一个 rank padded 到 16：
  所有 rank replay 16 graph
  小 rank pad 到 16
```

这个优化已经实现。

但它只处理 “padded bucket 不同”：

```text
raw 7 / 13 -> padded 8 / 16
```

它不处理 “raw 不同但 padded bucket 相同”：

```text
raw 13 / 15 -> padded 16 / 16
raw 63 / 64 -> padded 64 / 64
```

后一类正是这次 crash 的场景。

### 15.4 EXTEND / mixed prefill force eager：代码存在

当前 `_kunserve_batch_requires_eager_for_phase_e()`：

```python
if batch.forward_mode.is_extend(...):
    return True
return bool(getattr(batch, "is_extend_in_batch", False))
```

这个 bit 会进入 negotiate payload，并通过 `any_force_eager` 让所有 rank 同步 eager。

这部分设计是合理的。它解决 decode graph 遇到 prefill/extend shape 的问题。

本次 crash 最后阶段是 decode，`fe=0`，所以它不是本次直接原因。

### 15.5 prefill defer：代码存在

在 `_get_new_batch_prefill_raw()` 开头有：

```python
if self._phase_e_may_defer_prefill_for_cache():
    return None
```

触发条件：

- Phase E cache 开启；
- cache valid；
- cached steps left > 0；
- 本地有 waiting queue 或 chunked req。

这个优化的作用是避免某个 rank 在 cache window 中独自 admitted prefill，导致它要 negotiate/eager，而 peer 还在 replay cached graph。

本次 crash 最后阶段 `bw_throughput` 和 rebalancer events 都显示 waiting queue 为 0，且是 decode tail，所以 prefill defer 不是直接原因。

### 15.6 idle keepalive：代码存在，并且 overlap loop 里有 early keepalive

当前 keepalive 相关代码：

- `_build_balloon_keepalive_batch(...)`
- `_maybe_get_balloon_keepalive_batch(...)`
- `ScheduleBatch.prepare_for_idle(...)`

overlap loop 中还做了 early keepalive：

```python
if batch is None and phase_e_negotiated_max > 0:
    batch = self._build_balloon_keepalive_batch(...)
```

这段注释里明确说明它解决过一个历史问题：

```text
"last real batch still in result_queue" -> "queue empty" 的一拍空窗，
peer 正在 all_gather，而本 rank 没有 participant，导致 deadlock。
```

keepalive 使用：

- dummy input_ids
- seq_lens=1
- dummy_kv_slot
- phantom_req_idx
- ForwardMode.IDLE

本次 crash 时两个 replica 都还有真实 running 请求，r0=13、r1=15，不是 idle/busy 场景，所以 keepalive 不是直接原因。

### 15.7 release 收尾 drain：代码存在

`_kunserve_prepare_for_memory_release()` 中会：

- drain overlap `result_queue`
- stop keepalive
- reset Phase E cache
- clear force eager / graph bs override

这个优化解决的是 rollout 结束或释放显存时，queue 里还有 IDLE keepalive result 导致 no-request assertion 失败的问题。

本次 crash 发生时两个 replica 还在 running，不是 release 阶段，所以它不是直接原因。

### 15.8 padding row 写 dummy KV：代码存在，是必要但不充分

`cuda_graph_runner._kunserve_patch_global_graph_padding()` 会把 `raw_bs:graph_bs` 的 padding rows 写成：

- phantom req_pool row
- dummy KV slot
- seq_lens=1
- zero input/position

这个补丁很重要，它避免 padding rows 指向已经释放的 req/KV。

但是它只是 graph runner 内部的局部修补，不能替代 Phase E 协议层对 raw/effective/dummy 语义的同步。

### 15.9 graph guard：代码存在，但 guard 条件太弱

当前 graph guard 记录：

```text
expected_bs
max_bs
min_bs
state_fingerprint
```

replay 前检查：

```python
actual graph_bs == expected graph_bs
```

它能防止：

```text
Phase E 期望 graph 16，但 runner 实际选择 graph 8
```

但它不能防止：

```text
raw 13 / 15 都选择 graph 16
```

因为 actual graph bs 和 expected graph bs 都是 16。

所以本次 crash 能通过 graph guard。

## 16. 这些优化和本次 crash 的因果关系

结论：本次 crash 不是某一个优化单独导致的，而是当前 Phase E 优化路线中一个核心假设不成立。

这个核心假设是：

```text
只要所有 rank 使用同一个 graph bucket，就可以安全 replay GLOBAL CUDA graph。
```

目前代码将同步单位从“真实 raw batch shape”放宽成了“padded graph bucket shape”。这个放宽是为了性能：

- 避免 raw batch 一变化就全局重协商；
- 允许小 batch pad 到大 bucket；
- 允许 busy/busy bucket mismatch 走 graph override；
- 减少 eager fallback。

这些优化的方向不是错的，但它们需要一个前提：

```text
raw_bs != graph_bs 时，dummy rows / effective_bs / global_num_tokens /
lane collective / result processing 的语义必须被完整建模并被所有 rank 同步。
```

当前代码没有完整做到这一点。

### 16.1 不是“每 256 步才 negotiate”直接导致

本次日志最后阶段每步都有：

```text
neg_call
neg_ENTER
neg_EXIT
```

所以不能说是“cache window 太长，peer shrink 没被发现”直接导致本次 crash。

更准确地说：

```text
即使做了 negotiate，negotiate 的变量也不够。
```

因为它同步的是 padded bucket，而不是 raw/effective shape。

### 16.2 是“bucket-only negotiate”直接相关

本次具体链路是：

```text
raw r0 = 13
raw r1 = 15

padded r0 = 16
padded r1 = 16

negotiate 看到的是:
  min=16, max=16

apply decision 认为:
  uniform decode
  graph replay safe
```

如果 negotiate payload 区分 raw/padded，它至少能看到：

```text
raw_min=13
raw_max=15
padded_min=16
padded_max=16
```

然后可以由 policy 决定：

- 保守：raw mismatch 统一 eager；
- 激进：显式 materialize effective_bs=16，并验证 dummy rows。

现在这一步缺失，所以 crash 和 bucket-only negotiate 直接相关。

### 16.3 graph bucket override 是同一类假设，但本次不走 override 分支

之前为了避免 mismatched decode 走 eager，做了 graph bucket override：

```text
padded 8 / padded 16 -> all replay 16
```

本次是：

```text
padded 16 / padded 16
```

所以没有走 `mismatch_decode=True` 的 override 分支。

但二者共享同一个假设：

```text
pad 到同一 graph bucket 后，GLOBAL graph replay 就安全。
```

因此 graph bucket override 不是本次直接分支，但属于同一风险模型。

### 16.4 keepalive / prefill defer / release drain 不是本次直接原因

本次最后状态：

- 两个 replica 都 busy；
- running 分别还有 13 和 15；
- waiting 为 0；
- 还没进入 release；
- `fe=0`，不是 EXTEND/mixed。

所以这些优化不是直接触发点：

- idle keepalive；
- prefill defer；
- release drain；
- EXTEND force eager。

### 16.5 cache guard 有帮助，但不能解决 root cause

cache guard 把 raw state 放进 fingerprint，这说明代码已经意识到“只看 bucket 不够”。但目前 fingerprint 只用于判断 cache 是否复用，而不是用于 graph replay policy。

所以它最多能发现：

```text
state 发生变化，需要 reset cache
```

却不能表达：

```text
当前 state 虽然稳定，但 raw 13/15 与 graph 16/16 的语义是否安全。
```

这就是为什么 guard 不能防住本次 crash。

## 17. 对原 7.x 描述的修正版

原描述中大部分方向是对的，但需要修正以下几点。

### 17.1 当前默认不是“低频 collective”

更准确：

```text
当前默认是低频 full refresh + 每步 guard collective。
```

如果设置：

```text
KUNSERVE_PHASE_E_CACHE_GUARD=0
```

才会变成真正的窗口内无 collective。

但关闭 guard 后，peer shrink/growth/prefill 状态只能等 refresh step 才观察到，风险更高。

### 17.2 当前协商字段不是 `(local_padded_bs, local_force_eager)` 就够

当前设计实际还需要：

```text
raw_bs
padded_bs
mode
force_eager
is_idle / is_keepalive
state fingerprint
```

否则 graph/eager policy 无法区分：

```text
raw 16 / 16 -> padded 16 / 16
raw 13 / 15 -> padded 16 / 16
raw 0 / 16  -> keepalive / real
```

这三种在 graph bucket 上可能都像 16，但语义完全不同。

### 17.3 “mismatched decode 用 graph bucket override”需要重新定义

应该区分两类 mismatch：

1. padded bucket mismatch：

```text
raw 7 / 13 -> padded 8 / 16
```

2. raw mismatch but same bucket：

```text
raw 13 / 15 -> padded 16 / 16
```

当前代码只显式处理第 1 类。第 2 类被误判成 uniform。

### 17.4 下一版 Phase E 的正确抽象

建议把 Phase E 从“协商 graph bucket”升级为“协商本 step 的 global execution contract”：

```text
contract = {
  raw_bs_per_rank,
  padded_bs_per_rank,
  effective_bs,
  mode_per_rank,
  force_eager,
  graph_bucket,
  dummy_row_contract,
  state_fingerprint,
}
```

只有当 contract 满足当前 policy，才允许 GLOBAL CUDA graph replay。

如果 policy 暂时不支持 raw mismatch，就 fail-fast 或统一 eager，而不是进入 `CUDAGraph.replay()`。


## 18. 方案 A 落地状态（2026-06-09）

本次代码已经按方案 A 实现 conservative 修复：

1. `ModelRunner.negotiate_balloon_step_state()` 的 payload 从旧的 `[padded_bs, force_eager, signature...]` 扩展为：

```text
[raw_bs, padded_bs, force_eager, signature...]
```

返回值也从 4 元组扩展为 6 元组：

```text
(padded_max_bs, padded_min_bs, any_force_eager, state_fingerprint, raw_max_bs, raw_min_bs)
```

2. Scheduler 的 Phase E cache 保存 raw/padded 两组字段，并在 cache guard collective 中检查 raw min/max。

3. 策略上只对 busy/busy raw mismatch 强制 eager：

```text
raw_min_bs > 0 and raw_min_bs != raw_max_bs  -> force_eager=True
```

这不会把 idle/busy keepalive 错判成 raw mismatch；`raw_min_bs == 0 and padded_max_bs > 0` 仍然走 keepalive 逻辑。

4. `KUNSERVE_PHASE_E_CACHE_GUARD=0` 时不再启用 Phase E cache。原因是没有 guard collective 时，cache 窗口内无法观察 peer raw batch 变化，继续复用 cached bucket 会重新暴露 stale graph replay 风险。

5. GLOBAL graph guard 增加 raw 字段。如果未来某条错误路径试图在 busy/busy raw mismatch 下 replay CUDA graph，会在 `validate_balloon_graph_replay_guard()` 中抛 Python `RuntimeError`，而不是继续进入 `cudaGraphLaunch` 后以 native segfault 结束。

该方案牺牲的是 raw batch 不一致时的 graph replay 性能，但它不是 batch 阈值 hack；它把 Phase E 协议恢复到能表达真实安全条件的状态。后续如果要重新支持 `13/15 -> graph 16`，需要实现方案 B 中的一等 global execution contract / dummy row 语义，而不是只靠 graph runner 内部补 padding。
