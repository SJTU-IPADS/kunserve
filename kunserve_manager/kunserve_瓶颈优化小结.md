# KunServe 瓶颈优化小结

更新时间：2026-06-10

本文只总结当前代码里已经落地或保留为开关的“小优化”。它们主要围绕 balloon 后 GLOBAL decode 的三个问题：调度协商开销、跨 replica MoE 通信开销、H20/VMM 启动开销。当前 H20 上的 hang 还没有被这些优化解决，最新方向是用 `KUNSERVE_GLOBAL_FORWARD_PROBE=1` 继续定位 GLOBAL forward 内部卡点。

## 1. 调度与 graph bucket

### 1.1 Phase E 低频 negotiate + cache 复用

代码位置：

- `/workspace/sglang/python/sglang/srt/managers/scheduler.py`
- `/workspace/sglang/python/sglang/srt/model_executor/model_runner.py`

核心开关：

```bash
KUNSERVE_PHASE_E_NEGOTIATE_INTERVAL=256
KUNSERVE_PHASE_E_CACHE_GUARD=1
```

当前策略：

- 每个 refresh step 做一次 4-rank lockstep negotiate。
- cache step 复用上一次协商出的 bucket，避免每个 token 都做 runtime collective。
- 默认 interval 是 `256`，目的是把之前每步协商带来的 scheduler gap 降下来。
- 如果本地 batch 增长超过 cached bucket、出现 prefill/extend、raw batch mismatch、或 cache guard 发现状态变化，就 reset cache 或 force eager。

收益：

- 去掉了 steady decode 下大量重复 negotiate。
- 避免了“只有本 rank batch 变化时单边 negotiate”导致的死锁风险。

限制：

- 当前 raw busy mismatch 仍然强制 eager。比如 raw batch 是 `92/90`，即使 padded bucket 都是 `96`，也会 `force_eager=True`。
- 因此很多 GLOBAL 步并不会真正 replay global cuda graph，而是走 dynamic eager path。
- 这不是当前 H20 hang 的直接根因；最近日志显示 negotiate 已经返回，卡在后续 GLOBAL forward 内部。

### 1.2 idle keepalive batch

代码位置：

- `/workspace/sglang/python/sglang/srt/managers/scheduler.py`
- `/workspace/sglang/python/sglang/srt/managers/schedule_batch.py`

作用：

- 当一边 replica idle、另一边 replica busy 时，idle 侧构造 `ForwardMode.IDLE` keepalive batch。
- keepalive 的目标 batch size 来自 negotiated max bucket，确保所有 lane collective 按同样 shape 进入。

收益：

- 解决 last-request / idle-busy lockstep 问题。
- 让 GLOBAL graph 或 GLOBAL eager collective 不会因为一边没真实请求而少进一次 collective。

限制：

- keepalive 仍然会产生 padding 工作。
- 如果 peer shrink 很快，cache 窗口内可能多跑一段较大的 bucket。

## 2. GLOBAL MoE dispatch/combine 通信优化

### 2.1 PyNccl / registered collective

代码位置：

- `/workspace/sglang/python/sglang/srt/distributed/kunserve_pynccl.py`
- `/workspace/sglang/python/sglang/srt/model_executor/model_runner.py`

作用：

- 为 KunServe global/lane group 提供 PyNccl group coordinator。
- 支持 graph-safe registered collectives：`all_gather_into_tensor`、`reduce_scatter_tensor`、`all_reduce`。
- 在 graph capture 前做 collective preheat，降低第一次 capture/replay 时触发 NCCL lazy init 的风险。

收益：

- 让 GLOBAL cuda graph 有可能对齐 SGLang native TP 的通信方式。
- 避免 capture 路径意外落到 raw `torch.distributed`。

限制：

- eager runtime 下仍然可能走普通 PyNccl submit，不一定是 registered graph path。
- 当前 H20 hang 需要进一步确认卡在 PyNccl collective、TP all-reduce、还是其他 CUDA kernel。

### 2.2 dispatch 三个 all-gather 用 NCCL group 合并

代码位置：

- `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`
- `/workspace/sglang/python/sglang/srt/distributed/kunserve_pynccl.py`

默认开关：

```bash
KUNSERVE_DISPATCH_NCCL_GROUP=1
```

作用：

- GLOBAL static dispatch 原来有三次 all-gather：`hidden`、`topk_ids`、`topk_weights`。
- 当前默认用 `grouped_all_gather_into_tensor` 包进 `ncclGroupStart/End`。

收益：

- 每层 dispatch 少几个 NCCL launch 开销。
- 对 48 层模型这种每层都要 dispatch 的路径有累积收益。

限制：

- 只减少 launch overhead，不减少实际传输量。
- dynamic eager path 仍然按独立 all-gather 走。

### 2.3 static path 只传 graph bucket，不传 max capture capacity

代码位置：

- `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`

作用：

- static buffer 仍然按 `capture_max_m` 预分配，保证指针稳定。
- 但真正传给 NCCL 的 view 是当前 graph bucket `M`，不是整个 `capture_max_m`。

收益：

- 避免小 batch replay 时发送最大 capture 容量里的 padded/zero rows。
- 这是“先砍掉 max_capture_m -> graph_bucket_m 确定性浪费”的主要实现。

限制：

- 仍然会传 graph bucket 内的 padding。
- 如果为了减少 negotiate 固定到较大 bucket，通信浪费仍然存在。

### 2.4 fused remap topk ids

代码位置：

- `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`
- `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_remap.py`

默认开关：

```bash
KUNSERVE_REMAP_FUSED=1
```

作用：

- 把 global expert id remap 到本 rank local compact id。
- 用 fused kernel 替代原来的多步 torch op：比较、clamp、cast、gather、where、copy。

收益：

- 减少 dispatch remap 的 kernel launch 和中间 tensor 操作。
- 主要优化小 batch 下的固定开销。

限制：

- remap 不是当前最大瓶颈；收益小于跨 lane collective。

### 2.5 full-bucket 时跳过 pad zero/fill

代码位置：

- `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`

默认开关：

```bash
KUNSERVE_PAD_SKIP_WHEN_FULL=1
```

作用：

- 当 `local_m == graph_bucket_m` 时，`padded_hidden[:local_m].copy_()` 会覆盖整个 active view。
- 此时跳过前面的 `zero_()` / `fill_(-1)`。

收益：

- 少掉每层 dispatch 的一次或多次 padding 初始化 kernel。
- 对 full bucket replay 更有价值。

限制：

- partial bucket 不能跳过，因为 padding 区域必须被定义为 hidden=0、topk_ids=-1。

### 2.6 combine 默认 reduce-scatter

代码位置：

- `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`

默认开关：

```bash
KUNSERVE_STATIC_COMBINE_MODE=reduce_scatter
```

作用：

- Phase F lane combine 不再用 lane all-reduce 返回完整 union。
- 改成 lane reduce-scatter：lane 内对 partial 求和后，每个 rank 只取回自己 replica 的 chunk。
- 后续本 replica 内 TP all-reduce 仍由 MoE 层原生逻辑完成。

收益：

- 避免传回对本 replica 没用的 replica slot。
- 比 lane all-reduce 更接近“只返回本 rank 需要的数据”。

限制：

- 仍然是每层一个跨 lane collective。
- 还没有真正把 lane combine 和本地 TP MLP all-reduce 合成一个完全等价的 native TP=4 形态。

### 2.7 combine + TP all-reduce fusion 实验开关

代码位置：

- `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`
- `/workspace/sglang/python/sglang/srt/distributed/parallel_state.py`
- `/workspace/sglang/python/sglang/srt/models/qwen3_moe.py`

相关开关：

```bash
KUNSERVE_STATIC_COMBINE_TP_ALLREDUCE_FUSION=1
KUNSERVE_STATIC_COMBINE_MODE=reduce_scatter_tp_all_reduce
```

作用：

- 尝试把 lane reduce-scatter 和本地 TP all-reduce 串成一个 composite collective。
- 若成功，MoE 层会通过 `_kunserve_tp_allreduce_done` 避免再做一次普通 TP all-reduce。

状态：

- 这是实验路径，不是默认路径。
- 目前主要作为后续“最接近 native TP=4 通信形态”的方向保留。

风险：

- collective 顺序、shape、group name、capture/replay 语义都必须完全一致。
- 如果任一 rank 的 dispatch/combine 顺序不一致，会比普通路径更难 debug。

### 2.8 combine alt stream / chunked overlap 预留

代码位置：

- `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`
- `/workspace/sglang/python/sglang/srt/distributed/kunserve_pynccl.py`

相关开关：

```bash
KUNSERVE_COMBINE_ALT_STREAM=0
```

作用：

- 预留把 combine NCCL 放到 alt stream 的能力。
- preheat 里也预热了 alt-stream reduce-scatter/all-reduce/ncclReduce 模式。

状态：

- 默认关闭。
- 目前不是主要优化路径；后续如果做 chunked expert + chunked combine overlap 才有意义。

## 3. H20/VMM 启动路径优化

### 3.1 VMM map chunk 放大

代码位置：

- `/workspace/sglang/python/sglang/srt/utils/cuda_vmm.py`
- `/workspace/verl/data/compare_kunserve_vs_baseline.sh`
- `/workspace/verl/data/compare_kunserve_vs_baseline_80B.sh`

默认设置：

```bash
SGLANG_EXPERIMENTAL_VMM_MAP_CHUNK_MB=64
```

作用：

- H20/CUDA13 上 `cuMemCreate/cuMemMap/cuMemSetAccess` 的 per-call 固定开销很大。
- 原来按很小粒度逐块 map KV pool，启动阶段可能极慢。
- 当前对非 MoE weight 的 VMM mapping 使用更大的 chunk，减少 map 调用次数。

收益：

- 明显改善 H20 上 KunServe VMM KV pool 映射启动慢的问题。

限制：

- MoE weight 目前排除在这个 chunk 放大之外，避免改变 expert weight donor/映射语义。
- 这是启动优化，不解决 balloon 后 decode 性能瓶颈。

### 3.2 NCCL/Gloo 网卡默认绑定

代码位置：

- `/workspace/verl/data/compare_kunserve_vs_baseline.sh`
- `/workspace/verl/data/compare_kunserve_vs_baseline_80B.sh`

默认逻辑：

```bash
NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5; exit}')
GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
```

作用：

- 避免 NCCL/Gloo 枚举 docker/veth/无关网卡导致 bootstrap 慢或不稳定。

说明：

- 这主要影响 NCCL bootstrap / TCPStore / 跨进程控制面。
- 对单机 GPU 间 NVLink/P2P 数据面不应该造成负面影响；数据面仍由 NCCL topo 决定，例如日志中的 `via P2P/IPC`、`type NVL/PIX`。

## 4. 诊断辅助，不算性能优化

### 4.1 GLOBAL forward runtime probe

代码位置：

- `/workspace/sglang/python/sglang/srt/model_executor/model_runner.py`
- `/workspace/sglang/python/sglang/srt/models/qwen3_moe.py`
- `/workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- `/workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`

开关：

```bash
KUNSERVE_GLOBAL_FORWARD_PROBE=1
KUNSERVE_NVTX_STAGE_PROFILE=1
```

作用：

- 在 GLOBAL forward runtime path 记录 enter/exit：
  - model forward eager/graph replay
  - prepare sync
  - Qwen3 MoE router/topk/experts/MLP all-reduce
  - FusedMoE dispatch/core/combine/all-reduce
  - KunServe dispatch all-gather / remap
  - KunServe combine reduce-scatter / all-reduce
- 文本日志写入 `kunserve_sglang_detail.log`，NVTX range 进入 nsys。

用途：

- 当前 H20 hang 需要用这个定位。
- 判断规则：最后一个只有 `*_enter` 没有对应 `*_exit` 的事件，就是实际卡住的 op。

限制：

- 这是诊断日志，不是性能优化。
- 打开后会增加日志量，正常性能测试应关闭。

## 5. 当前仍未解决的问题

### 5.1 H20 balloon 后 hang

最新判断：

- 不是 Phase E negotiate 死锁。
- 日志显示 negotiate 已经返回，随后进入 GLOBAL forward，然后 SGLang decode 停住。
- 当前最可能卡在 GLOBAL eager forward 内部某个 collective 或 CUDA kernel。

下一步：

- 用 `KUNSERVE_GLOBAL_FORWARD_PROBE=1` 重新跑 H20。
- 根据最后一个 enter/exit 定位到底是：
  - dispatch lane all-gather；
  - expert kernel；
  - combine lane reduce-scatter；
  - 本地 TP all-reduce；
  - attention / model forward 其他位置。

### 5.2 raw batch mismatch 使 global graph replay 退回 eager

现状：

- 如果 raw batch 不一致，例如 `92/90`，当前策略会 force eager。
- 这保护 correctness，但会导致很多 balloon 后 GLOBAL decode 没有走 graph replay。

后续选择：

- 继续保守：先修 GLOBAL eager hang。
- 或重新设计 graph contract，让 raw mismatch 在 padded bucket 一致时也能安全 replay。

### 5.3 跨 replica MoE 通信仍是主要优化对象

目前已经做了 reduce-scatter、grouped all-gather、remap fused、pad skip，但结构上仍然是每层额外跨 lane dispatch/combine。

更大的优化方向仍是：

- token A2A dispatcher；
- 更窄的 peer exchange；
- lane combine 与本地 TP all-reduce 的语义融合；
- 尽量把 KunServe GLOBAL 通信形态逼近单实例 native TP=4。

## 6. 推荐使用方式

普通性能测试：

```bash
ONLY_RUN=kunserve \
KUNSERVE_PHASE_G=0 \
KUNSERVE_CAPTURE_POLICY=fixed_padded \
KUNSERVE_ROLLOUT_QUANTIZATION=none \
KUNSERVE_MOE_A2A_BACKEND=none \
KUNSERVE_MOE_RUNNER_BACKEND=triton \
bash /workspace/verl/data/compare_kunserve_vs_baseline.sh
```

H20 hang 定位：

```bash
OUT_ROOT=/workspace/verl/outputs/kunserve_probe_$(date +%Y%m%d_%H%M%S)

ONLY_RUN=kunserve \
OUT_ROOT="$OUT_ROOT" \
KUNSERVE_GLOBAL_FORWARD_PROBE=1 \
KUNSERVE_NVTX_STAGE_PROFILE=1 \
SGLANG_SERVER_NSYS_PROFILE=1 \
SGLANG_SERVER_NSYS_TRACE=cuda,nvtx,osrt \
bash /workspace/verl/data/compare_kunserve_vs_baseline.sh
```

查看卡点：

```bash
rg "model_forward|global_runtime" "$OUT_ROOT/kunserve/kunserve_sglang_detail.log" | tail -300
```
