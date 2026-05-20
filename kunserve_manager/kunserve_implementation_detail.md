# KunServe 当前实现说明（manager 模块化 + 两种跨实例 MoE 通信后端）

本文档对应当前工作区实现，重点解释：

1. `kunserve_manager` 为什么拆成多个模块，以及每个模块负责什么；
2. verl → manager → SGLang HTTP → scheduler/tp_worker/model_runner 的端到端控制链路；
3. BALLOON 后两种实例间 MoE 通信方法：`deepep` 与 `sglang`；
4. 当前脚本如何选择 bf16/no-quant 或 FP8/DeepEP 路径。

> 当前推荐的 correctness-first 初版：`KUNSERVE_COMM_BACKEND=sglang`，`KUNSERVE_CAPTURE_POLICY=disabled`，`moe_a2a_backend=none`，`moe_runner_backend=triton`，rollout 不传 `quantization`，因此 SGLang 按 rollout 默认 `dtype=bfloat16` 做 bf16/no-quant 推理。

---

## 1. manager 目录结构与职责边界

manager 是独立 sidecar，不在 SGLang 推理进程内部做 leader election，也不依赖 verl 内部 controller。verl 只负责：

- 启动多个 SGLang replica；
- 把 replica HTTP 地址传给 manager；
- 用 worker 发请求/收结果。

manager 负责 KunServe 控制面：状态轮询、layout 规划、跨 replica process group 初始化、GLOBAL runtime warmup/prepare/commit、状态日志输出。

当前拆分后的主要文件：

```text
sglang/kunserve_manager/kunserve_manager/
  cli.py              # 命令行入口：解析参数/env，创建并运行 KunServeController
  controller.py       # 状态机和 RPC 编排：start/tick/enter_balloon/restore/PG init
  client.py           # HTTP client + RPC 响应解析/成功判定 helper
  layout.py           # expert layout 规划：half split、global physical_to_logical map
  runtime_config.py   # 数据面 backend/capture policy 配置
  net.py              # get_free_port 等小型网络 helper
  __main__.py         # python -m kunserve_manager 入口
  __init__.py         # 对外导出 KunServeController
```

这种拆分后，`controller.py` 不再同时承载 HTTP、layout、配置校验、端口探测所有逻辑；它只保留“控制状态机 + 调用这些模块”的核心逻辑。

---

## 2. manager 侧各模块实现

### 2.1 `cli.py`

`cli.py` 做三件事：

1. 从 CLI/env 读取 replica、model path、poll interval、group/backend 等参数；
2. 新增两个数据面参数：
   - `--comm-backend deepep|sglang`
   - `--capture-policy auto|fixed_padded|disabled`
3. 调用：

```python
controller = KunServeController.from_server_addresses(...)
asyncio.run(_run(controller))
```

重要参数：

| 参数 | 含义 |
|---|---|
| `--replica HOST:PORT` | 每个 SGLang replica 的 HTTP 地址，当前要求正好 2 个 |
| `--backend nccl` | `init_weights_update_group` 的 torch distributed backend |
| `--comm-backend deepep|sglang` | BALLOON GLOBAL MoE 数据面后端 |
| `--capture-policy ...` | GLOBAL CUDA graph capture 策略；`sglang` 初版会强制不 capture |
| `--output-dir` | manager 写 `bw_throughput.jsonl` 等 artifact 的目录 |

### 2.2 `client.py`

`client.py` 包含：

- `KunServeReplicaClient` protocol：定义 controller 需要的 replica RPC 接口；
- `KunServeHttpReplicaClient`：真实 HTTP 实现；
- `_unwrap_status_response()` / `_unwrap_output_list()`：兼容 SGLang endpoint 返回 list 或 dict；
- `_response_succeeded()` / `_response_error()`：统一判断 RPC 是否成功；
- `_commit_response_succeeded()`：commit 的幂等成功判断。

`_commit_response_succeeded()` 很关键：有些情况下 HTTP 返回 `success=False`，但 status 已经进入目标 `balloon/global/offloaded=N` 状态；这时 controller 会把它当作幂等成功，避免重复 rollback。

### 2.3 `layout.py`

`layout.py` 负责从 `/kunserve/status` 生成 layout plan。

核心结构：

```python
@dataclass
class KunServeLayoutPlan:
    local_ep_size: int
    local_routed_experts: int
    retained_local_experts: int
    offload_local_experts: int
    global_world_size: int
    global_physical_to_logical_map: list[list[int]]
    replica_active_mappings: list[list[int]]
```

当前假设是 two-replica symmetric half split：

- replica0 保留每个 local EP rank 的前半 experts；
- replica1 保留每个 local EP rank 的后半 experts；
- 两个 replica 合起来仍覆盖原本完整 expert 数量。

例如每个 local EP rank 有 64 个 routed experts，保留/卸载各 32 个：

```text
replica0 active_local_expert_mapping = [0..31]
replica1 active_local_expert_mapping = [32..63]
```

`build_complementary_physical_to_logical_map()` 会把每层 local physical expert 布局重组成 GLOBAL physical dispatch domain：先放 replica0 retained prefix，再放 replica1 retained suffix。

### 2.4 `runtime_config.py`

`KunServeRuntimeBackendConfig` 校验并保存数据面策略：

```python
comm_backend: deepep | sglang
capture_policy: auto | fixed_padded | disabled
exchange_mode: global_dense_v0
```

目前：

- `deepep + auto`：沿用原先 DeepEP 逻辑，是否 capture 由 model_runner/DeepEP mode 决定；
- `sglang + auto/fixed_padded/disabled`：当前初版都不 capture GLOBAL graph，因为 dispatcher 使用 dynamic padded all-gather/all-reduce，不适合 CUDA graph replay。

### 2.5 `controller.py`

controller 是状态机核心。

启动阶段：

```text
start()
  ├─ _wait_for_replicas_ready()       # 等 /kunserve/status 可用
  ├─ _ensure_layout_plan()            # 调 layout.py 生成 plan
  ├─ _ensure_process_group()          # 初始化跨 replica global PG
  ├─ _warmup_balloon()                # 可选 eager warmup GLOBAL bundle
  └─ 启动后台 poll thread
```

轮询阶段：

```text
tick()
  ├─ _fetch_statuses()
  ├─ _write_bw_status_sample()        # 写 manager 侧 bw_throughput.jsonl
  ├─ _should_enter_balloon()
  ├─ enter_balloon() if needed
  └─ restore_balloon() if enable_restore and all drained
```

进入 BALLOON：

```text
enter_balloon()
  ├─ _ensure_process_group(plan)
  ├─ _build_layout_payloads(plan)
  ├─ POST /kunserve/prepare_balloon to both replicas
  ├─ POST /kunserve/commit_balloon  to both replicas
  └─ 状态变为 balloon
```

`_build_layout_payloads()` 是 manager 与 SGLang runtime 接口的核心，payload 包含：

```json
{
  "target_variant": "global",
  "runtime_ep_size": 4,
  "runtime_rank_offset": 0或2,
  "dispatch_rank_offset": 0或2,
  "active_local_expert_mapping": [0..31] 或 [32..63],
  "physical_to_logical_map": "GLOBAL complementary map",
  "process_group_name": "kunserve_global_ep_v...",
  "capture_cuda_graph": false,
  "kunserve_comm_backend": "sglang",
  "capture_policy": "disabled",
  "kunserve_pg_names": {"global": "..."},
  "kunserve_backend_config": {
    "exchange_mode": "global_dense_v0",
    "local_ep_size": 2,
    "num_replicas": 2,
    "global_world_size": 4
  }
}
```

---

## 3. 端到端控制链路

### 3.1 verl 启动 manager

`verl/verl/experimental/agent_loop/agent_loop.py` 中 `_maybe_spawn_kunserve_manager()` 在两个 SGLang replica 都启动后执行：

```text
python -m kunserve_manager \
  --model-path ... \
  --poll-interval ... \
  --group-name kunserve_global_ep \
  --backend nccl \
  --comm-backend sglang \
  --capture-policy disabled \
  --replica host0:port0 \
  --replica host1:port1
```

`verl/verl/workers/config/rollout.py` 新增：

```python
kunserve_comm_backend: str = "deepep"
kunserve_capture_policy: str = "auto"
```

这保持 verl 很轻：只传配置，不实现 KunServe 控制面。

### 3.2 SGLang HTTP endpoint

SGLang HTTP server 已提供：

```text
GET  /kunserve/status
POST /kunserve/warmup_balloon
POST /kunserve/prepare_balloon
POST /kunserve/commit_balloon
POST /kunserve/restore_from_balloon
POST /init_weights_update_group
POST /destroy_weights_update_group
```

manager 只通过这些 HTTP endpoint 与 replica 通信。

### 3.3 RPC 数据结构

`sglang/python/sglang/srt/managers/io_struct.py` 中：

- `PrepareBalloonReqInput`
- `WarmupBalloonReqInput`

新增字段：

```python
kunserve_comm_backend: str = "deepep"
capture_policy: str = "auto"
kunserve_pg_names: Optional[Dict[str, str]] = None
kunserve_backend_config: Optional[Dict[str, Any]] = None
```

这些字段经过：

```text
scheduler.py
  → tp_worker.py
    → model_runner.prepare_balloon()/warmup_balloon()
      → _warmup_balloon_global_runtime()
        → register_balloon_global_runtime_bundle()
```

---

## 4. SGLang ModelRunner GLOBAL bundle 注册

核心文件：

```text
sglang/python/sglang/srt/model_executor/model_runner.py
```

核心函数：

```python
register_balloon_global_runtime_bundle(...)
```

该函数负责给每个 `FusedMoE` 层注册一个 `global` runtime bundle。

### 4.1 共同逻辑

无论使用 `deepep` 还是 `sglang` backend，都会做：

1. 检查 `ep_dispatch_algorithm=static`；
2. 根据 manager 传入的 `physical_to_logical_map` 构建 GLOBAL expert metadata；
3. 解析当前 rank 在 GLOBAL EP world 中的：
   - `resolved_ep_size`
   - `resolved_moe_ep_rank`
   - `resolved_dispatch_ep_rank`
4. 为每层 MoE 生成：
   - `active_local_expert_mapping`
   - `dispatcher_local_expert_mapping`
   - `global_runner_config = replace(local_config, num_local_experts=retained)`
5. 调 `layer.register_runtime_bundle(variant="global", ...)`。

`reduce_results=False` 是共同设计：

- DeepEP 的 `combine()` 已经把远端 expert output 聚合回 token 原 rank；
- CrossReplicaStandardDispatcher 的 `combine()` 会 global all-reduce partial sum 后 slice 回本地 batch；
- 如果 FusedMoE 末尾再做 `tensor_model_parallel_all_reduce()`，会 double reduce。

---

## 5. 实例间 MoE 通信方法一：DeepEP backend

启用方式：

```bash
KUNSERVE_COMM_BACKEND=deepep
KUNSERVE_MOE_A2A_BACKEND=deepep
KUNSERVE_MOE_RUNNER_BACKEND=deep_gemm
KUNSERVE_ROLLOUT_QUANTIZATION=fp8
```

SGLang GLOBAL bundle 逻辑：

1. `kunserve_comm_backend == "deepep"` 时，保留原 hard guard：
   - `moe_a2a_backend` 不能是 `none`；
   - `moe_runner_backend` 需要是 `deep_gemm`。
2. 如果 `SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL=1`，显式构造：

```python
MaybeTboDeepEPDispatcher(..., deepep_mode=DeepEPMode.NORMAL)
```

3. 否则由 `create_moe_dispatcher()` 根据全局 `moe_a2a_backend=deepep` 自动创建 DeepEP dispatcher。
4. `FusedMoE.run_moe_core()` 使用 DeepGEMM/DeepEP 对应的 pre/post permutation 路径。

数据流：

```text
local hidden/topk
  → DeepEP dispatch：按 topk_ids 把 token hidden 发到 expert 所在 GLOBAL rank
  → local retained experts compute
  → DeepEP combine：把 expert output 发回 token 原 rank并加权组合
  → FusedMoE 返回 local batch output
```

优点：

- 真正 token-level dispatch/combine；
- 低延迟模式配合 GLOBAL CUDA graph 理论上性能最好。

风险/约束：

- LL/IBGDA/NVSHMEM 对环境敏感；
- H20 上这条链路通常要求 FP8 + DeepGEMM；
- NORMAL mode eager 性能很差。

---

## 6. 实例间 MoE 通信方法二：SGLang CrossReplicaStandardDispatcher backend

启用方式（当前默认 smoke/compare 路径）：

```bash
KUNSERVE_COMM_BACKEND=sglang
KUNSERVE_CAPTURE_POLICY=disabled
KUNSERVE_MOE_A2A_BACKEND=none
KUNSERVE_MOE_RUNNER_BACKEND=triton
KUNSERVE_ROLLOUT_QUANTIZATION=none   # 或 bf16/bfloat16，都表示不传 quantization
```

核心文件：

```text
sglang/python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py
```

核心类：

```python
CrossReplicaStandardDispatcher
```

### 6.1 为什么需要它

普通 `StandardDispatcher` 的假设是：同一个 SGLang 实例内部所有 EP ranks 一开始都有同一批 token hidden states，各自只算本地 expert，最后 TP/EP all-reduce partial sum。

跨 replica 时这个假设不成立：replica0 的 rank 没有 replica1 请求的 hidden states。因此 `CrossReplicaStandardDispatcher` 在 MoE 层内部先做一个 GLOBAL token union。

### 6.2 dispatch() 流程

对每个 MoE 层，每个 GLOBAL EP rank 执行：

```text
1. all_gather local token count
2. pad hidden/topk 到 max_m
3. all_gather hidden/topk_weights/topk_ids across global process group
4. lane select：
     lane = global_rank % local_ep_size
     rank0/rank2 选择 lane0 的所有 replica segment
     rank1/rank3 选择 lane1 的所有 replica segment
5. 根据 dispatcher_local_expert_mapping remap topk_ids：
     本 rank 持有的 physical expert → compact local expert id
     非本 rank 持有的 expert → -1
6. 返回 StandardDispatchOutput(union_hidden, remapped_topk)
```

这样每个 lane rank 都看到跨 replica 的 token union，但只计算自己 retained 的 expert rows。

### 6.3 local expert compute

GLOBAL bundle 显式使用：

```python
MoeRunner(MoeRunnerBackend.TRITON, global_runner_config)
```

因此不经过 DeepEP/NVSHMEM/DeepGEMM。

FP8 Triton path 也做了 runtime tensor 修正：`fp8.py` 的 Triton branch 现在用：

```python
layer.get_runtime_tensor("w13_weight")
layer.get_runtime_tensor("w2_weight")
...
```

而不是直接 `layer.w13_weight/layer.w2_weight`。这保证 GLOBAL bundle 的 active sliced experts 生效，尤其是 replica1 保留 suffix experts 时不会错误读取前半 expert rows。

### 6.4 combine() 流程

```text
1. runner 输出 union token 的本 rank partial expert sum
2. dist.all_reduce(SUM) across global process group
3. slice 回本 replica 原 local_m tokens
4. 返回 local batch output
```

这里的 global all-reduce 语义等价于“所有 GLOBAL EP ranks 对 union tokens 的 expert contribution 求和”。

### 6.5 当前限制

- correctness-first eager 实现；
- 每层都做 all-gather + all-reduce，性能不是最终目标；
- 不 capture GLOBAL CUDA graph；
- 当前只支持 `StandardTopKOutput`；
- 依赖同一 replica 内 EP ranks token count 一致，否则会显式报错。

---

## 7. FusedMoE runtime bundle 相关实现

核心文件：

```text
sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```

现有 runtime bundle 支持：

```python
FusedMoERuntimeBundle(
    variant,
    moe_runner_config,
    dispatcher,
    runner,
    moe_ep_size,
    moe_ep_rank,
    moe_tp_size,
    moe_tp_rank,
    num_local_experts,
    reduce_results,
    active_local_expert_mapping,
    active_tensors,
)
```

本次相关点：

1. GLOBAL bundle 可显式传 dispatcher/runner，不走默认 `create_moe_dispatcher()`；
2. `switch_runtime_bundle()` 会把 dispatcher/runner/reduce_results/active tensors 同步回 live layer；
3. `build_dense_expert_runtime_tensors()` 增加了：

```text
w13_input_scale
w2_input_scale
```

这样 FP8 static activation scale 也能随 active expert rows 一起切片。

---

## 8. 脚本现状：bf16/no-quant vs FP8

### 8.1 KunServe AB compare

`verl/data/compare_kunserve_vs_baseline.sh` 默认：

```bash
KUNSERVE_COMM_BACKEND=sglang
KUNSERVE_ROLLOUT_QUANTIZATION=none
GPU_MEMORY_UTILIZATION=0.65
```

因此 KunServe run 默认不传 `actor_rollout_ref.rollout.quantization`，走 bf16/no-quant；baseline `run_smoke_test.sh` 原本也不传 quantization，因此也是 bf16/no-quant。

### 8.2 KunServe smoke

`verl/data/train/run_smoke_test_kunserve_tp2_dual_replica.sh` 现在按 backend 自动设置默认值：

```bash
# sglang backend 默认
KUNSERVE_MOE_A2A_BACKEND=none
KUNSERVE_MOE_RUNNER_BACKEND=triton
KUNSERVE_ROLLOUT_QUANTIZATION=none

# deepep backend 默认
KUNSERVE_MOE_A2A_BACKEND=deepep
KUNSERVE_MOE_RUNNER_BACKEND=deep_gemm
KUNSERVE_ROLLOUT_QUANTIZATION=fp8
```

`none|null|bf16|bfloat16|false|0` 都表示“不传 rollout.quantization”。

### 8.3 TP=2 vs TP=4 compare

`verl/data/compare_tp2_tp4_single_sglang.sh` 默认调用 `run_smoke_test_single_sglang_tp.sh`，现在默认：

```bash
ROLLOUT_QUANTIZATION=none
MOE_A2A_BACKEND=none
MOE_RUNNER_BACKEND=triton
```

因此 TP/rank throughput 对比默认是 bf16/no-quant，不再混入 FP8/DeepEP 路径。

---

## 9. 建议的 grep 验证点

运行 KunServe smoke 后：

```bash
grep -aE "comm_backend=sglang|CrossReplicaStandardDispatcher|capture_graph=False|GLOBAL bundle uses" \
  /workspace/verl/outputs/<RUN>/kunserve/verl_training.log
```

期望看到：

```text
comm_backend=sglang
capture_policy=disabled
capture_graph=False
GLOBAL bundle uses CrossReplicaStandardDispatcher
CrossReplicaStandardDispatcher active
```

确认 no-quant/bf16：

```bash
grep -a "rollout.quantization" /workspace/verl/outputs/<RUN>/kunserve/verl_training.log
```

如果没有 `actor_rollout_ref.rollout.quantization=fp8`，且 SGLang args 里 `quantization=None`，就是 bf16/no-quant。

---

## 10. 后续优化方向

1. **sglang backend 性能优化**：当前每层 dense all-gather + all-reduce 是 correctness-first，后续可做 fixed padded graph、lane subgroup、reduce-scatter 等优化。
2. **idle keepalive**：BALLOON 后所有 ranks 必须共同进入 collective；长尾时如果一个 replica 完全 idle，仍需 dummy participation。
3. **DeepEP LL 路径**：若 nvidia-peermem/IBGDA/NVSHMEM 环境稳定，可继续作为高性能路径。
4. **更多 replica/general split**：当前 layout planner 写死 two-replica symmetric half split，可在 `layout.py` 扩展。

---

## 11. 本轮修改代码逐文件逐行注释版

前面章节解释的是设计和链路；这一节把**当前本轮新增/修改的核心代码**直接写进文档，并对每一行/每个连续语句给出注释。为了避免把已经删除的旧代码也重复贴出来，这里只覆盖“新增代码、变更后的关键代码块、脚本新增逻辑”。完整文件仍以工作区源码为准。

### 11.1 `kunserve_manager/runtime_config.py`：GLOBAL 通信后端配置

```python
from __future__ import annotations                 # 允许前向类型标注，保持 py3.10+ 兼容

from dataclasses import dataclass                  # 用 dataclass 表达只保存配置的轻量对象


@dataclass(frozen=True)                            # frozen=True：创建后不可变，避免运行中被误改
class KunServeRuntimeBackendConfig:                # manager 内部统一保存数据面 backend/capture 策略
    """Data-plane backend knobs sent from the sidecar to every sglang replica."""

    comm_backend: str = "deepep"                   # 默认保持旧 DeepEP 路径，不破坏历史行为
    capture_policy: str = "auto"                   # 默认保持旧自动 capture 策略
    exchange_mode: str = "global_dense_v0"         # 当前 sglang backend 的交换模式名，预留后续扩展

    def __post_init__(self) -> None:                # dataclass 初始化后做规范化/校验
        comm_backend = str(self.comm_backend).lower()       # backend 大小写归一
        capture_policy = str(self.capture_policy).lower()   # capture policy 大小写归一
        if comm_backend not in ("deepep", "sglang"):       # 只允许两条已实现路径
            raise ValueError(                                  # 启动阶段 fail fast
                f"Unsupported KunServe comm_backend={self.comm_backend!r}; "
                "expected 'deepep' or 'sglang'."
            )
        if capture_policy not in ("auto", "fixed_padded", "disabled"):  # 只允许规划内策略
            raise ValueError(
                f"Unsupported KunServe capture_policy={self.capture_policy!r}; "
                "expected 'auto', 'fixed_padded', or 'disabled'."
            )
        object.__setattr__(self, "comm_backend", comm_backend)       # frozen dataclass 内部写回规范值
        object.__setattr__(self, "capture_policy", capture_policy)   # 同上

    def should_capture_global_graph(self) -> bool:  # manager 构造 payload 时调用
        if self.capture_policy == "disabled":      # 显式禁用时直接 False
            return False
        if self.comm_backend == "sglang":           # sglang 初版是 dynamic all-gather/all-reduce
            # P0 sglang backend uses dynamic padded all-gather/all-reduce.
            # Capture support needs a later fixed-padded graph implementation;
            # until that lands, even capture_policy=fixed_padded is treated as
            # disabled rather than attempting an unsafe graph capture.
            return False                            # 因此不尝试 GLOBAL graph capture
        return True                                 # DeepEP 路径仍允许按旧逻辑 capture
```

### 11.2 `kunserve_manager/net.py`：端口 helper

```python
from __future__ import annotations                  # 前向标注兼容

import socket                                       # 用标准库 socket 探测端口


def get_free_port(host: str = "") -> tuple[int, socket.socket]:  # 返回当前可 bind 的端口
    """Return a currently-free TCP port on ``host``.

    The probe socket is closed before returning, matching the historical
    controller behavior.  The caller should still treat the result as best
    effort because another process may bind the port before the HTTP RPC reaches
    replica rank 0.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       # 创建 TCP socket
    sock.bind((host or "", 0))                                     # 端口 0 让内核分配空闲端口
    port = int(sock.getsockname()[1])                              # 读取实际端口号
    sock.close()                                                   # 保持历史行为：立即释放 probe socket
    return port, sock                                              # 返回端口；sock 已关闭，仅兼容旧签名
```

### 11.3 `kunserve_manager/layout.py`：expert half split 和 GLOBAL map

```python
@dataclass
class KunServeLayoutPlan:                         # manager 内部“一次 BALLOON 布局”的完整计划
    local_ep_size: int                            # 单个 SGLang replica 内部 EP size，例如 TP=EP=2 时为 2
    local_routed_experts: int                     # 每个 local EP rank 原本持有 routed experts 数，例如 64
    retained_local_experts: int                   # BALLOON 后本 replica 每个 rank 保留多少 experts，例如 32
    offload_local_experts: int                    # BALLOON commit 时卸载/借出的 experts 数，例如 32
    global_world_size: int                        # 两个 replica 合并后的 GLOBAL EP world size，例如 4
    global_physical_to_logical_map: list[list[int]] # 每层 GLOBAL physical id -> logical id 映射
    replica_active_mappings: list[list[int]]      # 每个 replica 保留的 local expert row，例如 [0..31]/[32..63]
```

`build_complementary_physical_to_logical_map()` 的关键代码：

```python
local_chunk = num_physical_experts // local_ep_size    # 每个 local EP rank 对应的 physical expert 连续块大小
if retained_local_experts * 2 != local_chunk:          # 当前只支持一半保留、一半卸载
    raise ValueError(...)

for layer_row in normalized:                           # 对每一层 MoE 独立构造 GLOBAL map
    global_row: list[int] = []                         # 这一层新的 GLOBAL physical->logical row
    for local_rank in range(local_ep_size):             # 第一段：每个 local rank 的 prefix half
        base = local_rank * local_chunk                 # 当前 local rank 在原 physical map 中的起点
        global_row.extend(layer_row[base : base + retained_local_experts])
    for local_rank in range(local_ep_size):             # 第二段：每个 local rank 的 suffix half
        base = local_rank * local_chunk + (local_chunk - retained_local_experts)
        global_row.extend(layer_row[base : base + retained_local_experts])
```

解释：

- 第一轮 loop 把 replica0 保留的 experts 放到 GLOBAL dispatch domain 前半部分；
- 第二轮 loop 把 replica1 保留的 experts 放到 GLOBAL dispatch domain 后半部分；
- 这样 GLOBAL physical expert 数量保持不变，但每个 physical row 实际来自不同 replica 的 retained rows。

`build_layout_plan_from_statuses()` 的关键代码：

```python
local_maps = [status.get("local_physical_to_logical_map") for status in statuses]  # 读取两边原始 expert map
if any(local_map is None for local_map in local_maps):                             # status 不完整直接失败
    raise ValueError(...)
if local_maps[0] != local_maps[1]:                                                 # 两个 replica baseline 必须一致
    raise ValueError(...)

local_ep_size = int(statuses[0]["local_ep_size"])                                 # 读取 replica 内 EP size
routed_by_layer = {                                                                # 读取每层 local routed experts 数
    int(layer_id): int(count)
    for layer_id, count in statuses[0]["local_routed_experts_per_layer"].items()
}
routed_values = set(routed_by_layer.values())                                      # 当前要求所有 MoE 层同一个 expert 数
if len(routed_values) != 1:
    raise ValueError(...)
local_routed_experts = routed_values.pop()                                         # 例如 64

resolved_offload = (                                                               # 用户指定则用用户值，否则默认 half
    int(offload_local_experts)
    if offload_local_experts is not None
    else local_routed_experts // 2
)
retained_local_experts = local_routed_experts - resolved_offload                   # half split 下等于 offload
if retained_local_experts != resolved_offload:
    raise ValueError(...)

replica_active_mappings = [                                                        # replica0 保留 prefix，replica1 保留 suffix
    list(range(retained_local_experts)),
    list(range(local_routed_experts - retained_local_experts, local_routed_experts)),
]
```

### 11.4 `kunserve_manager/client.py`：HTTP RPC 客户端和响应判定

关键响应 helper：

```python
def _unwrap_status_response(raw: Any) -> dict[str, Any]:      # /kunserve/status 可能返回 list 或 dict
    if isinstance(raw, list):                                 # 某些 SGLang RPC wrapper 返回 list[output]
        if not raw:                                           # 空 list 是协议错误
            raise ValueError("Empty balloon status response.")
        return dict(raw[0])                                   # 取第一个 rank/status
    if isinstance(raw, dict):                                 # 已经是 dict 就直接复制
        return dict(raw)
    raise TypeError(...)                                      # 其他类型直接暴露协议错误


def _response_succeeded(raw: Any) -> bool:                    # 通用 RPC 成功判定
    outputs = _unwrap_output_list(raw)                        # 先统一成 list[dict]
    return bool(outputs) and all(bool(item.get("success")) for item in outputs)


def _commit_response_succeeded(raw: Any, *, target_variant: str, offload_local_experts: int) -> bool:
    if _response_succeeded(raw):                              # 正常 success=True 当然成功
        return True
    outputs = _unwrap_output_list(raw)                        # 否则做幂等成功判定
    return bool(outputs) and all(                             # 所有返回项 status 都已到目标状态，也算成功
        _status_matches_balloon_target(
            item.get("status"),
            target_variant=target_variant,
            offload_local_experts=offload_local_experts,
        )
        for item in outputs
    )
```

`KunServeHttpReplicaClient._request()` 的控制流：

```python
for attempt in range(self.max_attempts):                       # 每个 HTTP RPC 最多重试 max_attempts 次
    try:
        async with self._get_session(timeout=timeout) as session:  # 每次创建短生命周期 aiohttp session
            if method.upper() == "GET":                       # status 走 GET
                async with session.get(url) as response:
                    response.raise_for_status()                 # 非 2xx 直接抛 ClientResponseError
                    return await _read_async_response(response) # JSON 或文本 fallback
            async with session.post(url, json=payload or {}) as response: # 其他 control RPC 走 POST JSON
                response.raise_for_status()
                return await _read_async_response(response)
    except asyncio.TimeoutError:                               # 超时：记录 warning 后重试
        logger.warning(...)
    except aiohttp.ClientConnectorError:                       # replica 未 ready / 连接失败：记录 warning 后重试
        logger.warning(...)
    except aiohttp.ClientResponseError as exc:                 # HTTP 错误通常不可恢复，直接 raise
        logger.error(...)
        raise
    except Exception as exc:                                   # 其他异常最后一次才抛出
        logger.error(...)
        if attempt == self.max_attempts - 1:
            raise

    if attempt < self.max_attempts - 1:                        # 指数退避，避免打爆 replica scheduler
        await asyncio.sleep(self.retry_delay * (2**attempt))
```

### 11.5 `kunserve_manager/controller.py`：被保留的状态机核心

新的 import 区域：

```python
from kunserve_manager.client import (                  # HTTP/RPC helper 全部移到 client.py
    KunServeHttpReplicaClient,
    KunServeReplicaClient,
    _commit_response_succeeded,
    _parse_server_address,
    _response_error,
    _response_succeeded,
    _unwrap_output_list,
    _unwrap_status_response,
)
from kunserve_manager.layout import (                  # layout 规划移到 layout.py
    KunServeLayoutPlan,
    build_layout_plan_from_statuses,
)
from kunserve_manager.net import get_free_port          # 端口 helper 移到 net.py
from kunserve_manager.runtime_config import KunServeRuntimeBackendConfig  # backend 配置移到 runtime_config.py
```

构造函数新增 backend/capture：

```python
comm_backend: str = "deepep",                          # 默认不破坏旧 DeepEP 行为
capture_policy: str = "auto",                          # 默认不破坏旧 capture 行为
...
self.runtime_backend = KunServeRuntimeBackendConfig(    # 统一校验并保存
    comm_backend=comm_backend,
    capture_policy=capture_policy,
)
```

layout 规划现在只委托给 `layout.py`：

```python
def _ensure_layout_plan(self, statuses: Sequence[dict[str, Any]]) -> KunServeLayoutPlan:
    if self._layout_plan is None:                       # plan 只生成一次，后续复用
        self._layout_plan = build_layout_plan_from_statuses(
            statuses,                                   # 来自两个 replica 的 /kunserve/status
            offload_local_experts=self.offload_local_experts,
            num_replicas=len(self._replicas),
        )
    return self._layout_plan
```

payload 构造新增 comm_backend/capture 字段：

```python
effective_capture_cuda_graph = bool(                    # manager 端先算最终是否 capture
    capture_cuda_graph and self.runtime_backend.should_capture_global_graph()
)
backend_config = {                                      # 给 SGLang GLOBAL dispatcher 的数据面配置
    "exchange_mode": self.runtime_backend.exchange_mode,
    "local_ep_size": plan.local_ep_size,
    "num_replicas": len(self._replicas),
    "global_world_size": plan.global_world_size,
}
pg_names = {"global": self.group_name}                 # 预留多 PG；当前只用 global PG
...
{
    "target_variant": "global",                       # GLOBAL runtime bundle
    "runtime_ep_size": plan.global_world_size,         # 例如 4
    "runtime_rank_offset": replica_idx * plan.local_ep_size,  # replica0=0, replica1=2
    "dispatch_rank_offset": replica_idx * plan.local_ep_size, # dispatch rank 同 runtime rank
    "active_local_expert_mapping": plan.replica_active_mappings[replica_idx], # prefix/suffix
    "physical_to_logical_map": plan.global_physical_to_logical_map,           # GLOBAL expert map
    "process_group_name": self.group_name,             # 已初始化的跨 replica PG
    "capture_cuda_graph": effective_capture_cuda_graph,# sglang backend 当前为 False
    "kunserve_comm_backend": self.runtime_backend.comm_backend,  # deepep 或 sglang
    "capture_policy": self.runtime_backend.capture_policy,       # auto/fixed_padded/disabled
    "kunserve_pg_names": pg_names,                     # 当前仅记录 global PG 名
    "kunserve_backend_config": backend_config,         # 给 CrossReplicaStandardDispatcher 使用
}
```

### 11.6 `io_struct.py` / `scheduler.py` / `tp_worker.py`：字段贯通

`io_struct.py` 新增字段：

```python
kunserve_comm_backend: str = "deepep"                   # 默认保持 DeepEP 旧路径
capture_policy: str = "auto"                            # 默认保持旧 capture 策略
kunserve_pg_names: Optional[Dict[str, str]] = None       # 预留多个 process group 名称
kunserve_backend_config: Optional[Dict[str, Any]] = None # 后端专属配置，例如 local_ep_size
```

`scheduler.py` 只做日志增强：

```python
"comm_backend=%s capture_policy=%s capture_graph=%s",   # 日志里能直接确认 manager 传来的策略
recv_req.kunserve_comm_backend,                          # deepep/sglang
recv_req.capture_policy,                                 # auto/fixed_padded/disabled
recv_req.capture_cuda_graph,                             # 最终是否 capture
```

`tp_worker.py` 只做透传：

```python
kunserve_comm_backend=recv_req.kunserve_comm_backend,    # 传给 model_runner
capture_policy=recv_req.capture_policy,                  # 传给 model_runner
kunserve_pg_names=recv_req.kunserve_pg_names,            # 传给 model_runner
kunserve_backend_config=recv_req.kunserve_backend_config,# 传给 model_runner
```

### 11.7 `model_runner.py`：GLOBAL bundle 两条数据面分支

函数签名新增：

```python
kunserve_comm_backend: str = "deepep",                  # 当前 GLOBAL MoE 通信后端
capture_policy: str = "auto",                          # 当前 GLOBAL graph capture 策略
kunserve_pg_names: Optional[Dict[str, str]] = None,      # 预留多 PG
kunserve_backend_config: Optional[Dict[str, Any]] = None,# sglang backend 需要 local_ep_size 等
```

归一化/校验：

```python
kunserve_comm_backend = str(kunserve_comm_backend or "deepep").lower()  # 空值回退 deepep
capture_policy = str(capture_policy or "auto").lower()                  # 空值回退 auto
kunserve_backend_config = dict(kunserve_backend_config or {})            # None 转空 dict
if kunserve_comm_backend not in ("deepep", "sglang"):                   # 只允许已实现路径
    raise ValueError(...)
```

DeepEP 路径 guard 现在只在 `kunserve_comm_backend == "deepep"` 时生效：

```python
if kunserve_comm_backend == "deepep":                  # 只约束旧 DeepEP 数据面
    if moe_a2a_backend.is_none():                       # DeepEP 路径不能没有 A2A backend
        raise ValueError(...)
    if (moe_a2a_backend.is_deepep() or moe_a2a_backend.is_mooncake()) \
       and str(getattr(self.server_args, "moe_runner_backend", None)) != "deep_gemm":
        raise ValueError(...)                           # DeepEP/Mooncake 当前仍要求 deep_gemm
```

sglang backend 分支：

```python
if kunserve_comm_backend == "sglang":
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.token_dispatcher.kunserve_standard import (
        CrossReplicaStandardDispatcher,
    )
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    local_ep_size = int(                                # manager payload 显式传 local_ep_size
        kunserve_backend_config.get("local_ep_size") or self.moe_ep_size
    )
    explicit_global_dispatcher = CrossReplicaStandardDispatcher(
        group=runtime_group,                            # 跨 replica GLOBAL process group
        moe_runner_config=global_runner_config,          # num_local_experts 已改成 retained 数
        local_expert_mapping=dispatcher_local_expert_mapping, # physical expert -> compact local expert
        local_ep_size=local_ep_size,                     # 用于 lane select 和 replica_rank 推导
        replica_rank=resolved_moe_ep_rank // local_ep_size,
        global_rank=resolved_moe_ep_rank,
        world_size=resolved_ep_size,
    )
    explicit_global_runner = MoeRunner(                  # 不用 DeepGEMM，显式使用 Triton core
        MoeRunnerBackend.TRITON,
        global_runner_config,
    )
```

GLOBAL graph skip：

```python
skip_capture = (
    target_variant == "global"
    and (
        envs.SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL.get() # DeepEP NORMAL dynamic shape 不 capture
        or str(kunserve_comm_backend or "deepep").lower() == "sglang" # sglang 初版也不 capture
    )
)
```

status 新增：

```python
"kunserve_comm_backend": self._balloon_kunserve_comm_backend, # /kunserve/status 可见当前数据面
"kunserve_capture_policy": self._balloon_capture_policy,     # /kunserve/status 可见 capture 策略
```

### 11.8 `CrossReplicaStandardDispatcher`：完整核心逻辑逐行注释

文件：`python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`

构造函数：

```python
self.group = group                                      # 跨 replica GLOBAL process group
self.local_ep_size = int(local_ep_size)                 # 单个 replica 内 EP size
self.global_rank = int(global_rank) ...                 # 当前 rank 在 GLOBAL group 中的 rank
self.world_size = int(world_size) ...                   # GLOBAL world size，例如 4
if self.world_size % self.local_ep_size != 0:           # 必须能整除才能推导 replica 数
    raise ValueError(...)

self.num_replicas = self.world_size // self.local_ep_size # 当前为 2
self.replica_rank = ...                                 # 当前 rank 属于第几个 replica
self.lane_rank = self.global_rank % self.local_ep_size  # rank0/rank2 是 lane0，rank1/rank3 是 lane1

self.num_experts = int(moe_runner_config.num_experts)   # GLOBAL dispatch domain expert 数
self.top_k = int(moe_runner_config.top_k)               # router topk
self.num_local_experts = int(moe_runner_config.num_local_experts) # retained experts 数
self.local_expert_mapping = self._normalize_local_expert_mapping(local_expert_mapping)
self.active_local_expert_mapping = self.local_expert_mapping
```

`dispatch()`：

```python
if not TopKOutputChecker.format_is_standard(topk_output): # 初版只支持 StandardTopKOutput
    raise NotImplementedError(...)

local_m = int(hidden_states.shape[0])                     # 本 rank 当前 local token 数
sizes = self._all_gather_sizes(local_m, hidden_states.device) # gather 所有 GLOBAL rank token 数
max_m = int(sizes.max().item()) if sizes.numel() > 0 else local_m # pad 到最大 token 数

padded_hidden = self._pad_dim0(hidden_states, max_m=max_m, pad_value=0)       # hidden padding 用 0
padded_topk_ids = self._pad_dim0(topk_output.topk_ids, max_m=max_m, pad_value=-1) # dummy expert id=-1
padded_topk_weights = self._pad_dim0(topk_output.topk_weights, max_m=max_m, pad_value=0) # dummy 权重 0

gathered_hidden = self._all_gather_padded(padded_hidden)       # 所有 rank hidden tensor list
gathered_topk_ids = self._all_gather_padded(padded_topk_ids)   # 所有 rank topk_ids tensor list
gathered_topk_weights = self._all_gather_padded(padded_topk_weights) # 所有 rank weights tensor list

union_hidden = self._select_lane_segments(gathered_hidden)     # 当前 lane 选每个 replica 对应 lane 的 segment
union_topk_ids = self._select_lane_segments(gathered_topk_ids) # 同样选择 ids
union_topk_weights = self._select_lane_segments(gathered_topk_weights) # 同样选择 weights

self._last_local_m = local_m                                  # combine 时 slice 回本地需要
self._last_max_m = max_m                                      # combine 时计算 replica segment 起点需要
self._last_slice_start = self.replica_rank * max_m            # 当前 replica 在 union tensor 中的起点

return StandardDispatchOutput(
    hidden_states=union_hidden,                               # 给 Triton runner 的 union token hidden
    hidden_states_scale=None,                                 # 当前 bf16/no-quant 或 standard fp8 path 不需要额外 hidden scale
    topk_output=StandardTopKOutput(
        topk_weights=union_topk_weights,                      # union token 的 router weights
        topk_ids=self._remap_topk_ids(union_topk_ids),         # physical ids -> compact local ids / -1
        router_logits=router_logits,                          # router logits 同步到 union shape（如可同步）
    ),
)
```

`combine()`：

```python
(hidden_states,) = combine_input                              # Triton runner 输出的是当前 rank partial sum
hidden_states = hidden_states.contiguous()                    # NCCL all_reduce 要求连续更安全
dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=self.group) # 所有 GLOBAL rank partial sum 求和

start = int(self._last_slice_start)                           # 当前 replica 在 union tensor 中的起点
end = start + int(self._last_local_m)                         # 只取真实 local token，不取 padding
return hidden_states[start:end].contiguous()                  # 恢复 FusedMoE 对外 local batch shape
```

### 11.9 `fused_moe_triton/layer.py` 与 `fp8.py`：active tensor 修正

`layer.py` 新增 active tensor 名：

```python
"w13_input_scale",      # FP8 static activation scale，随 active experts 一起切片
"w2_input_scale",       # 同上
```

`fp8.py` Triton branch 改为读 runtime tensor：

```python
get_runtime_tensor = getattr(layer, "get_runtime_tensor", None) # GLOBAL bundle 下返回 active_tensors 中的切片
if get_runtime_tensor is None:
    def get_runtime_tensor(name):
        return getattr(layer, name)                              # 非 bundle 场景回退原参数
...
w13_weight=get_runtime_tensor("w13_weight"),                    # 不再直接 layer.w13_weight
w2_weight=get_runtime_tensor("w2_weight"),                      # 避免 replica1 suffix experts 读错 rows
w13_scale=(get_runtime_tensor("w13_weight_scale_inv") if self.block_quant else get_runtime_tensor("w13_weight_scale")),
w2_scale=(get_runtime_tensor("w2_weight_scale_inv") if self.block_quant else get_runtime_tensor("w2_weight_scale")),
a13_scale=get_runtime_tensor("w13_input_scale"),
a2_scale=get_runtime_tensor("w2_input_scale"),
```

### 11.10 verl 接线：config 与 agent_loop

`rollout.py` 新增：

```python
kunserve_comm_backend: str = "deepep"       # 默认旧路径；脚本可覆盖成 sglang
kunserve_capture_policy: str = "auto"       # 默认旧策略；脚本可覆盖 disabled
```

`agent_loop.py` 启动 manager 时新增 CLI 参数：

```python
"--comm-backend",
str(OmegaConf.select(cfg, "kunserve_comm_backend", default="deepep")),
"--capture-policy",
str(OmegaConf.select(cfg, "kunserve_capture_policy", default="auto")),
```

### 11.11 脚本：bf16/no-quant 默认逻辑

`run_smoke_test_kunserve_tp2_dual_replica.sh`：

```bash
export KUNSERVE_COMM_BACKEND=${KUNSERVE_COMM_BACKEND:-sglang}       # 当前默认使用 sglang backend
export KUNSERVE_CAPTURE_POLICY=${KUNSERVE_CAPTURE_POLICY:-disabled} # sglang backend 初版不 capture
if [ "${KUNSERVE_COMM_BACKEND}" = "sglang" ]; then
    export KUNSERVE_MOE_A2A_BACKEND=${KUNSERVE_MOE_A2A_BACKEND:-none}       # 不走 DeepEP
    export KUNSERVE_MOE_RUNNER_BACKEND=${KUNSERVE_MOE_RUNNER_BACKEND:-triton} # 不走 DeepGEMM
    KUNSERVE_ROLLOUT_QUANTIZATION=${KUNSERVE_ROLLOUT_QUANTIZATION:-none}    # 默认不量化
else
    export KUNSERVE_MOE_A2A_BACKEND=${KUNSERVE_MOE_A2A_BACKEND:-deepep}     # DeepEP 路径
    export KUNSERVE_MOE_RUNNER_BACKEND=${KUNSERVE_MOE_RUNNER_BACKEND:-deep_gemm} # DeepGEMM 路径
    KUNSERVE_ROLLOUT_QUANTIZATION=${KUNSERVE_ROLLOUT_QUANTIZATION:-fp8}     # 旧路径默认 fp8
fi
ROLLOUT_QUANT_ARGS=()                                                       # 默认不传 quantization override
case "${KUNSERVE_ROLLOUT_QUANTIZATION}" in
    ""|none|null|None|bf16|bfloat16|false|False|0)                         # 这些都表示 bf16/no-quant
        ;;
    *)
        ROLLOUT_QUANT_ARGS=(actor_rollout_ref.rollout.quantization=${KUNSERVE_ROLLOUT_QUANTIZATION})
        ;;
esac
```

在 `ROLLOUT=(...)` 中：

```bash
"${ROLLOUT_QUANT_ARGS[@]}"   # 如果 no-quant，则展开为空；Hydra 不会收到 rollout.quantization
```

`run_smoke_test_single_sglang_tp.sh` 同理默认：

```bash
rollout_quantization=${ROLLOUT_QUANTIZATION:-none} # TP compare 默认 no-quant
moe_a2a_backend=${MOE_A2A_BACKEND:-none}           # 不走 DeepEP
moe_runner_backend=${MOE_RUNNER_BACKEND:-triton}   # bf16 standard Triton path
```

`compare_kunserve_vs_baseline.sh` 默认导出：

```bash
: "${KUNSERVE_COMM_BACKEND:=sglang}"              # AB 的 KunServe run 默认用 sglang backend
: "${KUNSERVE_ROLLOUT_QUANTIZATION:=none}"        # AB 的 KunServe run 默认 bf16/no-quant
```

这保证当前 compare 目标不是 FP8，而是 bf16/no-quant。


---

## 12. 完整修改代码索引（raw patch，便于逐行核对）

第 11 节给出逐行语义注释；本节贴出当前工作区修改的原始代码/patch，方便你按文件逐行核对。这里不把本文档自身再次贴入，避免递归。

### 12.x 新增文件 `kunserve_manager/kunserve_manager/client.py`

```python
"""HTTP client and response helpers for KunServe replica control-plane RPCs."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from urllib.parse import urlsplit

import aiohttp

logger = logging.getLogger(__name__)


async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    if resp.status == 204 or resp.content_length == 0:
        return {}

    try:
        return await resp.json(content_type=None)
    except Exception:
        try:
            text = await resp.text()
        except Exception:
            return {}
        return {
            "content_type": resp.headers.get("Content-Type", ""),
            "text": text,
        }


class KunServeReplicaClient(Protocol):
    name: str
    host: str

    async def get_balloon_status(self) -> dict[str, Any]: ...

    async def prepare_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def warmup_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def commit_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def restore_from_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def init_weights_update_group(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def destroy_weights_update_group(self, group_name: str) -> dict[str, Any]: ...


def _parse_server_address(server_address: str) -> tuple[str, int]:
    parts = urlsplit(f"http://{server_address}")
    if parts.hostname is None or parts.port is None:
        raise ValueError(f"Invalid server address: {server_address}")
    return parts.hostname, parts.port


def _unwrap_status_response(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        if not raw:
            raise ValueError("Empty balloon status response.")
        return dict(raw[0])
    if isinstance(raw, dict):
        return dict(raw)
    raise TypeError(f"Unsupported balloon status response type: {type(raw)!r}")


def _unwrap_output_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    if isinstance(raw, dict):
        return [dict(raw)]
    raise TypeError(f"Unsupported RPC response type: {type(raw)!r}")


def _response_succeeded(raw: Any) -> bool:
    outputs = _unwrap_output_list(raw)
    return bool(outputs) and all(bool(item.get("success")) for item in outputs)


def _response_error(raw: Any) -> str:
    outputs = _unwrap_output_list(raw)
    for item in outputs:
        if not item.get("success", False):
            return str(item.get("message", "unknown error"))
    return "unknown error"


def _status_matches_balloon_target(
    raw: Any, *, target_variant: str, offload_local_experts: int
) -> bool:
    try:
        status = _unwrap_status_response(raw)
    except (TypeError, ValueError):
        return False

    return (
        str(status.get("state", "")).lower() == "balloon"
        and str(status.get("runtime_variant", "")).lower()
        == str(target_variant).lower()
        and int(status.get("offloaded_local_experts", -1)) == int(offload_local_experts)
    )


def _commit_response_succeeded(
    raw: Any, *, target_variant: str, offload_local_experts: int
) -> bool:
    if _response_succeeded(raw):
        return True

    outputs = _unwrap_output_list(raw)
    return bool(outputs) and all(
        _status_matches_balloon_target(
            item.get("status"),
            target_variant=target_variant,
            offload_local_experts=offload_local_experts,
        )
        for item in outputs
    )


@dataclass
class KunServeHttpReplicaClient:
    name: str
    host: str
    port: int
    model_path: str
    timeout: float = 60.0
    # warmup_balloon does GLOBAL cuda graph capture which empirically takes
    # 5-10 minutes on H20 (deepgemm precompile + LL Buffer creation +
    # 35 batch sizes). The default 60s × 3 attempts (180s) is way too short —
    # in ab_20260506_172822 it triggered a false-positive "warmup failed"
    # while the capture was actually mid-flight. Override that one RPC.
    warmup_timeout: float = 600.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_start_wait_time: float = 300.0
    max_connections: int = 64

    def __post_init__(self) -> None:
        # Control-plane clients only talk to already-running HTTP servers.
        # They must not instantiate SGLang ServerArgs here because the controller
        # runs in a non-GPU Ray actor process where accelerator probing fails.
        self._base_url = f"http://{self.host}:{self.port}"
        logger.info(
            "[KunServeHttpReplicaClient] configured control-plane HTTP client for %s at %s",
            self.name,
            self._base_url,
        )
        print(
            f"[KunServeHttpReplicaClient:{self.name}] configured at {self._base_url}",
            flush=True,
        )

    @asynccontextmanager
    async def _get_session(self, timeout: Optional[float] = None):
        connector = aiohttp.TCPConnector(
            limit=max(1, self.max_connections),
            limit_per_host=max(1, self.max_connections // 4),
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        effective_timeout = self.timeout if timeout is None else float(timeout)
        client_timeout = aiohttp.ClientTimeout(total=effective_timeout)
        session = aiohttp.ClientSession(connector=connector, timeout=client_timeout)
        try:
            yield session
        finally:
            if not session.closed:
                await session.close()

    async def _request(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
        *,
        method: str = "POST",
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/{endpoint}"
        should_trace = endpoint != "kunserve/status"
        if should_trace:
            print(
                f"[KunServeHttpReplicaClient:{self.name}] {method.upper()} {endpoint} payload={payload or {}}",
                flush=True,
            )

        for attempt in range(self.max_attempts):
            try:
                async with self._get_session(timeout=timeout) as session:
                    if method.upper() == "GET":
                        async with session.get(url) as response:
                            response.raise_for_status()
                            result = await _read_async_response(response)
                            if should_trace:
                                print(
                                    f"[KunServeHttpReplicaClient:{self.name}] {endpoint} response={result}",
                                    flush=True,
                                )
                            return result
                    async with session.post(url, json=payload or {}) as response:
                        response.raise_for_status()
                        result = await _read_async_response(response)
                        if should_trace:
                            print(
                                f"[KunServeHttpReplicaClient:{self.name}] {endpoint} response={result}",
                                flush=True,
                            )
                        return result
            except asyncio.TimeoutError:
                logger.warning(
                    "[KunServeHttpReplicaClient] %s %s timed out (%d/%d)",
                    self.name,
                    endpoint,
                    attempt + 1,
                    self.max_attempts,
                )
            except aiohttp.ClientConnectorError:
                logger.warning(
                    "[KunServeHttpReplicaClient] %s %s connection error (%d/%d)",
                    self.name,
                    endpoint,
                    attempt + 1,
                    self.max_attempts,
                )
            except aiohttp.ClientResponseError as exc:
                logger.error(
                    "[KunServeHttpReplicaClient] %s %s HTTP error: %s",
                    self.name,
                    endpoint,
                    exc,
                )
                raise
            except Exception as exc:
                logger.error(
                    "[KunServeHttpReplicaClient] %s %s unexpected error: %s",
                    self.name,
                    endpoint,
                    exc,
                )
                if attempt == self.max_attempts - 1:
                    raise

            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(
            f"[KunServeHttpReplicaClient] {self.name} failed to call {endpoint} "
            f"after {self.max_attempts} attempts"
        )

    async def get_balloon_status(self) -> dict[str, Any]:
        return await self._request("kunserve/status", method="GET")

    async def prepare_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("kunserve/prepare_balloon", payload)

    async def warmup_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
        # warmup_balloon includes GLOBAL cuda graph capture which can take
        # 5-10 minutes on H20. Use the dedicated warmup_timeout (default 600s)
        # instead of the per-RPC default (60s) to avoid spuriously aborting
        # an in-flight capture.
        return await self._request(
            "kunserve/warmup_balloon", payload, timeout=self.warmup_timeout
        )

    async def commit_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
        # commit_balloon does borrow_tail (per-layer cuMemUnmap) + KV expand
        # (cuMemMap into the KV region for hundreds of donor segments). On
        # 4-rank kunserve at 32 offloaded experts × 48 layers this measured
        # ~67 s end-to-end (see ab_20260429_155927). Reuse warmup_timeout so
        # we don't trip the 60 s per-RPC default mid-mapping.
        return await self._request(
            "kunserve/commit_balloon", payload, timeout=self.warmup_timeout
        )

    async def restore_from_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("kunserve/restore_from_balloon", payload)

    async def init_weights_update_group(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request("init_weights_update_group", payload)

    async def destroy_weights_update_group(self, group_name: str) -> dict[str, Any]:
        return await self._request(
            "destroy_weights_update_group", {"group_name": group_name}
        )
```

### 12.x 新增文件 `kunserve_manager/kunserve_manager/layout.py`

```python
"""Expert layout planning helpers for the KunServe sidecar manager."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class KunServeLayoutPlan:
    """Per-run two-replica expert sharing plan.

    ``local_*`` fields describe each pre-BALLOON SGLang replica.  ``global_*``
    fields describe the temporary cross-replica GLOBAL EP world used while
    BALLOON is active.  ``replica_active_mappings`` is the compact local expert
    slice retained by each replica after half of the local experts are loaned to
    the peer.
    """

    local_ep_size: int
    local_routed_experts: int
    retained_local_experts: int
    offload_local_experts: int
    global_world_size: int
    global_physical_to_logical_map: list[list[int]]
    replica_active_mappings: list[list[int]]


def build_complementary_physical_to_logical_map(
    base_physical_to_logical_map: Sequence[Sequence[int]],
    *,
    local_ep_size: int,
    retained_local_experts: int,
) -> list[list[int]]:
    """Build the GLOBAL physical->logical expert map for two replicas.

    The current KunServe implementation assumes a symmetric half split.  For
    each local EP rank we first place the prefix half retained by replica 0, then
    the suffix half retained by replica 1.  That keeps the total physical expert
    count unchanged while making the GLOBAL dispatch domain cover both replicas'
    retained rows.
    """

    if local_ep_size <= 0:
        raise ValueError(f"local_ep_size must be positive, got {local_ep_size}")
    if retained_local_experts <= 0:
        raise ValueError(
            f"retained_local_experts must be positive, got {retained_local_experts}"
        )

    normalized = [list(map(int, row)) for row in base_physical_to_logical_map]
    if not normalized:
        raise ValueError("base_physical_to_logical_map must be non-empty")

    num_physical_experts = len(normalized[0])
    if any(len(row) != num_physical_experts for row in normalized):
        raise ValueError("All physical_to_logical rows must have the same length.")
    if num_physical_experts % local_ep_size != 0:
        raise ValueError(
            f"num_physical_experts={num_physical_experts} is not divisible by local_ep_size={local_ep_size}"
        )

    local_chunk = num_physical_experts // local_ep_size
    if retained_local_experts * 2 != local_chunk:
        raise ValueError(
            "Complementary two-replica balloon requires a symmetric half split per local rank."
        )

    merged: list[list[int]] = []
    for layer_row in normalized:
        global_row: list[int] = []
        for local_rank in range(local_ep_size):
            base = local_rank * local_chunk
            global_row.extend(layer_row[base : base + retained_local_experts])
        for local_rank in range(local_ep_size):
            base = local_rank * local_chunk + (local_chunk - retained_local_experts)
            global_row.extend(layer_row[base : base + retained_local_experts])
        if len(global_row) != num_physical_experts:
            raise ValueError(
                f"Expected merged row length {num_physical_experts}, got {len(global_row)}"
            )
        merged.append(global_row)
    return merged


def build_layout_plan_from_statuses(
    statuses: Sequence[dict[str, Any]],
    *,
    offload_local_experts: Optional[int],
    num_replicas: int = 2,
) -> KunServeLayoutPlan:
    """Derive a KunServe layout plan from replica ``/kunserve/status`` payloads."""

    if len(statuses) != num_replicas:
        raise ValueError(
            f"Expected {num_replicas} replica status payloads, got {len(statuses)}."
        )

    local_maps = [status.get("local_physical_to_logical_map") for status in statuses]
    if any(local_map is None for local_map in local_maps):
        raise ValueError("Balloon status is missing local_physical_to_logical_map.")
    if local_maps[0] != local_maps[1]:
        raise ValueError(
            "Replicas do not agree on the baseline physical_to_logical expert layout."
        )

    local_ep_size = int(statuses[0]["local_ep_size"])
    routed_by_layer = {
        int(layer_id): int(count)
        for layer_id, count in statuses[0]["local_routed_experts_per_layer"].items()
    }
    routed_values = set(routed_by_layer.values())
    if len(routed_values) != 1:
        raise ValueError(
            "Current KunServe controller requires all MoE layers to expose the same local routed expert count."
        )
    local_routed_experts = routed_values.pop()

    resolved_offload = (
        int(offload_local_experts)
        if offload_local_experts is not None
        else local_routed_experts // 2
    )
    if resolved_offload <= 0 or resolved_offload >= local_routed_experts:
        raise ValueError(
            f"Invalid offload_local_experts={resolved_offload} for local_routed_experts={local_routed_experts}"
        )
    retained_local_experts = local_routed_experts - resolved_offload
    if retained_local_experts != resolved_offload:
        raise ValueError(
            "Current KunServe controller requires a symmetric half split to preserve the original expert count."
        )

    global_map = build_complementary_physical_to_logical_map(
        local_maps[0],
        local_ep_size=local_ep_size,
        retained_local_experts=retained_local_experts,
    )
    replica_active_mappings = [
        list(range(retained_local_experts)),
        list(range(local_routed_experts - retained_local_experts, local_routed_experts)),
    ]
    plan = KunServeLayoutPlan(
        local_ep_size=local_ep_size,
        local_routed_experts=local_routed_experts,
        retained_local_experts=retained_local_experts,
        offload_local_experts=resolved_offload,
        global_world_size=local_ep_size * num_replicas,
        global_physical_to_logical_map=global_map,
        replica_active_mappings=replica_active_mappings,
    )
    logger.info(
        "[KunServeController] layout plan ready: local_ep_size=%d local_routed=%d retained=%d offload=%d",
        plan.local_ep_size,
        plan.local_routed_experts,
        plan.retained_local_experts,
        plan.offload_local_experts,
    )
    return plan
```

### 12.x 新增文件 `kunserve_manager/kunserve_manager/net.py`

```python
"""Small networking helpers for the KunServe sidecar manager."""

from __future__ import annotations

import socket


def get_free_port(host: str = "") -> tuple[int, socket.socket]:
    """Return a currently-free TCP port on ``host``.

    The probe socket is closed before returning, matching the historical
    controller behavior.  The caller should still treat the result as best
    effort because another process may bind the port before the HTTP RPC reaches
    replica rank 0.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host or "", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port, sock
```

### 12.x 新增文件 `kunserve_manager/kunserve_manager/runtime_config.py`

```python
"""Runtime backend configuration for KunServe GLOBAL data-plane choices."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KunServeRuntimeBackendConfig:
    """Data-plane backend knobs sent from the sidecar to every sglang replica."""

    comm_backend: str = "deepep"
    capture_policy: str = "auto"
    exchange_mode: str = "global_dense_v0"

    def __post_init__(self) -> None:
        comm_backend = str(self.comm_backend).lower()
        capture_policy = str(self.capture_policy).lower()
        if comm_backend not in ("deepep", "sglang"):
            raise ValueError(
                f"Unsupported KunServe comm_backend={self.comm_backend!r}; "
                "expected 'deepep' or 'sglang'."
            )
        if capture_policy not in ("auto", "fixed_padded", "disabled"):
            raise ValueError(
                f"Unsupported KunServe capture_policy={self.capture_policy!r}; "
                "expected 'auto', 'fixed_padded', or 'disabled'."
            )
        object.__setattr__(self, "comm_backend", comm_backend)
        object.__setattr__(self, "capture_policy", capture_policy)

    def should_capture_global_graph(self) -> bool:
        if self.capture_policy == "disabled":
            return False
        if self.comm_backend == "sglang":
            # P0 sglang backend uses dynamic padded all-gather/all-reduce.
            # Capture support needs a later fixed-padded graph implementation;
            # until that lands, even capture_policy=fixed_padded is treated as
            # disabled rather than attempting an unsafe graph capture.
            return False
        return True
```

### 12.x 新增文件 `python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py`

```python
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)

logger = logging.getLogger(__name__)


class CrossReplicaStandardDispatcher(BaseDispatcher):
    """KunServe GLOBAL dispatcher implemented with plain torch collectives.

    This is the correctness-first, eager-mode implementation for KunServe's
    cross-replica expert sharing path.  It deliberately preserves the normal
    FusedMoE contract:

    * input to ``dispatch`` is this replica's local-token hidden states;
    * output from ``combine`` is sliced back to the same local-token shape.

    Internally, each global EP rank performs a padded all-gather across the
    cross-replica process group, selects the same tensor-parallel/EP lane from
    every replica, computes only its retained local experts, then all-reduces
    the partial MoE sums across the full global EP group.

    This is not CUDA-graph safe in its initial form because token counts are
    dynamic and ``torch.distributed`` collectives allocate/work eagerly.  The
    manager therefore sends ``capture_cuda_graph=False`` when this dispatcher is
    selected.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        moe_runner_config: MoeRunnerConfig,
        local_expert_mapping: torch.Tensor,
        local_ep_size: int,
        replica_rank: Optional[int] = None,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if group is None:
            raise ValueError("CrossReplicaStandardDispatcher requires a process group.")
        if local_ep_size <= 0:
            raise ValueError(f"local_ep_size must be positive, got {local_ep_size}.")

        self.group = group
        self.local_ep_size = int(local_ep_size)
        self.global_rank = (
            int(global_rank)
            if global_rank is not None
            else int(dist.get_rank(group=group))
        )
        self.world_size = (
            int(world_size)
            if world_size is not None
            else int(dist.get_world_size(group=group))
        )
        if self.world_size % self.local_ep_size != 0:
            raise ValueError(
                "CrossReplicaStandardDispatcher requires global world size to be "
                f"divisible by local_ep_size, got world={self.world_size}, "
                f"local_ep_size={self.local_ep_size}."
            )

        self.num_replicas = self.world_size // self.local_ep_size
        self.replica_rank = (
            int(replica_rank)
            if replica_rank is not None
            else self.global_rank // self.local_ep_size
        )
        self.lane_rank = self.global_rank % self.local_ep_size

        self.num_experts = int(moe_runner_config.num_experts)
        self.top_k = int(moe_runner_config.top_k)
        self.num_local_experts = int(moe_runner_config.num_local_experts)
        self.local_expert_mapping = self._normalize_local_expert_mapping(
            local_expert_mapping
        )
        self.active_local_expert_mapping = self.local_expert_mapping

        # State saved by dispatch and consumed by the matching combine.  A
        # FusedMoE layer calls dispatch -> run_moe_core -> combine
        # synchronously, so a single in-flight state per dispatcher is enough.
        self._last_local_m: Optional[int] = None
        self._last_max_m: Optional[int] = None
        self._last_slice_start: Optional[int] = None
        self._logged_shape: bool = False

    def _normalize_local_expert_mapping(self, mapping: torch.Tensor) -> torch.Tensor:
        if not isinstance(mapping, torch.Tensor):
            mapping = torch.tensor(mapping)
        if mapping.dim() != 1:
            raise ValueError("local_expert_mapping must be a 1D tensor.")
        if int(mapping.numel()) != self.num_experts:
            raise ValueError(
                "local_expert_mapping must contain one entry per dispatch-domain "
                f"expert, got {int(mapping.numel())} vs {self.num_experts}."
            )
        return mapping.to(dtype=torch.int32)

    def _mapping_on(self, device: torch.device) -> torch.Tensor:
        if self.local_expert_mapping.device != device:
            self.local_expert_mapping = self.local_expert_mapping.to(
                device=device, non_blocking=True
            )
            self.active_local_expert_mapping = self.local_expert_mapping
        return self.local_expert_mapping

    def _all_gather_sizes(self, local_m: int, device: torch.device) -> torch.Tensor:
        local_size = torch.tensor([int(local_m)], dtype=torch.int64, device=device)
        gathered = [torch.empty_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_size, group=self.group)
        sizes = torch.cat(gathered, dim=0)

        # In SGLang's TP/EP=local_ep_size baseline, ranks in the same replica
        # carry the same token batch.  The cross-replica Standard-like
        # algorithm relies on that invariant because lane 0 and lane 1 produce
        # partial sums for the same union-token order before all-reduce.
        sizes_cpu = sizes.detach().cpu().tolist()
        for replica_idx in range(self.num_replicas):
            base = replica_idx * self.local_ep_size
            replica_sizes = sizes_cpu[base : base + self.local_ep_size]
            if len(set(replica_sizes)) != 1:
                raise RuntimeError(
                    "CrossReplicaStandardDispatcher requires all local EP ranks "
                    "inside a replica to see the same token count. "
                    f"replica={replica_idx} sizes={replica_sizes} all_sizes={sizes_cpu}"
                )
        return sizes

    def _pad_dim0(
        self,
        tensor: torch.Tensor,
        *,
        max_m: int,
        pad_value: float | int = 0,
    ) -> torch.Tensor:
        local_m = int(tensor.shape[0])
        if local_m == max_m:
            return tensor.contiguous()
        out = torch.full(
            (max_m, *tensor.shape[1:]),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if local_m > 0:
            out[:local_m].copy_(tensor)
        return out

    def _all_gather_padded(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=self.group)
        return gathered

    def _select_lane_segments(self, gathered: list[torch.Tensor]) -> torch.Tensor:
        segments = [
            gathered[replica_idx * self.local_ep_size + self.lane_rank]
            for replica_idx in range(self.num_replicas)
        ]
        return torch.cat(segments, dim=0).contiguous()

    def _remap_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        mapping = self._mapping_on(topk_ids.device)
        remapped = torch.full_like(topk_ids, -1, dtype=torch.int32)
        valid = (topk_ids >= 0) & (topk_ids < self.num_experts)
        if bool(valid.any()):
            remapped[valid] = mapping[topk_ids[valid].to(dtype=torch.long)]
        return remapped

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise NotImplementedError(
                "CrossReplicaStandardDispatcher currently supports only "
                f"StandardTopKOutput, got {type(topk_output)!r}."
            )

        local_m = int(hidden_states.shape[0])
        sizes = self._all_gather_sizes(local_m, hidden_states.device)
        max_m = int(sizes.max().item()) if sizes.numel() > 0 else local_m

        padded_hidden = self._pad_dim0(hidden_states, max_m=max_m, pad_value=0)
        padded_topk_ids = self._pad_dim0(
            topk_output.topk_ids, max_m=max_m, pad_value=-1
        )
        padded_topk_weights = self._pad_dim0(
            topk_output.topk_weights, max_m=max_m, pad_value=0
        )

        gathered_hidden = self._all_gather_padded(padded_hidden)
        gathered_topk_ids = self._all_gather_padded(padded_topk_ids)
        gathered_topk_weights = self._all_gather_padded(padded_topk_weights)

        union_hidden = self._select_lane_segments(gathered_hidden)
        union_topk_ids = self._select_lane_segments(gathered_topk_ids)
        union_topk_weights = self._select_lane_segments(gathered_topk_weights)

        router_logits = topk_output.router_logits
        if (
            isinstance(router_logits, torch.Tensor)
            and router_logits.dim() >= 1
            and int(router_logits.shape[0]) == local_m
        ):
            padded_router_logits = self._pad_dim0(
                router_logits, max_m=max_m, pad_value=0
            )
            router_logits = self._select_lane_segments(
                self._all_gather_padded(padded_router_logits)
            )

        self._last_local_m = local_m
        self._last_max_m = max_m
        self._last_slice_start = self.replica_rank * max_m

        if not self._logged_shape:
            logger.warning(
                "[KUNSERVE-MS] CrossReplicaStandardDispatcher active: "
                "rank=%d world=%d local_ep=%d replica=%d lane=%d "
                "local_m=%d max_m=%d union_m=%d",
                self.global_rank,
                self.world_size,
                self.local_ep_size,
                self.replica_rank,
                self.lane_rank,
                local_m,
                max_m,
                int(union_hidden.shape[0]),
            )
            self._logged_shape = True

        return StandardDispatchOutput(
            hidden_states=union_hidden,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=union_topk_weights,
                topk_ids=self._remap_topk_ids(union_topk_ids),
                router_logits=router_logits,
            ),
        )

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        if (
            self._last_local_m is None
            or self._last_max_m is None
            or self._last_slice_start is None
        ):
            raise RuntimeError(
                "CrossReplicaStandardDispatcher.combine called before dispatch."
            )

        (hidden_states,) = combine_input
        hidden_states = hidden_states.contiguous()
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=self.group)

        start = int(self._last_slice_start)
        end = start + int(self._last_local_m)
        return hidden_states[start:end].contiguous()
```

### 12.y SGLang tracked 修改 diff

```diff
diff --git a/kunserve_manager/kunserve_manager/cli.py b/kunserve_manager/kunserve_manager/cli.py
index 7167b3731..e14e15428 100644
--- a/kunserve_manager/kunserve_manager/cli.py
+++ b/kunserve_manager/kunserve_manager/cli.py
@@ -93,6 +93,26 @@ def _build_parser() -> argparse.ArgumentParser:
         default=_env_or("KUNSERVE_MANAGER_BACKEND", "nccl"),
         help="init_weights_update_group backend (default: nccl).",
     )
+    p.add_argument(
+        "--comm-backend",
+        choices=("deepep", "sglang"),
+        default=_env_or("KUNSERVE_MANAGER_COMM_BACKEND", "deepep"),
+        help=(
+            "KunServe GLOBAL MoE communication backend. 'deepep' keeps the "
+            "existing DeepEP dispatcher path; 'sglang' uses the "
+            "correctness-first CrossReplicaStandardDispatcher (default: deepep)."
+        ),
+    )
+    p.add_argument(
+        "--capture-policy",
+        choices=("auto", "fixed_padded", "disabled"),
+        default=_env_or("KUNSERVE_MANAGER_CAPTURE_POLICY", "auto"),
+        help=(
+            "GLOBAL CUDA graph capture policy. For comm-backend=sglang this "
+            "currently resolves to disabled because the initial dispatcher uses "
+            "dynamic all-gather/all-reduce collectives."
+        ),
+    )
     p.add_argument(
         "--enable-restore",
         action="store_true",
@@ -201,6 +221,8 @@ def main(argv: Optional[list[str]] = None) -> int:
         offload_local_experts=args.offload_local_experts,
         group_name=args.group_name,
         backend=args.backend,
+        comm_backend=args.comm_backend,
+        capture_policy=args.capture_policy,
         enable_restore=args.enable_restore,
         eager_warmup=args.eager_warmup,
         output_dir=args.output_dir,
@@ -209,12 +231,15 @@ def main(argv: Optional[list[str]] = None) -> int:
 
     logger.info(
         "kunserve_manager starting: replicas=%s model_path=%s poll=%.2fs "
-        "group=%s backend=%s eager_warmup=%s enable_restore=%s output_dir=%s bw_log=%s",
+        "group=%s backend=%s comm_backend=%s capture_policy=%s eager_warmup=%s "
+        "enable_restore=%s output_dir=%s bw_log=%s",
         replicas,
         args.model_path,
         args.poll_interval,
         args.group_name,
         args.backend,
+        args.comm_backend,
+        args.capture_policy,
         args.eager_warmup,
         args.enable_restore,
         args.output_dir,
diff --git a/kunserve_manager/kunserve_manager/controller.py b/kunserve_manager/kunserve_manager/controller.py
index f075f26b4..4fb356b9f 100644
--- a/kunserve_manager/kunserve_manager/controller.py
+++ b/kunserve_manager/kunserve_manager/controller.py
@@ -3,369 +3,33 @@ from __future__ import annotations
 import asyncio
 import json
 import logging
-import random
 import os
+import random
 import threading
 import time
-from contextlib import asynccontextmanager
-from dataclasses import dataclass
 from pathlib import Path
-from typing import Any, Optional, Protocol, Sequence
-from urllib.parse import urlsplit
-
-import aiohttp
-
-import socket
-
-
-def get_free_port(host: str = "") -> tuple[int, socket.socket]:
-    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
-    sock.bind((host or "", 0))
-    port = int(sock.getsockname()[1])
-    sock.close()
-    return port, sock
-
+from typing import Any, Optional, Sequence
+
+from kunserve_manager.client import (
+    KunServeHttpReplicaClient,
+    KunServeReplicaClient,
+    _commit_response_succeeded,
+    _parse_server_address,
+    _response_error,
+    _response_succeeded,
+    _unwrap_output_list,
+    _unwrap_status_response,
+)
+from kunserve_manager.layout import (
+    KunServeLayoutPlan,
+    build_layout_plan_from_statuses,
+)
+from kunserve_manager.net import get_free_port
+from kunserve_manager.runtime_config import KunServeRuntimeBackendConfig
 
 logger = logging.getLogger(__name__)
 
 
-async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
-    if resp.status == 204 or resp.content_length == 0:
-        return {}
-
-    try:
-        return await resp.json(content_type=None)
-    except Exception:
-        try:
-            text = await resp.text()
-        except Exception:
-            return {}
-        return {
-            "content_type": resp.headers.get("Content-Type", ""),
-            "text": text,
-        }
-
-
-class KunServeReplicaClient(Protocol):
-    name: str
-    host: str
-
-    async def get_balloon_status(self) -> dict[str, Any]: ...
-
-    async def prepare_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...
-
-    async def warmup_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...
-
-    async def commit_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...
-
-    async def restore_from_balloon(self, payload: dict[str, Any]) -> dict[str, Any]: ...
-
-    async def init_weights_update_group(
-        self, payload: dict[str, Any]
-    ) -> dict[str, Any]: ...
-
-    async def destroy_weights_update_group(self, group_name: str) -> dict[str, Any]: ...
-
-
-def _parse_server_address(server_address: str) -> tuple[str, int]:
-    parts = urlsplit(f"http://{server_address}")
-    if parts.hostname is None or parts.port is None:
-        raise ValueError(f"Invalid server address: {server_address}")
-    return parts.hostname, parts.port
-
-
-def _unwrap_status_response(raw: Any) -> dict[str, Any]:
-    if isinstance(raw, list):
-        if not raw:
-            raise ValueError("Empty balloon status response.")
-        return dict(raw[0])
-    if isinstance(raw, dict):
-        return dict(raw)
-    raise TypeError(f"Unsupported balloon status response type: {type(raw)!r}")
-
-
-def _unwrap_output_list(raw: Any) -> list[dict[str, Any]]:
-    if isinstance(raw, list):
-        return [dict(item) for item in raw]
-    if isinstance(raw, dict):
-        return [dict(raw)]
-    raise TypeError(f"Unsupported RPC response type: {type(raw)!r}")
-
-
-def _response_succeeded(raw: Any) -> bool:
-    outputs = _unwrap_output_list(raw)
-    return bool(outputs) and all(bool(item.get("success")) for item in outputs)
-
-
-def _response_error(raw: Any) -> str:
-    outputs = _unwrap_output_list(raw)
-    for item in outputs:
-        if not item.get("success", False):
-            return str(item.get("message", "unknown error"))
-    return "unknown error"
-
-
-def _status_matches_balloon_target(
-    raw: Any, *, target_variant: str, offload_local_experts: int
-) -> bool:
-    try:
-        status = _unwrap_status_response(raw)
-    except (TypeError, ValueError):
-        return False
-
-    return (
-        str(status.get("state", "")).lower() == "balloon"
-        and str(status.get("runtime_variant", "")).lower()
-        == str(target_variant).lower()
-        and int(status.get("offloaded_local_experts", -1)) == int(offload_local_experts)
-    )
-
-
-def _commit_response_succeeded(
-    raw: Any, *, target_variant: str, offload_local_experts: int
-) -> bool:
-    if _response_succeeded(raw):
-        return True
-
-    outputs = _unwrap_output_list(raw)
-    return bool(outputs) and all(
-        _status_matches_balloon_target(
-            item.get("status"),
-            target_variant=target_variant,
-            offload_local_experts=offload_local_experts,
-        )
-        for item in outputs
-    )
-
-
-def build_complementary_physical_to_logical_map(
-    base_physical_to_logical_map: Sequence[Sequence[int]],
-    *,
-    local_ep_size: int,
-    retained_local_experts: int,
-) -> list[list[int]]:
-    if local_ep_size <= 0:
-        raise ValueError(f"local_ep_size must be positive, got {local_ep_size}")
-    if retained_local_experts <= 0:
-        raise ValueError(
-            f"retained_local_experts must be positive, got {retained_local_experts}"
-        )
-
-    normalized = [list(map(int, row)) for row in base_physical_to_logical_map]
-    if not normalized:
-        raise ValueError("base_physical_to_logical_map must be non-empty")
-
-    num_physical_experts = len(normalized[0])
-    if any(len(row) != num_physical_experts for row in normalized):
-        raise ValueError("All physical_to_logical rows must have the same length.")
-    if num_physical_experts % local_ep_size != 0:
-        raise ValueError(
-            f"num_physical_experts={num_physical_experts} is not divisible by local_ep_size={local_ep_size}"
-        )
-
-    local_chunk = num_physical_experts // local_ep_size
-    if retained_local_experts * 2 != local_chunk:
-        raise ValueError(
-            "Complementary two-replica balloon requires a symmetric half split per local rank."
-        )
-
-    merged: list[list[int]] = []
-    for layer_row in normalized:
-        global_row: list[int] = []
-        for local_rank in range(local_ep_size):
-            base = local_rank * local_chunk
-            global_row.extend(layer_row[base : base + retained_local_experts])
-        for local_rank in range(local_ep_size):
-            base = local_rank * local_chunk + (local_chunk - retained_local_experts)
-            global_row.extend(layer_row[base : base + retained_local_experts])
-        if len(global_row) != num_physical_experts:
-            raise ValueError(
-                f"Expected merged row length {num_physical_experts}, got {len(global_row)}"
-            )
-        merged.append(global_row)
-    return merged
-
-
-@dataclass
-class KunServeLayoutPlan:
-    local_ep_size: int
-    local_routed_experts: int
-    retained_local_experts: int
-    offload_local_experts: int
-    global_world_size: int
-    global_physical_to_logical_map: list[list[int]]
-    replica_active_mappings: list[list[int]]
-
-
-@dataclass
-class KunServeHttpReplicaClient:
-    name: str
-    host: str
-    port: int
-    model_path: str
-    timeout: float = 60.0
-    # warmup_balloon does GLOBAL cuda graph capture which empirically takes
-    # 5-10 minutes on H20 (deepgemm precompile + LL Buffer creation +
-    # 35 batch sizes). The default 60s × 3 attempts (180s) is way too short —
-    # in ab_20260506_172822 it triggered a false-positive "warmup failed"
-    # while the capture was actually mid-flight. Override that one RPC.
-    warmup_timeout: float = 600.0
-    max_attempts: int = 3
-    retry_delay: float = 2.0
-    max_start_wait_time: float = 300.0
-    max_connections: int = 64
-
-    def __post_init__(self) -> None:
-        # Control-plane clients only talk to already-running HTTP servers.
-        # They must not instantiate SGLang ServerArgs here because the controller
-        # runs in a non-GPU Ray actor process where accelerator probing fails.
-        self._base_url = f"http://{self.host}:{self.port}"
-        logger.info(
-            "[KunServeHttpReplicaClient] configured control-plane HTTP client for %s at %s",
-            self.name,
-            self._base_url,
-        )
-        print(
-            f"[KunServeHttpReplicaClient:{self.name}] configured at {self._base_url}",
-            flush=True,
-        )
-
-    @asynccontextmanager
-    async def _get_session(self, timeout: Optional[float] = None):
-        connector = aiohttp.TCPConnector(
-            limit=max(1, self.max_connections),
-            limit_per_host=max(1, self.max_connections // 4),
-            ttl_dns_cache=300,
-            use_dns_cache=True,
-        )
-        effective_timeout = self.timeout if timeout is None else float(timeout)
-        client_timeout = aiohttp.ClientTimeout(total=effective_timeout)
-        session = aiohttp.ClientSession(connector=connector, timeout=client_timeout)
-        try:
-            yield session
-        finally:
-            if not session.closed:
-                await session.close()
-
-    async def _request(
-        self,
-        endpoint: str,
-        payload: Optional[dict[str, Any]] = None,
-        *,
-        method: str = "POST",
-        timeout: Optional[float] = None,
-    ) -> dict[str, Any]:
-        url = f"{self._base_url}/{endpoint}"
-        should_trace = endpoint != "kunserve/status"
-        if should_trace:
-            print(
-                f"[KunServeHttpReplicaClient:{self.name}] {method.upper()} {endpoint} payload={payload or {}}",
-                flush=True,
-            )
-
-        for attempt in range(self.max_attempts):
-            try:
-                async with self._get_session(timeout=timeout) as session:
-                    if method.upper() == "GET":
-                        async with session.get(url) as response:
-                            response.raise_for_status()
-                            result = await _read_async_response(response)
-                            if should_trace:
-                                print(
-                                    f"[KunServeHttpReplicaClient:{self.name}] {endpoint} response={result}",
-                                    flush=True,
-                                )
-                            return result
-                    async with session.post(url, json=payload or {}) as response:
-                        response.raise_for_status()
-                        result = await _read_async_response(response)
-                        if should_trace:
-                            print(
-                                f"[KunServeHttpReplicaClient:{self.name}] {endpoint} response={result}",
-                                flush=True,
-                            )
-                        return result
-            except asyncio.TimeoutError:
-                logger.warning(
-                    "[KunServeHttpReplicaClient] %s %s timed out (%d/%d)",
-                    self.name,
-                    endpoint,
-                    attempt + 1,
-                    self.max_attempts,
-                )
-            except aiohttp.ClientConnectorError:
-                logger.warning(
-                    "[KunServeHttpReplicaClient] %s %s connection error (%d/%d)",
-                    self.name,
-                    endpoint,
-                    attempt + 1,
-                    self.max_attempts,
-                )
-            except aiohttp.ClientResponseError as exc:
-                logger.error(
-                    "[KunServeHttpReplicaClient] %s %s HTTP error: %s",
-                    self.name,
-                    endpoint,
-                    exc,
-                )
-                raise
-            except Exception as exc:
-                logger.error(
-                    "[KunServeHttpReplicaClient] %s %s unexpected error: %s",
-                    self.name,
-                    endpoint,
-                    exc,
-                )
-                if attempt == self.max_attempts - 1:
-                    raise
-
-            if attempt < self.max_attempts - 1:
-                await asyncio.sleep(self.retry_delay * (2**attempt))
-
-        raise RuntimeError(
-            f"[KunServeHttpReplicaClient] {self.name} failed to call {endpoint} "
-            f"after {self.max_attempts} attempts"
-        )
-
-    async def get_balloon_status(self) -> dict[str, Any]:
-        return await self._request("kunserve/status", method="GET")
-
-    async def prepare_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
-        return await self._request("kunserve/prepare_balloon", payload)
-
-    async def warmup_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
-        # warmup_balloon includes GLOBAL cuda graph capture which can take
-        # 5-10 minutes on H20. Use the dedicated warmup_timeout (default 600s)
-        # instead of the per-RPC default (60s) to avoid spuriously aborting
-        # an in-flight capture.
-        return await self._request(
-            "kunserve/warmup_balloon", payload, timeout=self.warmup_timeout
-        )
-
-    async def commit_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
-        # commit_balloon does borrow_tail (per-layer cuMemUnmap) + KV expand
-        # (cuMemMap into the KV region for hundreds of donor segments). On
-        # 4-rank kunserve at 32 offloaded experts × 48 layers this measured
-        # ~67 s end-to-end (see ab_20260429_155927). Reuse warmup_timeout so
-        # we don't trip the 60 s per-RPC default mid-mapping.
-        return await self._request(
-            "kunserve/commit_balloon", payload, timeout=self.warmup_timeout
-        )
-
-    async def restore_from_balloon(self, payload: dict[str, Any]) -> dict[str, Any]:
-        return await self._request("kunserve/restore_from_balloon", payload)
-
-    async def init_weights_update_group(
-        self, payload: dict[str, Any]
-    ) -> dict[str, Any]:
-        return await self._request("init_weights_update_group", payload)
-
-    async def destroy_weights_update_group(self, group_name: str) -> dict[str, Any]:
-        return await self._request(
-            "destroy_weights_update_group", {"group_name": group_name}
-        )
-
-
 class KunServeController:
     def __init__(
         self,
@@ -376,6 +40,8 @@ class KunServeController:
         offload_local_experts: Optional[int] = None,
         group_name: str = "kunserve_global_ep",
         backend: str = "nccl",
+        comm_backend: str = "deepep",
+        capture_policy: str = "auto",
         enable_restore: bool = False,
         pg_init_max_attempts: int = 8,
         pg_init_retry_delay: float = 1.0,
@@ -399,6 +65,10 @@ class KunServeController:
         self._base_group_name = group_name
         self.group_name = group_name
         self.backend = backend
+        self.runtime_backend = KunServeRuntimeBackendConfig(
+            comm_backend=comm_backend,
+            capture_policy=capture_policy,
+        )
         # Disabling RESTORE keeps the control surface minimal for the first
         # end-to-end bring-up. The restore code paths are preserved unchanged
         # so they can be re-enabled later as a follow-up optimization.
@@ -501,13 +171,16 @@ class KunServeController:
         if self._thread is not None and self._thread.is_alive():
             return
         self._emit(
-            "starting: replicas=%d poll_interval=%.2fs min_running=%d group=%s backend=%s enable_restore=%s"
+            "starting: replicas=%d poll_interval=%.2fs min_running=%d group=%s "
+            "backend=%s comm_backend=%s capture_policy=%s enable_restore=%s"
             % (
                 len(self._replicas),
                 self.poll_interval,
                 self.min_running_requests_per_replica,
                 self.group_name,
                 self.backend,
+                self.runtime_backend.comm_backend,
+                self.runtime_backend.capture_policy,
                 self.enable_restore,
             )
         )
@@ -776,6 +449,16 @@ class KunServeController:
         (target_variant, runtime_ep_size, active_local_expert_mapping, etc.),
         so we generate it once and the call site picks the endpoint.
         """
+        effective_capture_cuda_graph = bool(
+            capture_cuda_graph and self.runtime_backend.should_capture_global_graph()
+        )
+        backend_config = {
+            "exchange_mode": self.runtime_backend.exchange_mode,
+            "local_ep_size": plan.local_ep_size,
+            "num_replicas": len(self._replicas),
+            "global_world_size": plan.global_world_size,
+        }
+        pg_names = {"global": self.group_name}
         payloads: list[dict[str, Any]] = []
         for replica_idx in range(len(self._replicas)):
             payloads.append(
@@ -789,7 +472,11 @@ class KunServeController:
                     ],
                     "physical_to_logical_map": plan.global_physical_to_logical_map,
                     "process_group_name": self.group_name,
-                    "capture_cuda_graph": bool(capture_cuda_graph),
+                    "capture_cuda_graph": effective_capture_cuda_graph,
+                    "kunserve_comm_backend": self.runtime_backend.comm_backend,
+                    "capture_policy": self.runtime_backend.capture_policy,
+                    "kunserve_pg_names": pg_names,
+                    "kunserve_backend_config": backend_config,
                 }
             )
         return payloads
@@ -804,19 +491,28 @@ class KunServeController:
         skip the heavy capture step.
         """
         payloads = self._build_layout_payloads(plan, capture_cuda_graph=True)
+        capture_cuda_graph = bool(payloads and payloads[0].get("capture_cuda_graph"))
         self._emit(
-            "warmup balloon: world=%d retained=%d offload=%d (capturing GLOBAL graph in parallel)"
+            "warmup balloon: world=%d retained=%d offload=%d comm_backend=%s "
+            "capture_policy=%s capture_graph=%s"
             % (
                 plan.global_world_size,
                 plan.retained_local_experts,
                 plan.offload_local_experts,
+                self.runtime_backend.comm_backend,
+                self.runtime_backend.capture_policy,
+                capture_cuda_graph,
             )
         )
         logger.warning(
-            "[KUNSERVE-MS] WARMUP dispatch warmup_balloon: world=%d retained=%d offload=%d",
+            "[KUNSERVE-MS] WARMUP dispatch warmup_balloon: world=%d retained=%d "
+            "offload=%d comm_backend=%s capture_policy=%s capture_graph=%s",
             plan.global_world_size,
             plan.retained_local_experts,
             plan.offload_local_experts,
+            self.runtime_backend.comm_backend,
+            self.runtime_backend.capture_policy,
+            capture_cuda_graph,
         )
         results = await asyncio.gather(
             *[
@@ -835,8 +531,9 @@ class KunServeController:
         if errors:
             raise RuntimeError(f"warmup_balloon failed: {'; '.join(errors)}")
         logger.warning(
-            "[KUNSERVE-MS] WARMUP done: replicas=%d (GLOBAL graph cached, state stays LOCAL)",
+            "[KUNSERVE-MS] WARMUP done: replicas=%d capture_graph=%s (state stays LOCAL)",
             len(self._replicas),
+            capture_cuda_graph,
         )
 
     async def enter_balloon(
@@ -855,13 +552,20 @@ class KunServeController:
         # short-circuits via has_captured_variant("global"), so this becomes a fast
         # state-flip + the much smaller commit_balloon work below.
         prepare_payloads = self._build_layout_payloads(plan, capture_cuda_graph=True)
+        prepare_capture_graph = bool(
+            prepare_payloads and prepare_payloads[0].get("capture_cuda_graph")
+        )
 
         self._emit(
-            "preparing balloon: offload=%d retained=%d world=%d payloads=%s"
+            "preparing balloon: offload=%d retained=%d world=%d "
+            "comm_backend=%s capture_policy=%s capture_graph=%s payloads=%s"
             % (
                 plan.offload_local_experts,
                 plan.retained_local_experts,
                 plan.global_world_size,
+                self.runtime_backend.comm_backend,
+                self.runtime_backend.capture_policy,
+                prepare_capture_graph,
                 [
                     {
                         "replica": replica_idx,
@@ -876,10 +580,14 @@ class KunServeController:
         )
 
         logger.warning(
-            "[KUNSERVE-MS] BALLOON dispatch prepare_balloon: offload=%d retained=%d world=%d",
+            "[KUNSERVE-MS] BALLOON dispatch prepare_balloon: offload=%d retained=%d "
+            "world=%d comm_backend=%s capture_policy=%s capture_graph=%s",
             plan.offload_local_experts,
             plan.retained_local_experts,
             plan.global_world_size,
+            self.runtime_backend.comm_backend,
+            self.runtime_backend.capture_policy,
+            prepare_capture_graph,
         )
         prepare_results = await asyncio.gather(
             *[
@@ -1040,75 +748,12 @@ class KunServeController:
     def _ensure_layout_plan(
         self, statuses: Sequence[dict[str, Any]]
     ) -> KunServeLayoutPlan:
-        if self._layout_plan is not None:
-            return self._layout_plan
-
-        local_maps = [
-            status.get("local_physical_to_logical_map") for status in statuses
-        ]
-        if any(local_map is None for local_map in local_maps):
-            raise ValueError("Balloon status is missing local_physical_to_logical_map.")
-        if local_maps[0] != local_maps[1]:
-            raise ValueError(
-                "Replicas do not agree on the baseline physical_to_logical expert layout."
+        if self._layout_plan is None:
+            self._layout_plan = build_layout_plan_from_statuses(
+                statuses,
+                offload_local_experts=self.offload_local_experts,
+                num_replicas=len(self._replicas),
             )
-
-        local_ep_size = int(statuses[0]["local_ep_size"])
-        routed_by_layer = {
-            int(layer_id): int(count)
-            for layer_id, count in statuses[0]["local_routed_experts_per_layer"].items()
-        }
-        routed_values = set(routed_by_layer.values())
-        if len(routed_values) != 1:
-            raise ValueError(
-                "Current KunServe controller requires all MoE layers to expose the same local routed expert count."
-            )
-        local_routed_experts = routed_values.pop()
-
-        offload_local_experts = (
-            self.offload_local_experts
-            if self.offload_local_experts is not None
-            else local_routed_experts // 2
-        )
-        if offload_local_experts <= 0 or offload_local_experts >= local_routed_experts:
-            raise ValueError(
-                f"Invalid offload_local_experts={offload_local_experts} for local_routed_experts={local_routed_experts}"
-            )
-        retained_local_experts = local_routed_experts - offload_local_experts
-        if retained_local_experts != offload_local_experts:
-            raise ValueError(
-                "Current KunServe controller requires a symmetric half split to preserve the original expert count."
-            )
-
-        global_map = build_complementary_physical_to_logical_map(
-            local_maps[0],
-            local_ep_size=local_ep_size,
-            retained_local_experts=retained_local_experts,
-        )
-        replica_active_mappings = [
-            list(range(retained_local_experts)),
-            list(
-                range(
-                    local_routed_experts - retained_local_experts, local_routed_experts
-                )
-            ),
-        ]
-        self._layout_plan = KunServeLayoutPlan(
-            local_ep_size=local_ep_size,
-            local_routed_experts=local_routed_experts,
-            retained_local_experts=retained_local_experts,
-            offload_local_experts=offload_local_experts,
-            global_world_size=local_ep_size * len(self._replicas),
-            global_physical_to_logical_map=global_map,
-            replica_active_mappings=replica_active_mappings,
-        )
-        logger.info(
-            "[KunServeController] layout plan ready: local_ep_size=%d local_routed=%d retained=%d offload=%d",
-            local_ep_size,
-            local_routed_experts,
-            retained_local_experts,
-            offload_local_experts,
-        )
         return self._layout_plan
 
     async def _ensure_process_group(self, plan: KunServeLayoutPlan) -> None:
diff --git a/python/sglang/srt/layers/moe/fused_moe_triton/layer.py b/python/sglang/srt/layers/moe/fused_moe_triton/layer.py
index 1fda07bcf..af64f3c3d 100644
--- a/python/sglang/srt/layers/moe/fused_moe_triton/layer.py
+++ b/python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -454,6 +454,8 @@ class FusedMoE(torch.nn.Module):
             "w2_weight_scale",
             "w13_weight_scale_inv",
             "w2_weight_scale_inv",
+            "w13_input_scale",
+            "w2_input_scale",
         )
         active_tensors: Dict[str, torch.Tensor] = {}
         length = int(active_local_expert_mapping.numel())
diff --git a/python/sglang/srt/layers/moe/token_dispatcher/__init__.py b/python/sglang/srt/layers/moe/token_dispatcher/__init__.py
index 209570073..ff461be7f 100644
--- a/python/sglang/srt/layers/moe/token_dispatcher/__init__.py
+++ b/python/sglang/srt/layers/moe/token_dispatcher/__init__.py
@@ -20,6 +20,9 @@ from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
     FlashinferDispatcher,
     FlashinferDispatchOutput,
 )
+from sglang.srt.layers.moe.token_dispatcher.kunserve_standard import (
+    CrossReplicaStandardDispatcher,
+)
 from sglang.srt.layers.moe.token_dispatcher.fuseep import NpuFuseEPDispatcher
 from sglang.srt.layers.moe.token_dispatcher.mooncake import (
     MooncakeCombineInput,
@@ -48,6 +51,7 @@ __all__ = [
     "DispatchOutputChecker",
     "FlashinferDispatchOutput",
     "FlashinferDispatcher",
+    "CrossReplicaStandardDispatcher",
     "MooncakeCombineInput",
     "MooncakeDispatchOutput",
     "MooncakeEPDispatcher",
diff --git a/python/sglang/srt/layers/quantization/fp8.py b/python/sglang/srt/layers/quantization/fp8.py
index 0a7668d6c..13e0e7773 100644
--- a/python/sglang/srt/layers/quantization/fp8.py
+++ b/python/sglang/srt/layers/quantization/fp8.py
@@ -1535,6 +1535,12 @@ class Fp8MoEMethod(FusedMoEMethodBase):
             )
             return StandardCombineInput(hidden_states=output)
 
+        get_runtime_tensor = getattr(layer, "get_runtime_tensor", None)
+        if get_runtime_tensor is None:
+
+            def get_runtime_tensor(name):
+                return getattr(layer, name)
+
         if self.runner.runner_backend.is_deep_gemm():
 
             get_runtime_tensor = getattr(layer, "get_runtime_tensor", None)
@@ -1628,23 +1634,31 @@ class Fp8MoEMethod(FusedMoEMethodBase):
             )
         elif self.runner.runner_backend.is_triton():
             quant_info = TritonMoeQuantInfo(
-                w13_weight=layer.w13_weight,
-                w2_weight=layer.w2_weight,
-                b13=getattr(layer, "w13_weight_bias", None),
-                b2=getattr(layer, "w2_weight_bias", None),
+                w13_weight=get_runtime_tensor("w13_weight"),
+                w2_weight=get_runtime_tensor("w2_weight"),
+                b13=(
+                    layer.get_runtime_bias("w13_weight_bias")
+                    if hasattr(layer, "get_runtime_bias")
+                    else getattr(layer, "w13_weight_bias", None)
+                ),
+                b2=(
+                    layer.get_runtime_bias("w2_weight_bias")
+                    if hasattr(layer, "get_runtime_bias")
+                    else getattr(layer, "w2_weight_bias", None)
+                ),
                 use_fp8_w8a8=True,
                 w13_scale=(
-                    layer.w13_weight_scale_inv
+                    get_runtime_tensor("w13_weight_scale_inv")
                     if self.block_quant
-                    else layer.w13_weight_scale
+                    else get_runtime_tensor("w13_weight_scale")
                 ),
                 w2_scale=(
-                    layer.w2_weight_scale_inv
+                    get_runtime_tensor("w2_weight_scale_inv")
                     if self.block_quant
-                    else layer.w2_weight_scale
+                    else get_runtime_tensor("w2_weight_scale")
                 ),
-                a13_scale=layer.w13_input_scale,
-                a2_scale=layer.w2_input_scale,
+                a13_scale=get_runtime_tensor("w13_input_scale"),
+                a2_scale=get_runtime_tensor("w2_input_scale"),
                 block_shape=self.quant_config.weight_block_size,
             )
         else:
diff --git a/python/sglang/srt/managers/io_struct.py b/python/sglang/srt/managers/io_struct.py
index 3b0ba545f..e63eb77b8 100644
--- a/python/sglang/srt/managers/io_struct.py
+++ b/python/sglang/srt/managers/io_struct.py
@@ -1586,6 +1586,10 @@ class PrepareBalloonReqInput(BaseReq):
     physical_to_logical_map: Optional[List[List[int]]] = None
     process_group_name: Optional[str] = None
     capture_cuda_graph: bool = True
+    kunserve_comm_backend: str = "deepep"
+    capture_policy: str = "auto"
+    kunserve_pg_names: Optional[Dict[str, str]] = None
+    kunserve_backend_config: Optional[Dict[str, Any]] = None
 
 
 @dataclass
@@ -1610,6 +1614,10 @@ class WarmupBalloonReqInput(BaseReq):
     physical_to_logical_map: Optional[List[List[int]]] = None
     process_group_name: Optional[str] = None
     capture_cuda_graph: bool = True
+    kunserve_comm_backend: str = "deepep"
+    capture_policy: str = "auto"
+    kunserve_pg_names: Optional[Dict[str, str]] = None
+    kunserve_backend_config: Optional[Dict[str, Any]] = None
 
 
 @dataclass
diff --git a/python/sglang/srt/managers/scheduler.py b/python/sglang/srt/managers/scheduler.py
index f844ff109..b2d09ea93 100644
--- a/python/sglang/srt/managers/scheduler.py
+++ b/python/sglang/srt/managers/scheduler.py
@@ -3085,12 +3085,15 @@ class Scheduler(
     def prepare_balloon(self, recv_req: PrepareBalloonReqInput):
         logger.info(
             "[KunServeScheduler] prepare_balloon request: target=%s runtime_ep_size=%s "
-            "runtime_rank_offset=%s dispatch_rank_offset=%s process_group=%s capture_graph=%s",
+            "runtime_rank_offset=%s dispatch_rank_offset=%s process_group=%s "
+            "comm_backend=%s capture_policy=%s capture_graph=%s",
             recv_req.target_variant,
             recv_req.runtime_ep_size,
             recv_req.runtime_rank_offset,
             recv_req.dispatch_rank_offset,
             recv_req.process_group_name,
+            recv_req.kunserve_comm_backend,
+            recv_req.capture_policy,
             recv_req.capture_cuda_graph,
         )
         try:
@@ -3119,12 +3122,15 @@ class Scheduler(
     def warmup_balloon(self, recv_req: WarmupBalloonReqInput):
         logger.info(
             "[KunServeScheduler] warmup_balloon request: target=%s runtime_ep_size=%s "
-            "runtime_rank_offset=%s dispatch_rank_offset=%s process_group=%s capture_graph=%s",
+            "runtime_rank_offset=%s dispatch_rank_offset=%s process_group=%s "
+            "comm_backend=%s capture_policy=%s capture_graph=%s",
             recv_req.target_variant,
             recv_req.runtime_ep_size,
             recv_req.runtime_rank_offset,
             recv_req.dispatch_rank_offset,
             recv_req.process_group_name,
+            recv_req.kunserve_comm_backend,
+            recv_req.capture_policy,
             recv_req.capture_cuda_graph,
         )
         try:
diff --git a/python/sglang/srt/managers/tp_worker.py b/python/sglang/srt/managers/tp_worker.py
index 7dbaf6386..f2f69cbc7 100644
--- a/python/sglang/srt/managers/tp_worker.py
+++ b/python/sglang/srt/managers/tp_worker.py
@@ -197,6 +197,10 @@ class BaseTpWorker(ABC):
             physical_to_logical_map=recv_req.physical_to_logical_map,
             process_group_name=recv_req.process_group_name,
             capture_cuda_graph=recv_req.capture_cuda_graph,
+            kunserve_comm_backend=recv_req.kunserve_comm_backend,
+            capture_policy=recv_req.capture_policy,
+            kunserve_pg_names=recv_req.kunserve_pg_names,
+            kunserve_backend_config=recv_req.kunserve_backend_config,
         )
         self.max_total_num_tokens = self.model_runner.max_total_num_tokens
         return status
@@ -215,6 +219,10 @@ class BaseTpWorker(ABC):
             physical_to_logical_map=recv_req.physical_to_logical_map,
             process_group_name=recv_req.process_group_name,
             capture_cuda_graph=recv_req.capture_cuda_graph,
+            kunserve_comm_backend=recv_req.kunserve_comm_backend,
+            capture_policy=recv_req.capture_policy,
+            kunserve_pg_names=recv_req.kunserve_pg_names,
+            kunserve_backend_config=recv_req.kunserve_backend_config,
         )
         # Warmup does not change max_total_num_tokens, but mirror the pattern
         # used by other balloon entry points so any future internal cache stays
diff --git a/python/sglang/srt/model_executor/model_runner.py b/python/sglang/srt/model_executor/model_runner.py
index 7d4d3a923..36fe61e1d 100644
--- a/python/sglang/srt/model_executor/model_runner.py
+++ b/python/sglang/srt/model_executor/model_runner.py
@@ -706,6 +706,8 @@ class ModelRunner(ModelRunnerKVCacheMixin):
         self._balloon_offloaded_local_experts = 0
         self._balloon_last_error = None
         self._balloon_process_group_name = None
+        self._balloon_kunserve_comm_backend = "deepep"
+        self._balloon_capture_policy = "auto"
         self._balloon_fused_moe_layers: Optional[List[torch.nn.Module]] = None
 
         # KunServe LOCAL/GLOBAL split:
@@ -1092,11 +1094,24 @@ class ModelRunner(ModelRunnerKVCacheMixin):
         active_local_expert_mapping_by_layer: Optional[Dict[int, List[int]]] = None,
         physical_to_logical_map=None,
         process_group_name: Optional[str] = None,
+        kunserve_comm_backend: str = "deepep",
+        capture_policy: str = "auto",
+        kunserve_pg_names: Optional[Dict[str, str]] = None,
+        kunserve_backend_config: Optional[Dict[str, Any]] = None,
     ) -> Dict[int, torch.Tensor]:
         fused_layers = self._iter_fused_moe_layers()
         if not fused_layers:
             raise ValueError("Balloon runtime requires a model with FusedMoE layers.")
 
+        kunserve_comm_backend = str(kunserve_comm_backend or "deepep").lower()
+        capture_policy = str(capture_policy or "auto").lower()
+        kunserve_backend_config = dict(kunserve_backend_config or {})
+        if kunserve_comm_backend not in ("deepep", "sglang"):
+            raise ValueError(
+                "Unsupported KunServe GLOBAL communication backend "
+                f"{kunserve_comm_backend!r}; expected 'deepep' or 'sglang'."
+            )
+
         # KunServe BALLOON publishes a complementary physical_to_logical_map
         # across replicas (e.g. replica 0 retains [0..31, 64..95] while replica
         # 1 retains [32..63, 96..127]). GLOBAL runtime bundles use physical
@@ -1117,34 +1132,41 @@ class ModelRunner(ModelRunnerKVCacheMixin):
                 f"{getattr(self.server_args, 'ep_dispatch_algorithm', None)!r}."
             )
         moe_a2a_backend = get_moe_a2a_backend()
-        if moe_a2a_backend.is_none():
-            raise ValueError(
-                "Balloon GLOBAL bundle requires a cross-rank MoE A2A backend. "
-                "moe_a2a_backend='none' would make the GLOBAL bundle's "
-                "auto-registered dispatcher fall back to StandardDispatcher, "
-                "which only does rank-local expert compute + intra-TP "
-                "all-reduce; it cannot route tokens to experts retained by "
-                "the peer KunServe replica. Pass "
-                "engine_kwargs.sglang.moe_a2a_backend=deepep and "
-                "engine_kwargs.sglang.moe_runner_backend=deep_gemm. NOTE: the "
-                "LOCAL bundle is intentionally overridden back to Standard by "
-                "_force_local_bundle_to_standard_dispatcher (gated on "
-                "SGLANG_EXPERIMENTAL_VMM_MOE_WEIGHTS), so this flag only "
-                "affects the GLOBAL bundle's default."
-            )
-        if (moe_a2a_backend.is_deepep() or moe_a2a_backend.is_mooncake()) and str(
-            getattr(self.server_args, "moe_runner_backend", None)
-        ) != "deep_gemm":
-            raise ValueError(
-                "Balloon GLOBAL bundle with moe_a2a_backend="
-                f"{moe_a2a_backend.value!r} requires "
-                "server_args.moe_runner_backend='deep_gemm'. This sglang build "
-                "only registers DeepEP/Mooncake MoE pre/post permutation paths "
-                "for the deep_gemm runner; leaving the runner as 'auto' or "
-                "'triton' can crash the scheduler during warmup/cuda-graph "
-                "capture. Pass engine_kwargs.sglang.moe_runner_backend=deep_gemm. "
-                f"Current value: {getattr(self.server_args, 'moe_runner_backend', None)!r}."
-            )
+        if kunserve_comm_backend == "deepep":
+            if moe_a2a_backend.is_none():
+                raise ValueError(
+                    "Balloon GLOBAL bundle with kunserve_comm_backend='deepep' "
+                    "requires a cross-rank MoE A2A backend. "
+                    "moe_a2a_backend='none' would make the GLOBAL bundle's "
+                    "auto-registered dispatcher fall back to StandardDispatcher, "
+                    "which only does rank-local expert compute + intra-TP "
+                    "all-reduce; it cannot route tokens to experts retained by "
+                    "the peer KunServe replica. Pass "
+                    "engine_kwargs.sglang.moe_a2a_backend=deepep and "
+                    "engine_kwargs.sglang.moe_runner_backend=deep_gemm, or set "
+                    "kunserve_comm_backend='sglang' to use the new "
+                    "CrossReplicaStandardDispatcher. NOTE: the LOCAL bundle is "
+                    "intentionally overridden back to Standard by "
+                    "_force_local_bundle_to_standard_dispatcher (gated on "
+                    "SGLANG_EXPERIMENTAL_VMM_MOE_WEIGHTS), so this flag only "
+                    "affects the GLOBAL bundle's default."
+                )
+            if (
+                moe_a2a_backend.is_deepep() or moe_a2a_backend.is_mooncake()
+            ) and str(getattr(self.server_args, "moe_runner_backend", None)) != (
+                "deep_gemm"
+            ):
+                raise ValueError(
+                    "Balloon GLOBAL bundle with kunserve_comm_backend='deepep' and "
+                    "moe_a2a_backend="
+                    f"{moe_a2a_backend.value!r} requires "
+                    "server_args.moe_runner_backend='deep_gemm'. This sglang build "
+                    "only registers DeepEP/Mooncake MoE pre/post permutation paths "
+                    "for the deep_gemm runner; leaving the runner as 'auto' or "
+                    "'triton' can crash the scheduler during warmup/cuda-graph "
+                    "capture. Pass engine_kwargs.sglang.moe_runner_backend=deep_gemm. "
+                    f"Current value: {getattr(self.server_args, 'moe_runner_backend', None)!r}."
+                )
 
         active_mappings = self._normalize_balloon_active_mappings(
             retained_local_experts=retained_local_experts,
@@ -1162,6 +1184,11 @@ class ModelRunner(ModelRunnerKVCacheMixin):
                 )
             except Exception:
                 runtime_group_size = "unknown"
+        if kunserve_comm_backend == "sglang" and runtime_group is None:
+            raise ValueError(
+                "KunServe comm backend 'sglang' requires process_group_name to "
+                "resolve to an initialized global process group."
+            )
         self._balloon_global_expert_location_metadata = (
             self._build_balloon_global_metadata(
                 physical_to_logical_map=physical_to_logical_map,
@@ -1201,7 +1228,8 @@ class ModelRunner(ModelRunnerKVCacheMixin):
                     "[KUNSERVE-DBG] register_balloon_global_runtime_bundle: "
                     "tp_rank=%s GLOBAL_dispatch_map shape=%s "
                     "layer0[(0,32,64,96)]=(%d,%d,%d,%d) "
-                    "p2l_layer0[:8]=%s a2a_backend=%s",
+                    "p2l_layer0[:8]=%s a2a_backend=%s kunserve_comm_backend=%s "
+                    "capture_policy=%s",
                     self.tp_rank,
                     tuple(gmap.shape),
                     int(gmap[0, 0].item()),
@@ -1214,11 +1242,15 @@ class ModelRunner(ModelRunnerKVCacheMixin):
                         else "?"
                     ),
                     get_moe_a2a_backend().value,
+                    kunserve_comm_backend,
+                    capture_policy,
                 )
         self._balloon_prepared_active_mappings = {
             layer_id: mapping.clone() for layer_id, mapping in active_mappings.items()
         }
         self._balloon_process_group_name = process_group_name
+        self._balloon_kunserve_comm_backend = kunserve_comm_backend
+        self._balloon_capture_policy = capture_policy
 
         resolved_ep_size = (
             int(runtime_ep_size) if runtime_ep_size is not None else self.moe_ep_size
@@ -1278,6 +1310,48 @@ class ModelRunner(ModelRunnerKVCacheMixin):
                 num_local_experts=int(mapping.numel()),
             )
 
+            explicit_global_dispatcher = None
+            explicit_global_runner = None
+
+            if kunserve_comm_backend == "sglang":
+                from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
+                from sglang.srt.layers.moe.token_dispatcher.kunserve_standard import (
+                    CrossReplicaStandardDispatcher,
+                )
+                from sglang.srt.layers.moe.utils import MoeRunnerBackend
+
+                local_ep_size = int(
+                    kunserve_backend_config.get("local_ep_size") or self.moe_ep_size
+                )
+                explicit_global_dispatcher = CrossReplicaStandardDispatcher(
+                    group=runtime_group,
+                    moe_runner_config=global_runner_config,
+                    local_expert_mapping=dispatcher_local_expert_mapping,
+                    local_ep_size=local_ep_size,
+                    replica_rank=resolved_moe_ep_rank // local_ep_size,
+                    global_rank=resolved_moe_ep_rank,
+                    world_size=resolved_ep_size,
+                )
+                # Use the normal Standard/Triton MoE core for the correctness
+                # backend. This avoids DeepEP/NVSHMEM and DeepGEMM entirely;
+                # the dispatcher itself performs global all-gather + all-reduce.
+                explicit_global_runner = MoeRunner(
+                    MoeRunnerBackend.TRITON,
+                    global_runner_config,
+                )
+                if int(layer.layer_id) == 0:
+                    _kunserve_ms(
+                        "[KUNSERVE-MS] GLOBAL bundle uses CrossReplicaStandardDispatcher: "
+                        "tp_rank=%s global_rank=%s world=%s local_ep_size=%s "
+                        "replica_rank=%s capture_policy=%s backend_config=%s",
+                        self.tp_rank,
+                        resolved_moe_ep_rank,
+                        resolved_ep_size,
+                        local_ep_size,
+                        resolved_moe_ep_rank // local_ep_size,
+                        capture_policy,
+                        kunserve_backend_config,
+                    )
             # When SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL is on (default), build
             # the GLOBAL bundle's dispatcher in DeepEP NORMAL mode explicitly
             # (NOT LL). Reason: this host's container does not expose
@@ -1292,8 +1366,7 @@ class ModelRunner(ModelRunnerKVCacheMixin):
             # init entirely. Trade-off: dynamic-shape dispatch can't be cuda
             # graph captured, so GLOBAL forward runs eager. LOCAL forward
             # (StandardDispatcher) keeps its cuda graph.
-            explicit_global_dispatcher = None
-            if envs.SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL.get():
+            elif envs.SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL.get():
                 from sglang.srt.batch_overlap.two_batch_overlap import (
                     MaybeTboDeepEPDispatcher,
                 )
@@ -1315,6 +1388,7 @@ class ModelRunner(ModelRunnerKVCacheMixin):
                 variant="global",
                 moe_runner_config=global_runner_config,
                 dispatcher=explicit_global_dispatcher,  # None ⇒ default DeepEP via create_moe_dispatcher
+                runner=explicit_global_runner,
                 group=runtime_group,
                 moe_ep_size=resolved_ep_size,
                 moe_ep_rank=resolved_moe_ep_rank,
@@ -1323,14 +1397,17 @@ class ModelRunner(ModelRunnerKVCacheMixin):
                 num_local_experts=int(mapping.numel()),
                 active_local_expert_mapping=mapping,
                 dispatcher_local_expert_mapping=dispatcher_local_expert_mapping,
-                # GLOBAL bundle uses a DeepEP dispatcher whose combine step
-                # already aggregates each token's expert outputs across the
-                # whole cross-replica EP world. Adding the FusedMoE-level
-                # tp_group all-reduce on top would double-reduce within the
-                # local TP group. Companion: LOCAL bundle (StandardDispatcher)
-                # registered in _force_local_bundle_to_standard_dispatcher
-                # passes reduce_results=True because Standard's combine is a
-                # no-op and partial sums need a final tp_group all-reduce.
+                # GLOBAL bundle combine already aggregates each token's expert
+                # outputs across the whole cross-replica EP world:
+                #   - DeepEP does it inside DeepEP combine;
+                #   - CrossReplicaStandardDispatcher does it with a global
+                #     all-reduce then slices back to local tokens.
+                # Adding the FusedMoE-level tp_group all-reduce on top would
+                # double-reduce within the local TP group. Companion: LOCAL
+                # bundle (StandardDispatcher) registered in
+                # _force_local_bundle_to_standard_dispatcher passes
+                # reduce_results=True because Standard's combine is a no-op and
+                # partial sums need a final tp_group all-reduce.
                 reduce_results=False,
             )
         return active_mappings
@@ -1458,6 +1535,10 @@ class ModelRunner(ModelRunnerKVCacheMixin):
         physical_to_logical_map,
         process_group_name: Optional[str],
         capture_cuda_graph: bool,
+        kunserve_comm_backend: str = "deepep",
+        capture_policy: str = "auto",
+        kunserve_pg_names: Optional[Dict[str, str]] = None,
+        kunserve_backend_config: Optional[Dict[str, Any]] = None,
     ) -> List[torch.nn.Module]:
         """Idempotent setup that prepares the global runtime bundle and CUDA graph.
 
@@ -1505,6 +1586,10 @@ class ModelRunner(ModelRunnerKVCacheMixin):
                 active_local_expert_mapping_by_layer=active_local_expert_mapping_by_layer,
                 physical_to_logical_map=physical_to_logical_map,
                 process_group_name=process_group_name,
+                kunserve_comm_backend=kunserve_comm_backend,
+                capture_policy=capture_policy,
+                kunserve_pg_names=kunserve_pg_names,
+                kunserve_backend_config=kunserve_backend_config,
             )
 
         # ensure_cuda_graph_variant_captured short-circuits when the graph for
@@ -1520,15 +1605,21 @@ class ModelRunner(ModelRunnerKVCacheMixin):
             # startup before this code path runs.
             skip_capture = (
                 target_variant == "global"
-                and envs.SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL.get()
+                and (
+                    envs.SGLANG_KUNSERVE_GLOBAL_DEEPEP_NORMAL.get()
+                    or str(kunserve_comm_backend or "deepep").lower() == "sglang"
+                )
             )
             if not skip_capture:
                 self.ensure_cuda_graph_variant_captured(target_variant)
             else:
                 _kunserve_ms(
                     "[KUNSERVE-MS] skip GLOBAL cuda graph capture: "
-                    "GLOBAL bundle is in DeepEP NORMAL mode (dynamic shape, "
-                    "non-capturable). BALLOON forward will run eager.",
+                    "GLOBAL bundle uses a dynamic eager communication path "
+                    "(comm_backend=%s, capture_policy=%s). BALLOON forward "
+                    "will run eager.",
+                    kunserve_comm_backend,
+                    capture_policy,
                 )
 
         return fused_layers
@@ -1548,20 +1639,28 @@ class ModelRunner(ModelRunnerKVCacheMixin):
         physical_to_logical_map=None,
         process_group_name: Optional[str] = None,
         capture_cuda_graph: bool = True,
+        kunserve_comm_backend: str = "deepep",
+        capture_policy: str = "auto",
+        kunserve_pg_names: Optional[Dict[str, str]] = None,
+        kunserve_backend_config: Optional[Dict[str, Any]] = None,
     ) -> Dict[str, Any]:
         target_variant = self._normalize_balloon_variant(target_variant)
         _kunserve_ms(
-            "[KUNSERVE-MS] prepare start: target_variant=%s pg=%s capture_cuda_graph=%s "
+            "[KUNSERVE-MS] prepare start: target_variant=%s pg=%s "
+            "comm_backend=%s capture_policy=%s capture_cuda_graph=%s "
             "runtime_ep_size=%s rank_offset=%s",
             target_variant,
             process_group_name,
+            kunserve_comm_backend,
+            capture_policy,
             capture_cuda_graph,
             runtime_ep_size,
             runtime_rank_offset,
         )
         logger.info(
             "Prepare balloon: target_variant=%s runtime_ep_size=%s moe_ep_rank=%s dispatch_ep_rank=%s "
-            "runtime_rank_offset=%s dispatch_rank_offset=%s process_group=%s capture_cuda_graph=%s",
+            "runtime_rank_offset=%s dispatch_rank_offset=%s process_group=%s "
+            "comm_backend=%s capture_policy=%s capture_cuda_graph=%s",
             target_variant,
             runtime_ep_size,
             moe_ep_rank,
@@ -1569,6 +1668,8 @@ class ModelRunner(ModelRunnerKVCacheMixin):
             runtime_rank_offset,
             dispatch_rank_offset,
             process_group_name,
+            kunserve_comm_backend,
+            capture_policy,
             capture_cuda_graph,
         )
 
@@ -1585,6 +1686,10 @@ class ModelRunner(ModelRunnerKVCacheMixin):
             physical_to_logical_map=physical_to_logical_map,
             process_group_name=process_group_name,
             capture_cuda_graph=capture_cuda_graph,
+            kunserve_comm_backend=kunserve_comm_backend,
+            capture_policy=capture_policy,
+            kunserve_pg_names=kunserve_pg_names,
+            kunserve_backend_config=kunserve_backend_config,
         )
 
         self._balloon_prepared_variant = target_variant
@@ -1613,6 +1718,10 @@ class ModelRunner(ModelRunnerKVCacheMixin):
         physical_to_logical_map=None,
         process_group_name: Optional[str] = None,
         capture_cuda_graph: bool = True,
+        kunserve_comm_backend: str = "deepep",
+        capture_policy: str = "auto",
+        kunserve_pg_names: Optional[Dict[str, str]] = None,
+        kunserve_backend_config: Optional[Dict[str, Any]] = None,
     ) -> Dict[str, Any]:
         """Pre-build the GLOBAL runtime bundle and capture its CUDA graph
         without changing balloon state.
@@ -1637,22 +1746,28 @@ class ModelRunner(ModelRunnerKVCacheMixin):
             return self.get_balloon_status()
 
         _kunserve_ms(
-            "[KUNSERVE-MS] warmup start: target_variant=%s pg=%s capture_cuda_graph=%s "
+            "[KUNSERVE-MS] warmup start: target_variant=%s pg=%s "
+            "comm_backend=%s capture_policy=%s capture_cuda_graph=%s "
             "runtime_ep_size=%s rank_offset=%s",
             target_variant,
             process_group_name,
+            kunserve_comm_backend,
+            capture_policy,
             capture_cuda_graph,
             runtime_ep_size,
             runtime_rank_offset,
         )
         logger.info(
             "Warmup balloon: target_variant=%s runtime_ep_size=%s moe_ep_rank=%s "
-            "runtime_rank_offset=%s process_group=%s capture_cuda_graph=%s",
+            "runtime_rank_offset=%s process_group=%s comm_backend=%s "
+            "capture_policy=%s capture_cuda_graph=%s",
             target_variant,
             runtime_ep_size,
             moe_ep_rank,
             runtime_rank_offset,
             process_group_name,
+            kunserve_comm_backend,
+            capture_policy,
             capture_cuda_graph,
         )
 
@@ -1670,6 +1785,10 @@ class ModelRunner(ModelRunnerKVCacheMixin):
                 physical_to_logical_map=physical_to_logical_map,
                 process_group_name=process_group_name,
                 capture_cuda_graph=capture_cuda_graph,
+                kunserve_comm_backend=kunserve_comm_backend,
+                capture_policy=capture_policy,
+                kunserve_pg_names=kunserve_pg_names,
+                kunserve_backend_config=kunserve_backend_config,
             )
         except Exception as exc:
             self._balloon_last_error = str(exc)
@@ -2109,6 +2228,8 @@ class ModelRunner(ModelRunnerKVCacheMixin):
             "tp_size": int(self.tp_size),
             "local_ep_size": int(self.moe_ep_size),
             "balloon_process_group_name": self._balloon_process_group_name,
+            "kunserve_comm_backend": self._balloon_kunserve_comm_backend,
+            "kunserve_capture_policy": self._balloon_capture_policy,
             "local_num_experts_per_layer": {
                 int(layer.layer_id): int(layer.local_bundle.num_local_experts)
                 for layer in fused_layers
```

### 12.z Verl tracked 修改 diff

```diff
diff --git a/data/compare_kunserve_vs_baseline.sh b/data/compare_kunserve_vs_baseline.sh
index 53992207..ba8eded9 100755
--- a/data/compare_kunserve_vs_baseline.sh
+++ b/data/compare_kunserve_vs_baseline.sh
@@ -42,14 +42,18 @@ export NCCL_DEBUG_SUBSYS=INIT,GRAPH
 : "${MAX_PROMPT_LENGTH:=1024}"
 : "${TRAIN_BATCH_SIZE:=32}"
 : "${GPU_MEMORY_UTILIZATION:=0.65}"
+: "${KUNSERVE_COMM_BACKEND:=sglang}"
+: "${KUNSERVE_ROLLOUT_QUANTIZATION:=none}"
 export CUDA_VISIBLE_DEVICES N_GPUS_PER_NODE MAX_RESPONSE_LENGTH MAX_PROMPT_LENGTH \
-       TRAIN_BATCH_SIZE GPU_MEMORY_UTILIZATION
+       TRAIN_BATCH_SIZE GPU_MEMORY_UTILIZATION KUNSERVE_COMM_BACKEND \
+       KUNSERVE_ROLLOUT_QUANTIZATION
 
 mkdir -p "${OUT_ROOT}"
 echo "[AB] OUT_ROOT=${OUT_ROOT}"
 echo "[AB] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
 echo "[AB] N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
 echo "[AB] MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
+echo "[AB] KUNSERVE_COMM_BACKEND=${KUNSERVE_COMM_BACKEND} KUNSERVE_ROLLOUT_QUANTIZATION=${KUNSERVE_ROLLOUT_QUANTIZATION}"
 
 # Shared keyword filter that captures the kunserve milestone trail in
 # training.log. Updated as new tags are added.
diff --git a/data/compare_tp2_tp4_single_sglang.sh b/data/compare_tp2_tp4_single_sglang.sh
index d9b77e0d..8be86275 100755
--- a/data/compare_tp2_tp4_single_sglang.sh
+++ b/data/compare_tp2_tp4_single_sglang.sh
@@ -16,7 +16,8 @@
 #   OUT_ROOT=/workspace/verl/outputs/tp_rank_xxx
 #   TP2_BATCH=64 TP4_BATCH=128
 #   ROLLOUT_N=8 GPU_MEMORY_UTILIZATION=0.85
-#   MOE_A2A_BACKEND=deepep MOE_RUNNER_BACKEND=auto ROLLOUT_QUANTIZATION=fp8
+#   默认 bf16/no-quant: MOE_A2A_BACKEND=none MOE_RUNNER_BACKEND=triton
+#   如需复测 FP8/DeepEP: MOE_A2A_BACKEND=deepep MOE_RUNNER_BACKEND=auto ROLLOUT_QUANTIZATION=fp8
 # ============================================================
 set -uo pipefail
 
diff --git a/data/train/run_smoke_test_kunserve_tp2_dual_replica.sh b/data/train/run_smoke_test_kunserve_tp2_dual_replica.sh
index a2f9f7cd..e76fd386 100644
--- a/data/train/run_smoke_test_kunserve_tp2_dual_replica.sh
+++ b/data/train/run_smoke_test_kunserve_tp2_dual_replica.sh
@@ -59,6 +59,30 @@ export RAY_local_fs_monitor_interval_ms=${RAY_local_fs_monitor_interval_ms:-3000
 export KUNSERVE_DETAIL_LOG=${KUNSERVE_DETAIL_LOG:-/tmp/kunserve_detail.log}
 export SGLANG_KUNSERVE_OUTPUT_DIR=${SGLANG_KUNSERVE_OUTPUT_DIR:-/workspace/sglang/output}
 export KUNSERVE_MANAGER_OUTPUT_DIR=${KUNSERVE_MANAGER_OUTPUT_DIR:-${SGLANG_KUNSERVE_OUTPUT_DIR}}
+export KUNSERVE_COMM_BACKEND=${KUNSERVE_COMM_BACKEND:-sglang}
+export KUNSERVE_CAPTURE_POLICY=${KUNSERVE_CAPTURE_POLICY:-disabled}
+if [ "${KUNSERVE_COMM_BACKEND}" = "sglang" ]; then
+    # Correctness-first KunServe GLOBAL backend implemented inside sglang:
+    # no DeepEP/NVSHMEM data path, no DeepGEMM GLOBAL runner, and no GLOBAL
+    # cuda graph capture yet.
+    export KUNSERVE_MOE_A2A_BACKEND=${KUNSERVE_MOE_A2A_BACKEND:-none}
+    export KUNSERVE_MOE_RUNNER_BACKEND=${KUNSERVE_MOE_RUNNER_BACKEND:-triton}
+    KUNSERVE_ROLLOUT_QUANTIZATION=${KUNSERVE_ROLLOUT_QUANTIZATION:-none}
+else
+    export KUNSERVE_MOE_A2A_BACKEND=${KUNSERVE_MOE_A2A_BACKEND:-deepep}
+    export KUNSERVE_MOE_RUNNER_BACKEND=${KUNSERVE_MOE_RUNNER_BACKEND:-deep_gemm}
+    KUNSERVE_ROLLOUT_QUANTIZATION=${KUNSERVE_ROLLOUT_QUANTIZATION:-fp8}
+fi
+ROLLOUT_QUANT_ARGS=()
+case "${KUNSERVE_ROLLOUT_QUANTIZATION}" in
+    ""|none|null|None|bf16|bfloat16|false|False|0)
+        # Do not pass rollout.quantization. RolloutConfig defaults to
+        # quantization=None and dtype=bfloat16, i.e. bf16/no-quant inference.
+        ;;
+    *)
+        ROLLOUT_QUANT_ARGS=(actor_rollout_ref.rollout.quantization=${KUNSERVE_ROLLOUT_QUANTIZATION})
+        ;;
+esac
 
 rollout_name=sglang
 project_name=verl_grpo_openr1_math_fsdp_smoke
@@ -66,6 +90,8 @@ exp_name=${EXP_NAME:-Qwen3_30B_A3B_Thinking_2507_fsdp_sglang_kunserve_tp2x2}
 adv_estimator=grpo
 N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
 RAY_DATA_HOME=${RAY_DATA_HOME:-/workspace/verl}
+echo "[kunserve smoke] comm_backend=${KUNSERVE_COMM_BACKEND} capture_policy=${KUNSERVE_CAPTURE_POLICY}"
+echo "[kunserve smoke] moe_a2a_backend=${KUNSERVE_MOE_A2A_BACKEND} moe_runner_backend=${KUNSERVE_MOE_RUNNER_BACKEND} rollout_quantization=${KUNSERVE_ROLLOUT_QUANTIZATION}"
 # 0.0009 (=0.09%) was at noise level; ab_20260507_113241 saw rebalancer fire
 # every poll on diffs like 0.000912/0.001059, aborting R0 work every ~20s.
 # 0.10 = 10% load gap before migrating; cooldown 20 polls × 3s = 60s rest.
@@ -148,7 +174,7 @@ ROLLOUT=(
     actor_rollout_ref.rollout.temperature=1.0
     actor_rollout_ref.rollout.max_model_len=${max_token_len_per_gpu}
     actor_rollout_ref.rollout.disable_log_stats=False
-    actor_rollout_ref.rollout.quantization=${KUNSERVE_ROLLOUT_QUANTIZATION:-fp8}
+    "${ROLLOUT_QUANT_ARGS[@]}"
     actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-4}
     actor_rollout_ref.rollout.prometheus.enable=True
     # Rebalancer is disabled while we validate end-to-end KunServe correctness
@@ -182,6 +208,8 @@ ROLLOUT=(
     +actor_rollout_ref.rollout.kunserve_min_running_requests_per_replica=${KUNSERVE_MIN_RUNNING_REQUESTS_PER_REPLICA:-1}
     +actor_rollout_ref.rollout.kunserve_group_name=${KUNSERVE_GROUP_NAME:-kunserve_global_ep}
     +actor_rollout_ref.rollout.kunserve_backend=${KUNSERVE_BACKEND:-nccl}
+    +actor_rollout_ref.rollout.kunserve_comm_backend=${KUNSERVE_COMM_BACKEND}
+    +actor_rollout_ref.rollout.kunserve_capture_policy=${KUNSERVE_CAPTURE_POLICY}
     +actor_rollout_ref.rollout.kunserve_enable_restore=${KUNSERVE_ENABLE_RESTORE:-False}
     +actor_rollout_ref.rollout.engine_kwargs.sglang.schedule_conservativeness=0.15
     +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=524288
@@ -189,14 +217,14 @@ ROLLOUT=(
     # KunServe BALLOON ships a complementary physical_to_logical_map across the
     # two replicas (replica 0 retains [0..31, 64..95]; replica 1 retains
     # [32..63, 96..127]). GLOBAL bundles therefore need two pieces:
-    # 1. DeepEP moves token hidden states to the peer replica that owns a
-    #    retained physical expert row, then combines the result back.
+    # 1. A KunServe GLOBAL dispatcher moves/replicates token hidden states
+    #    across replicas and combines remote expert contributions back.
+    #    KUNSERVE_COMM_BACKEND=deepep uses DeepEP; =sglang uses the new
+    #    CrossReplicaStandardDispatcher and can run with moe_a2a_backend=none.
     # 2. ep_dispatch_algorithm=static remaps router logical expert ids into the
     #    GLOBAL physical-id dispatch domain after commit_balloon swaps metadata.
-    # With moe_a2a_backend=none, sglang only combines local TP ranks and silently
-    # drops the peer-replica expert contributions, producing degenerate output.
-    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend=${KUNSERVE_MOE_A2A_BACKEND:-deepep}
-    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=${KUNSERVE_MOE_RUNNER_BACKEND:-deep_gemm}
+    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend=${KUNSERVE_MOE_A2A_BACKEND}
+    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=${KUNSERVE_MOE_RUNNER_BACKEND}
     # auto: prefill→normal, decode→low_latency. We use this together with
     # the model_runner's _force_local_bundle_to_standard_dispatcher override
     # (gated on SGLANG_EXPERIMENTAL_VMM_MOE_WEIGHTS=1). LOCAL bundle runs on
diff --git a/data/train/run_smoke_test_single_sglang_tp.sh b/data/train/run_smoke_test_single_sglang_tp.sh
index 402d75bd..5a51c28e 100755
--- a/data/train/run_smoke_test_single_sglang_tp.sh
+++ b/data/train/run_smoke_test_single_sglang_tp.sh
@@ -10,9 +10,8 @@
 #   - rollout.data_parallel_size == 1
 #   => AgentLoopManager 只会创建 1 个 SGLang replica / HTTP server。
 #
-# 默认保持当前 KunServe 相关实验的 local baseline 路径：
-#   quantization=fp8, moe_a2a_backend=deepep, moe_runner_backend=auto,
-#   deepep_mode=auto, ep_dispatch_algorithm=static
+# 默认用于 TP/rank throughput 对比：bf16/no-quant 推理，避免把 FP8/DeepEP
+# backend 的收益或限制混进 TP=2 vs TP=4 的每-rank 对比。
 #
 # 可通过环境变量覆盖：
 #   TP_SIZE=2 TRAIN_BATCH_SIZE=64  bash ...
@@ -55,9 +54,19 @@ rollout_n=${ROLLOUT_N:-8}
 agent_num_workers=${AGENT_NUM_WORKERS:-4}
 max_running_requests=${MAX_RUNNING_REQUESTS:-256}
 
-rollout_quantization=${ROLLOUT_QUANTIZATION:-fp8}
-moe_a2a_backend=${MOE_A2A_BACKEND:-deepep}
-moe_runner_backend=${MOE_RUNNER_BACKEND:-auto}
+rollout_quantization=${ROLLOUT_QUANTIZATION:-none}
+ROLLOUT_QUANT_ARGS=()
+case "${rollout_quantization}" in
+    ""|none|null|None|bf16|bfloat16|false|False|0)
+        # Do not pass rollout.quantization. RolloutConfig defaults to
+        # quantization=None and dtype=bfloat16, i.e. bf16/no-quant inference.
+        ;;
+    *)
+        ROLLOUT_QUANT_ARGS=(actor_rollout_ref.rollout.quantization=${rollout_quantization})
+        ;;
+esac
+moe_a2a_backend=${MOE_A2A_BACKEND:-none}
+moe_runner_backend=${MOE_RUNNER_BACKEND:-triton}
 deepep_mode=${DEEPEP_MODE:-auto}
 ep_dispatch_algorithm=${EP_DISPATCH_ALGORITHM:-static}
 
@@ -123,7 +132,7 @@ ROLLOUT=(
     actor_rollout_ref.rollout.temperature=1.0
     actor_rollout_ref.rollout.max_model_len=${max_token_len_per_gpu}
     actor_rollout_ref.rollout.disable_log_stats=False
-    actor_rollout_ref.rollout.quantization=${rollout_quantization}
+    "${ROLLOUT_QUANT_ARGS[@]}"
     actor_rollout_ref.rollout.agent.num_workers=${agent_num_workers}
     actor_rollout_ref.rollout.prometheus.enable=True
     +actor_rollout_ref.rollout.rebalancer_enable=False
diff --git a/verl/experimental/agent_loop/agent_loop.py b/verl/experimental/agent_loop/agent_loop.py
index a235a4b2..bce2060a 100644
--- a/verl/experimental/agent_loop/agent_loop.py
+++ b/verl/experimental/agent_loop/agent_loop.py
@@ -1236,6 +1236,10 @@ class AgentLoopManager:
             str(OmegaConf.select(cfg, "kunserve_group_name", default="kunserve_global_ep")),
             "--backend",
             str(OmegaConf.select(cfg, "kunserve_backend", default="nccl")),
+            "--comm-backend",
+            str(OmegaConf.select(cfg, "kunserve_comm_backend", default="deepep")),
+            "--capture-policy",
+            str(OmegaConf.select(cfg, "kunserve_capture_policy", default="auto")),
         ]
         offload = OmegaConf.select(cfg, "kunserve_offload_local_experts", default=None)
         if offload is not None:
diff --git a/verl/workers/config/rollout.py b/verl/workers/config/rollout.py
index c6bd21ec..2d6f54fc 100644
--- a/verl/workers/config/rollout.py
+++ b/verl/workers/config/rollout.py
@@ -247,6 +247,13 @@ class RolloutConfig(BaseConfig):
     kunserve_min_running_requests_per_replica: int = 1
     kunserve_group_name: str = "kunserve_global_ep"
     kunserve_backend: str = "nccl"
+    # Data-plane backend used after BALLOON:
+    # - deepep: existing DeepEP GLOBAL dispatcher path
+    # - sglang: correctness-first CrossReplicaStandardDispatcher in sglang
+    kunserve_comm_backend: str = "deepep"
+    # auto keeps existing behavior for deepep. For sglang, the manager resolves
+    # auto to disabled because the initial implementation is eager/dynamic.
+    kunserve_capture_policy: str = "auto"
     kunserve_enable_restore: bool = False
     # When set, override the layout-derived offload count.
     kunserve_offload_local_experts: Optional[int] = None
```
