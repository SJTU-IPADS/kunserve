# Cross-replica Standard-like Dispatcher 详细实现规划（带 2 replica / TP=2 贯穿例子）

> 这版文档把旧版里“global process group / lane subgroup / padded all-gather / runtime bundle / idle keepalive / pre-capture”等抽象概念全部改成一个固定场景来解释：**两个 SGLang 实例，每个实例 TP=EP=2，总共 4 个 GPU rank**。先把这个例子讲透，再给实现规划。

## 实现状态快查（2026-05-24）

| Phase | 状态 | 一句话总结 |
|-------|------|-----------|
| Phase A: manager 重构 | ✅ 已落地 | `BackendStrategy` + `kunserve_pg_names` 协议 |
| Phase B: backend 参数统一 | ✅ 已落地 | `--comm-backend sglang/deepep` |
| Phase C: SGLang eager correctness | ✅ 已落地（2026-05-24 修复 active_mapping override bug） | `CrossReplicaStandardDispatcher` 动态路径，weight 路由 + 数值都用 probe 验证（`KUNSERVE_DISPATCH_PROBE=1`、`KUNSERVE_WEIGHT_PROBE=1`），BALLOON 后输出从乱码变成 coherent 数学推理 |
| Phase D: fixed_padded pre-capture | ✅ 已落地 | `KUNSERVE_CAPTURE_POLICY=fixed_padded` 启用静态 buffer + graph 安全路径 |
| Phase E: idle keepalive | ⏳ **脚手架完成，blocked on latent bug** | 外层控制流 bug + `prepare_for_idle(target_bs>0)` 内层 CUDA illegal access。详见 § 8.4 与 [phase_e_keepalive_dilemma.md](phase_e_keepalive_dilemma.md) |
| Phase F: lane subgroup 优化 | ✅ 已落地 | `lane_group.all_gather + reduce_scatter`，消除 `[A,A,B,B]` 冗余 |
| Phase G: token-level a2a | 🟡 可选未做 | 等 Phase F 收益榨干后再考虑 |

**当前主要阻塞**：BALLOON 模式下副本完成时间不均衡时，先 drain 的副本 hang 在 lane collective（Phase E 未完整实现）。`CrossReplicaStandardDispatcher` 本身数值正确（probe 验证 + 多个 `finish=stop` 自然 EOS 证据）。

**辅助调试工具（本 session 新增）**：
- `KUNSERVE_DISPATCH_PROBE=1` 在 dispatcher 里打 stats（NaN/Inf/mean/absmax + 路由分布），写入 `KUNSERVE_DETAIL_LOG`。
- `KUNSERVE_WEIGHT_PROBE=1` 在 `build_dense_expert_runtime_tensors` 打 weight checksum（narrowed slice vs alt slice），写入同 detail log。**关键调试武器**：用来确认每个 rank 的 weight 实际指向哪些 experts。
- `SGLANG_STREAMING_PROMPT_ANSWER_LOG=<path>` 在 `async_sglang_server.py` 每请求完成时立刻 dump prompt+answer，规避 trainer-level hang 导致看不到任何文本。**核心验证武器**：BALLOON 路径的乱码只在响应尾部出现（开头 pre-balloon 都正常），必须检查尾部 token。

## ⚠️ 历史 bug 与教训（2026-05-24）

`model_runner.py:register_balloon_global_runtime_bundle` 曾有 sglang-only `active_mapping = arange(num_local)` override：强制把所有 rank 的 weight 都 narrow 到 rows 0-31。逻辑出发点是"rows 32-63 是 VMM 虚地址、warmup capture 会触发 `cudaErrorIllegalAddress`"。

实际验证（`KUNSERVE_WEIGHT_PROBE`）结果：
- 4 个 rank 的 rows 32-63 完全 accessible，warmup / prepare / forward 都不 crash。
- override 让 replica 1 错读 rows 0-31（实际存的是镜像于 replica 0 的 lower-half experts，且这些 row 在 balloon 后已被回收为 KV 字节）→ experts 32-63 和 96-127 **永远没被计算** → BALLOON 后期输出退化成 `"so. the. so. seeking."` 死循环乱码。

修复后（2026-05-24）：dispatcher 用 controller 给的原始 mapping（`[0..31]` for replica 0，`[32..63]` for replica 1）。weight checksum 实测：
- rank 0 rows 0-31 sum = 1.472e+06，rank 2 rows 32-63 sum = 1.378e+06（不同 experts）✓
- rank 1 rows 0-31 sum = 1.432e+06，rank 3 rows 32-63 sum = 1.322e+06（不同 experts）✓
- 4 rank 共同覆盖 128 unique experts，符合 [Phase C 设计](#phase-c-sglang-eager-correctness)

BALLOON 后输出从死循环乱码变成 `"Set ratio = 7 ⇒ (8 - k)/k = 7 ⇒ 8 - k = 7k"` 这种真实的代数推理。

**教训**：dispatcher 数值健康（无 NaN/Inf、magnitude 合理）**不等于** 路由正确。错误的 weight↔expert 绑定也会产出"看起来合理但语义错误"的 logits。验证 dispatcher correctness 必须同时检查：
1. weight 物理布局（`KUNSERVE_WEIGHT_PROBE`）
2. dispatcher 数值（`KUNSERVE_DISPATCH_PROBE`）
3. 实际输出文本的 TAIL（不只是开头）

---

## 0. 先固定一个贯穿全篇的具体场景

我们讨论的第一版只考虑这个最小可跑通场景：

```text
replica0 = SGLang 实例 0，占 2 张 GPU
  replica0 local rank0 -> global rank0
  replica0 local rank1 -> global rank1

replica1 = SGLang 实例 1，占 2 张 GPU
  replica1 local rank0 -> global rank2
  replica1 local rank1 -> global rank3
```

也就是：

```text
GLOBAL rank layout:

              local rank0       local rank1
replica0      global rank0      global rank1
replica1      global rank2      global rank3
```

decode 某一步时：

```text
replica0 正在处理自己的请求 batch A
replica1 正在处理自己的请求 batch B

rank0 有 A
rank1 有 A
rank2 有 B
rank3 有 B
```

注意：在一个 SGLang 实例内部，因为 TP/EP 并行的设计，rank0 和 rank1 在 MoE 层入口处通常都能看到同一个 replica 的 hidden states，只是后续各自计算不同 expert 的 partial output。

跨 replica 的困难在于：

```text
rank0/rank1 没有 B
rank2/rank3 没有 A
```

所以，如果 BALLOON 后想让两个实例共享 experts，那么 MoE 层必须先让参与 GLOBAL MoE 的 rank 看到同一批 token：

```text
union batch = A + B
```

这就是 cross-replica Standard-like dispatcher 要做的事。

---

## 1. 先把几个容易绕晕的概念讲清楚

### 1.1 process group 是什么？

`process group` 就是“一组会一起做 collective 通信的 GPU rank”。

例如如果创建一个 global process group：

```text
global_pg = [rank0, rank1, rank2, rank3]
```

那么这 4 个 rank 可以一起做：

```text
all-gather
all-reduce
barrier
```

硬规则：**group 里的所有 rank 必须以相同顺序调用 collective**。如果 rank2 进入 all-reduce，rank0 没进，rank2 就会等到超时或者 hang。

### 1.2 local lane 是什么？

在本文里，`lane` 指“两个 replica 中 local rank 位置相同的一条跨实例通路”。

```text
lane0 = replica0 local rank0 + replica1 local rank0 = [rank0, rank2]
lane1 = replica0 local rank1 + replica1 local rank1 = [rank1, rank3]
```

为什么有 lane？因为：

```text
rank0 和 rank1 都已经有 A
rank2 和 rank3 都已经有 B
```

理论上，rank0 只需要和 rank2 交换 A/B，rank1 只需要和 rank3 交换 A/B。

最理想的通信是：

```text
rank0 <-> rank2 得到 A+B
rank1 <-> rank3 得到 A+B
```

这就是后文说的 `same-local-rank exchange subgroup`。

### 1.3 为什么第一版不直接做 lane subgroup？

因为当前 SGLang 已有的 `/init_weights_update_group` 接口只能创建这种 group：

```text
rank = rank_offset + self.tp_rank
```

manager 对 replica0 发：

```text
rank_offset = 0
```

replica0 的两个 local TP workers 会自动加入：

```text
local tp_rank0 -> global rank0
local tp_rank1 -> global rank1
```

manager 对 replica1 发：

```text
rank_offset = 2
```

replica1 的两个 local TP workers 会自动加入：

```text
local tp_rank0 -> global rank2
local tp_rank1 -> global rank3
```

这个接口天然得到的是：

```text
[rank0, rank1, rank2, rank3]
```

但它不能表达：

```text
只让 local rank0 加入 [rank0, rank2]
只让 local rank1 加入 [rank1, rank3]
```

所以第一版如果直接做 lane subgroup，就要先重做 process group 初始化接口。那会把问题从“验证 dispatcher 正确性”变成“同时调试 subgroup 初始化 + dispatcher + cuda graph + idle keepalive”。风险太大。

因此建议：

```text
P0 correctness:
  只建一个 global process group [0,1,2,3]
  先用这个 group 跑通正确性

P1 performance:
  再新增 lane subgroup [0,2] / [1,3]
  去掉 P0 的重复通信
```

### 1.4 P0 的 global all-gather 为什么有重复？

P0 用 global group 做 all-gather。

输入：

```text
rank0: A
rank1: A
rank2: B
rank3: B
```

四个 rank all-gather 后，每个 rank 都看到：

```text
[rank0 的 A, rank1 的 A, rank2 的 B, rank3 的 B]
 = [A, A, B, B]
```

这确实重复了。但我们可以让每个 rank 只选择自己 lane 上的数据：

```text
rank0 是 lane0 -> 选 rank0 的 A + rank2 的 B
rank2 是 lane0 -> 选 rank0 的 A + rank2 的 B

rank1 是 lane1 -> 选 rank1 的 A + rank3 的 B
rank3 是 lane1 -> 选 rank1 的 A + rank3 的 B
```

由于 rank0/rank1 的 A 在 MoE 输入语义上应该相同，rank2/rank3 的 B 也应该相同，所以两条 lane 得到的都是语义一致的 `A+B`。

代价是：P0 多传了一份 A 和一份 B。它不是最优，但实现简单，非常适合第一版 correctness。

### 1.5 padded all-gather 是什么？为什么要 padded？

两个 replica 的 batch token 数可能不同：

```text
A 有 30 个 token
B 有 45 个 token
hidden_size = H
```

普通 `all_gather_into_tensor` 通常要求每个 rank 输入 shape 一样。CUDA graph 更严格，capture/replay 时 shape 也必须稳定。

所以 P0 做法是先 pad：

```text
A: [30, H] -> pad 到 [45, H]
B: [45, H] -> 已经是 [45, H]
```

同时保存 valid mask：

```text
A 前 30 行有效，后 15 行 dummy
B 前 45 行有效
```

dummy 行设置为：

```text
hidden = 0
topk_weight = 0
topk_id 最终映射成 -1 或者权重为 0
```

这样 dummy token 不会影响最终 MoE 输出。

在 CUDA graph 的 fixed padded 版本里，`max_m` 甚至不取当前 step 的 max，而是固定成 capture batch size，例如 64：

```text
A: [30,H] -> [64,H]
B: [45,H] -> [64,H]
union = [128,H]
```

这会浪费一些计算，但 graph 可以捕获。

### 1.6 GLOBAL all-reduce 为什么能聚合 MoE 输出？

StandardDispatcher 的核心思想是：

```text
每个 rank 只算自己持有的 expert；
每个 rank 得到 partial output；
最后 all-reduce sum 得到完整 output。
```

举例：一个 token 的 top-k experts 分布在四个 global rank 上：

```text
rank0 算 expert 0/1 的贡献 -> partial_0
rank1 算 expert 2/3 的贡献 -> partial_1
rank2 算 expert 4/5 的贡献 -> partial_2
rank3 算 expert 6/7 的贡献 -> partial_3
```

完整 MoE 输出就是：

```text
partial_0 + partial_1 + partial_2 + partial_3
```

所以在 `[rank0, rank1, rank2, rank3]` 上做 all-reduce sum 后，每个 rank 都拿到完整结果。

注意：这里不能用 SGLang 原来的 `tensor_model_parallel_all_reduce()`，因为它固定用 local TP group。我们需要的是跨两个 replica 的 global group。

---

## 2. 当前 SGLang baseline StandardDispatcher 路径

SGLang 原生 baseline 中，如果：

```text
moe_a2a_backend=none
```

MoE 通常走 `StandardDispatcher`。

对应代码：

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
  create_moe_dispatcher(): moe_a2a_backend none -> StandardDispatcher

python/sglang/srt/layers/moe/token_dispatcher/standard.py
  StandardDispatcher.dispatch()
  StandardDispatcher.combine()

python/sglang/srt/layers/moe/fused_moe_triton/layer.py
  FusedMoE.forward_impl()

python/sglang/srt/distributed/communication_op.py
  tensor_model_parallel_all_reduce()
```

baseline 的执行流程是：

```text
1. 每个 EP rank 都拿到同一份 hidden_states。
2. 每个 rank 都跑 router，得到相同 topk_ids/topk_weights。
3. StandardDispatcher 把非本 rank expert 映射成 -1。
4. MoE kernel 自动跳过 -1 expert。
5. 每个 rank 只算本地 expert partial output。
6. FusedMoE.forward_impl() 末尾 all-reduce sum partial output。
```

这条路很适合参考，但不能直接跨 replica 使用。原因还是那句话：

```text
同一个 replica 内 rank0/rank1 都有 A；
但另一个 replica 的 B 不在 rank0/rank1 上。
```

所以 cross-replica Standard-like 的核心就是：

```text
先补上 A/B 的交换，再复用 Standard 的 local filtering + partial sum + all-reduce 思想。
```

---

## 3. 当前 FusedMoE runtime bundle 到底是什么？

### 3.1 用具体例子理解 LOCAL/GLOBAL bundle

把每个 `FusedMoE` 层想成一个可以换挡的模块。

它可以提前存两套运行配置：

```text
LOCAL bundle:
  不使用 KunServe 时的本地配置。

GLOBAL bundle:
  进入 BALLOON 后的跨 replica 配置。
```

### 3.2 LOCAL bundle 里有什么？

在我们的例子里，replica0 的 LOCAL bundle 大概是：

```text
runtime_variant = local
ep_size = 2
rank0/rank1 只在 replica0 内合作
expert weights = replica0 自己加载的本地 experts
dispatcher = 原本 SGLang 的 MoE dispatcher
runner = 原本 runner
reduce_results = 按 baseline 需要决定是否 local TP all-reduce
```

replica1 也有自己的 LOCAL bundle，互相独立。

### 3.3 GLOBAL bundle 里有什么？

BALLOON 后，两个 replica 互补保留 experts。比如每个 local rank 原来有 64 个 experts，BALLOON 后各保留 32 个：

```text
replica0 rank0 保留前半段 experts
replica0 rank1 保留前半段 experts
replica1 rank0 保留后半段 experts
replica1 rank1 保留后半段 experts
```

GLOBAL bundle 里要记录：

```text
runtime_variant = global
global_ep_size = 4
global_ep_rank = 0/1/2/3
dispatcher = DeepEP dispatcher 或 CrossReplicaStandardDispatcher
runner = GLOBAL runner
active_local_expert_mapping = 当前 rank 真正保留的 expert rows
physical_to_logical_map = 两个 replica 合起来后的 expert 物理/逻辑布局
process_group = [rank0,rank1,rank2,rank3]
```

进入 BALLOON 时，不需要重建整个模型，只要让每个 MoE 层：

```python
layer.switch_runtime_bundle("global")
```

恢复时：

```python
layer.switch_runtime_bundle("local")
```

### 3.4 为什么 runtime bundle 对 pre-capture 很重要？

CUDA graph capture 要提前知道 forward 里用的是哪套 dispatcher/runner/weights view。

如果未来 BALLOON 要走 GLOBAL，那么我们可以在真正 BALLOON 前：

```text
1. 先创建 GLOBAL bundle。
2. 暂时切到 GLOBAL bundle。
3. capture GLOBAL cuda graph。
4. capture 完切回 LOCAL。
5. 继续正常推理。
```

后面真正进入 BALLOON 时，GLOBAL graph 已经准备好，不需要在请求已经拥堵时再花几十秒 capture。

这就是本文说的 pre-capture。

---

## 4. 现有 DeepEP KunServe 路径与新 SGLang 路径的关系

我们不要假设之前写的 DeepEP KunServe dispatcher 一定正确，但必须保留它。

因此新的配置应该是：

```text
kunserve_comm_backend=deepep
kunserve_comm_backend=sglang
```

这不是普通 SGLang 的 `moe_a2a_backend`。它只表示 KunServe BALLOON 后，两个 replica 之间的 expert sharing 用哪套数据面。

### 4.1 deepep backend

```text
dispatcher = DeepEP dispatcher
通信语义 = token dispatch/combine
依赖 = DeepEP / NVSHMEM / IBGDA 或相关 fallback
优点 = 理论通信量小，性能上限高
风险 = 依赖复杂，debug 难
```

### 4.2 sglang backend

```text
dispatcher = CrossReplicaStandardDispatcher
通信语义 = dense all-gather + global all-reduce
依赖 = torch distributed / NCCL / SGLang collectives 思路
优点 = 更像 baseline StandardDispatcher，debug 直观，不依赖 DeepEP
风险 = 通信量更大，需要 fixed padding 才容易 capture graph
```

对 verl 来说，两条路接口应该一样。verl 只需要传：

```text
kunserve_enable=True
kunserve_comm_backend=deepep|sglang
kunserve_pre_capture=True|False
kunserve_capture_policy=auto|fixed_padded|disabled
```

剩下的 layout、PG、warmup、prepare、commit 都应该由 `kunserve_manager` 和 SGLang server 内部处理。

---

## 5. CrossReplicaStandardDispatcher 的完整运行流程

下面用一个具体 decode step 讲。

设：

```text
replica0 batch A: M0 = 30 tokens
replica1 batch B: M1 = 45 tokens
hidden size = H
top_k = K
```

### 5.1 MoE 层入口

四个 rank 的输入是：

```text
rank0: hidden_A [30,H], topk_A [30,K]
rank1: hidden_A [30,H], topk_A [30,K]
rank2: hidden_B [45,H], topk_B [45,K]
rank3: hidden_B [45,H], topk_B [45,K]
```

### 5.2 P0 global padded all-gather

第一步 all-gather sizes：

```text
sizes = [30, 30, 45, 45]
max_m = 45
```

然后每个 rank pad 到 `[45,H]`：

```text
rank0: A_pad [45,H]
rank1: A_pad [45,H]
rank2: B_pad [45,H]
rank3: B_pad [45,H]
```

global all-gather 后，每个 rank 都得到：

```text
gather_hidden = [A_pad, A_pad, B_pad, B_pad]
```

### 5.3 每个 rank 选择自己 lane 的 segment

```text
rank0 lane0 -> 选 gather[0] 和 gather[2] -> A_pad + B_pad
rank2 lane0 -> 选 gather[0] 和 gather[2] -> A_pad + B_pad

rank1 lane1 -> 选 gather[1] 和 gather[3] -> A_pad + B_pad
rank3 lane1 -> 选 gather[1] 和 gather[3] -> A_pad + B_pad
```

所以每个 rank 都得到 union batch：

```text
union_hidden = [A_pad, B_pad] = [90,H]
```

其中 A_pad 的后 15 行是 dummy。

### 5.4 expert id remap：只算本 rank 保留的 experts

GLOBAL metadata 已经把 logical expert 映射到了 GLOBAL physical expert id。

dispatcher 再构造一个映射表：

```text
global physical expert id -> local row id or -1
```

例如 rank0 只保留 global physical expert 0..31，那么：

```text
mapping[0..31] = 0..31
mapping[其他] = -1
```

对 union_topk_ids 做映射后：

```text
属于本 rank 的 expert -> local row id
不属于本 rank 的 expert -> -1
padding dummy token -> -1 或 weight=0
```

这样 MoE kernel 只会计算本 rank 保留的 experts。

### 5.5 本地 MoE core 得到 partial output

每个 rank 对同一个 union batch `[90,H]` 计算自己的 partial output：

```text
rank0: partial_0 [90,H]
rank1: partial_1 [90,H]
rank2: partial_2 [90,H]
rank3: partial_3 [90,H]
```

每个 partial 只包含本 rank experts 的贡献。

### 5.6 global all-reduce 聚合完整 MoE output

在 global group `[0,1,2,3]` 上做 sum：

```text
full_union = partial_0 + partial_1 + partial_2 + partial_3
```

all-reduce 后，每个 rank 都得到相同的：

```text
full_union [90,H]
```

### 5.7 slice 回本 replica 的 batch

replica0 只需要 A 的输出：

```text
rank0/rank1 返回 full_union[0:30]
```

replica1 只需要 B 的输出：

```text
rank2/rank3 返回 full_union[45:45+45]
```

中间 A 的 padding 行被丢掉。

### 5.8 一句话总结 dispatcher 数据面

```text
先把 A/B 拼成 union batch；
每个 rank 只算自己保留 expert 的 partial；
用 global all-reduce 拼完整；
最后每个 replica 切回自己的请求。
```

---

## 6. 为什么 P0 correctness 和 P1 performance 要分开？

### 6.1 P0 目标：先证明数学语义正确

P0 只建一个 global group，优点是：

```text
1. 复用当前 /init_weights_update_group。
2. 不新增 subgroup 初始化协议。
3. 所有 rank 都在一个 group，debug 简单。
4. 更容易确认输出是否正确。
```

缺点是：

```text
all-gather 得到 [A,A,B,B]，有重复通信。
```

但 P0 的目标不是最优性能，而是确认：

```text
CrossReplicaStandardDispatcher 的 hidden/topk gather、expert remap、partial compute、global reduce、slice 都是对的。
```

### 6.2 P1 目标：去掉重复 all-gather

P1 新增 lane subgroup：

```text
lane0_pg = [rank0, rank2]
lane1_pg = [rank1, rank3]
```

此时 dispatch all-gather 只在 lane 内做：

```text
rank0/rank2 gather -> [A,B]
rank1/rank3 gather -> [A,B]
```

不再产生 `[A,A,B,B]`。

但为了支持这个，需要新增更通用的 process group 初始化 API，因为当前 `/init_weights_update_group` 没法只让某个 local rank 加入某个 subgroup。

---


## 7. 能不能把 union hidden state 跨层保留下来，避免每层 MoE 都 all-gather？

这是一个很自然的问题。P0 在每个 MoE 层里先把：

```text
rank0/rank1 的 A
rank2/rank3 的 B
```

通过 all-gather 变成：

```text
每个 rank 临时都有 A+B
```

那么看起来似乎可以继续往后传 `A+B`，这样下一层 MoE 入口时就不用再传 hidden states 了。

结论先说清楚：

```text
在“一个 MoE 层内部”，这个想法是对的：
  只要每个 rank 都已经有 union hidden，那么 expert 计算前不需要再做 DeepEP 那种 token dispatch。
  每个 rank 直接从 union hidden 里取 token，算自己保留的 expert partial output 即可。

但在“跨多个 decoder layer 持续保留 union hidden”上，当前 SGLang/Qwen3Moe 代码不适合作为第一版。
  因为 MoE 和下一层 MoE 之间隔着 attention、residual、layernorm、KV cache、request metadata。
  这些状态目前都是按本 replica 的 batch 管理的，不是按 A+B 的全局 batch 管理的。
```

所以本文把两种方案明确区分开：

```text
方案 A：per-MoE temporary union
  每个 MoE 层内部临时 all-gather 出 A+B；
  MoE partial 计算 + global all-reduce 后，立即 slice 回本 replica 的 A 或 B；
  下一层 attention 仍然只看本 replica 的 batch。

方案 B：persistent union hidden
  一旦进入 GLOBAL/BALLOON，hidden_states/residual 在后续层一直保持 A+B；
  下一层 MoE 入口不再 all-gather hidden；
  但 attention/KV/ForwardBatch 也必须跟着全局化。
```

### 7.1 先看当前代码的真实层顺序

以 Qwen3Moe 为例，`Qwen3MoeDecoderLayer.forward()` 的顺序是：

```text
python/sglang/srt/models/qwen3_moe.py

prepare_attn_and_capture_last_layer_outputs()
self_attn(..., forward_batch)
prepare_mlp()
self.mlp(...)     # 这里才进入 MoE / FusedMoE
postprocess_layer()
```

也就是说，每一层不是“MoE 接 MoE”，而是：

```text
attention -> MoE -> 下一层 attention -> 下一层 MoE -> ...
```

同时 `ModelRunner.forward()` 在模型 forward 前会根据 `ForwardBatch` 做 padding / MLP sync / attention scatter 准备：

```text
python/sglang/srt/model_executor/model_runner.py
  forward_batch.prepare_mlp_sync_batch(...)
  或 forward_batch.prepare_attn_tp_scatter_input(...)

python/sglang/srt/model_executor/forward_batch_info.py
  prepare_mlp_sync_batch()
  _pad_inputs_to_size()
```

这些函数不仅处理 hidden shape，还会处理：

```text
input_ids
positions
seq_lens
req_pool_indices
out_cache_loc
extend_* fields
spec decode fields
```

这些字段共同定义了 attention/KV/cache 写入位置和请求状态。

### 7.2 为什么 union hidden 不能直接喂给下一层 attention？

仍然用 2 replica / TP=2 的例子：

```text
replica0: rank0/rank1，真实请求 batch A
replica1: rank2/rank3，真实请求 batch B
```

方案 A 在 MoE 内部得到：

```text
rank0/rank1/rank2/rank3 都临时拿到 full_union = MoE(A+B)
```

如果我们不 slice，而是直接把 `A+B` 传给下一层，那么下一层 attention 在 replica0 上会看到：

```text
hidden_states = A+B
```

但 replica0 的 scheduler / req pool / KV cache 只拥有 A 的请求状态：

```text
replica0 有：
  A 的 req_pool_indices
  A 的 seq_lens
  A 的 positions
  A 的 out_cache_loc
  A 的 KV cache block

replica0 没有：
  B 的请求对象
  B 的 req_pool_indices
  B 的 KV cache 写入位置
  B 的 prefix/cache metadata
```

因此下一层 attention 不知道 B 的 token 应该读写哪个 KV slot，也不知道 B 的 seq_len/position/prefix 信息。强行让 replica0 对 B 做 attention，要么 shape 对不上，要么写错 KV，要么把 dummy/padding 当成真实请求。

反过来，replica1 对 A 也有同样问题。

所以只要 attention 仍然是“每个 SGLang 实例管理自己的请求和 KV cache”，MoE 后就必须：

```text
replica0 slice 回 A
replica1 slice 回 B
```

这样下一层 attention 才是当前代码能正确理解的 local batch 语义。

### 7.3 如果 MoE 后 slice 回 local，下一层 MoE 还能省 all-gather 吗？

不能。

流程会变成：

```text
第 L 层 MoE:
  A/B -> all-gather -> A+B -> MoE -> global all-reduce -> slice 回 A/B

第 L+1 层 attention:
  replica0 只处理 A
  replica1 只处理 B

第 L+1 层 MoE:
  输入又变回：rank0/rank1 只有 A，rank2/rank3 只有 B
  所以还得重新 all-gather 成 A+B
```

注意：也不能缓存“上一层的 A+B”给下一层 MoE 用。因为下一层 MoE 的输入不是上一层 MoE 输出本身，而是经过了下一层 attention、residual、layernorm 之后的新 hidden states：

```text
next_moe_input = post_attention_layernorm(residual + attention(...))
```

这个值每一层都会变化，不能复用上一层的 union buffer。

### 7.4 那 persistent union hidden 要真正成立，需要改什么？

要让方案 B 成立，必须把全链路语义从“两个独立 SGLang 实例”改成“两个实例在 BALLOON 后共同执行一个 global batch”。至少要处理：

```text
1. ForwardBatch 全局化
   每个 rank 的 batch_size / positions / seq_lens / req_pool_indices / out_cache_loc 都要能描述 A+B。

2. KV cache 全局化或镜像化
   replica0 要能对 B 做 attention，就必须能读写 B 的 KV；replica1 也要能读写 A 的 KV。
   这接近请求迁移 / KV cache 合并 / shared KV cache，而不只是 MoE dispatcher。

3. scheduler 请求状态全局化
   stop condition、sampling、regex state、logprob、spec decode、prefix tree 等都要知道全局 batch 中哪些 token 属于本 replica，哪些属于对端 replica。

4. layer communicator / TP scatter-reduce 语义重写
   当前 `LayerCommunicator.prepare_attn()`、`prepare_mlp()`、`postprocess_layer()` 都默认 forward_batch 和 hidden_states shape 是一致的。
   persistent union 会让 hidden_states shape 变成 A+B，但本地 forward_batch 还是 A 或 B，语义会断裂。

5. CUDA graph capture/replay 重新设计
   graph capture 的输入 buffer、ForwardBatch metadata、attention backend metadata 都要固定到 global padded shape。
   这比只在 MoE dispatcher 内部做 fixed padded union 难很多。
```

这已经不再是“实现一个 dispatcher”，而是接近“BALLOON 后临时合并两个 SGLang 实例”。

### 7.5 显存开销怎么估？

先区分两类显存。

#### 方案 A：per-MoE temporary union 的显存

在某个 MoE 层内部，dispatcher 需要临时 buffer：

```text
P0 global gather receive buffer:  global_world_size * padded_m * hidden_size * dtype_bytes
lane 选择后的 union buffer:       num_replicas * padded_m * hidden_size * dtype_bytes
MoE partial / full output buffer:  num_replicas * padded_m * hidden_size * dtype_bytes
```

其中 P0 因为 global group 是 `[rank0,rank1,rank2,rank3]`，会临时收到 `[A,A,B,B]`；P1 lane subgroup 后，dispatch gather buffer 可以降到 `[A,B]`。

这些 buffer 的特点是：

```text
只在 MoE dispatcher / MoE core 附近使用；
MoE combine 后 slice 回 local；
不会要求 attention/KV/request metadata 常驻全局 shape。
```

#### 方案 B：persistent union hidden 的显存

如果 hidden/residual 在整个后续 layer 都保持 A+B，那么不只是 MoE buffer 变大，attention 和 layernorm 的中间张量也会跟着变大：

```text
hidden_states
residual
layernorm output
attention input/output
MLP/MoE input/output
CUDA graph static input/output/workspace
```

粗略公式：

```text
额外 token 维度倍率 ≈ num_replicas
额外单个 hidden-like tensor ≈ (num_replicas - 1) * padded_m * hidden_size * dtype_bytes
```

举例，如果：

```text
num_replicas = 2
padded_m = 64
hidden_size = 2048
bf16/fp16 dtype_bytes = 2
```

那么一个 hidden-like tensor 从：

```text
64 * 2048 * 2 ≈ 256 KiB
```

变成：

```text
128 * 2048 * 2 ≈ 512 KiB
```

单个 tensor 看起来不大，但真实运行中会有多个 live buffer、MoE permute buffer、topk buffer、attention backend workspace、CUDA graph 私有内存池；如果 prefill 或 fixed padded batch 更大，这个开销会被放大。更关键的是，它还要求 KV/request metadata 全局化，工程风险远大于这点 hidden buffer 本身。

### 7.6 通信开销到底省不省？

方案 B 理论上能省掉“后续每个 MoE 层入口的 hidden all-gather”，但只在一个前提下成立：

```text
从上一层 MoE 输出到下一层 MoE 输入之间，hidden 一直保持 global union，并且中间 attention 也能正确处理 global union。
```

当前 Qwen3Moe/SGLang 不满足这个前提。因为 MoE 和下一层 MoE 之间有 local attention/KV/request state。

所以在当前代码语义下：

```text
如果 MoE 后 slice 回 local：
  下一层 MoE 仍然要重新 all-gather，通信没有省。

如果 MoE 后不 slice：
  需要把 attention/KV/scheduler 全部改成 global batch，已经不是 dispatcher 初版。
```

另外，即使 persistent union 成立，也只省 MoE 前的 hidden all-gather；MoE 后的 expert partial 仍然要聚合：

```text
每个 rank 只算自己保留 expert 的 partial output；
完整 MoE output 仍然需要 global all-reduce 或等价 reduce。
```

也就是说，它不会把 MoE 的跨 rank 通信全部消掉，只是把 `gather hidden` 这半边通信挪掉/合并掉。

### 7.7 两个方案的优劣对比

| 方案 | 核心语义 | 通信 | 显存 | CUDA graph | 代码改动 | 适合作为初版吗 |
|---|---|---|---|---|---|---|
| 方案 A：per-MoE temporary union | 只在 MoE dispatcher 内临时把 A/B 拼成 A+B，MoE 后 slice 回本 replica | 每个 MoE 层都要 gather hidden；P0 有 `[A,A,B,B]` 重复，P1 可用 lane subgroup 降低 | 主要是 MoE 内部临时 buffer | fixed padded 后可局部 capture，ForwardBatch 仍保持 local 语义 | 局部：新增 dispatcher + global all-reduce + bundle/payload | **是。正确性最稳** |
| 方案 B：persistent union hidden | BALLOON 后 hidden/residual 跨层一直保持 A+B | 理论上省后续 MoE 前 gather；仍需要 MoE 后 reduce | hidden/residual/attention/graph workspace 都按 global batch 放大 | 需要 global ForwardBatch/attention metadata，graph 输入语义大改 | 全局：attention、KV、scheduler、ForwardBatch、sampling 都要改 | **不建议第一版** |

### 7.8 推荐的初版不变量

为了先把 correctness 跑通，建议第一版明确维护这个不变量：

```text
CrossReplicaStandardDispatcher 可以在 FusedMoE 内部临时制造 union hidden，
但 FusedMoE.forward_impl() 对外的输入和输出 shape 必须仍然是本 replica local batch。
```

换句话说：

```text
进入 self.mlp 前：
  replica0 rank0/rank1 hidden shape = A
  replica1 rank2/rank3 hidden shape = B

CrossReplicaStandardDispatcher.dispatch 内部：
  临时 A/B -> A+B

CrossReplicaStandardDispatcher.combine 内部：
  global all-reduce 得 full A+B
  slice 回 local

离开 self.mlp 后：
  replica0 rank0/rank1 hidden shape = A
  replica1 rank2/rank3 hidden shape = B
```

这个不变量的好处是：

```text
1. 不碰 attention/KV/request pool。
2. 不改变 scheduler 对 batch 的理解。
3. 不改变 Qwen3MoeDecoderLayer 的层间 hidden_states 语义。
4. CUDA graph 只需要让 MoE 内部 exchange fixed padded，而不是把整个模型改成 global batch graph。
5. 后续 P1 lane subgroup 可以直接优化 dispatch all-gather，不推翻 P0 correctness。
```

因此我认为更优秀的第一版路线是：

```text
P0: per-MoE temporary union + global dense all-gather + lane select + global all-reduce + slice-back，先 eager correctness。
P0.5: 同一套 dispatcher 加 fixed padded buffer，做 GLOBAL pre-capture。
P1: 新增 lane subgroup，把 P0 的 [A,A,B,B] dispatch gather 优化成 [A,B]。
P2: 再考虑是否有必要做更激进的 persistent union；只有在准备同时改 ForwardBatch/KV/scheduler/attention 时再启动。
```

## 8. idle keepalive 为什么必须存在？

这是最容易导致“最后一个实例结束、另一个还剩两个请求时 crash/hang”的原因。

### 8.1 问题场景

BALLOON 后：

```text
replica0 的请求先结束了
replica1 还有 2 个长请求
```

如果 replica0 没有真实请求，就不跑 forward；但 replica1 还在 decode，每层 MoE 都会进入：

```text
all-gather
all-reduce
```

collective 需要所有 global ranks 参加：

```text
rank0, rank1, rank2, rank3 都必须进来
```

如果 rank0/rank1 不进来，rank2/rank3 就会卡住。

### 8.2 idle keepalive 的作用

idle keepalive 的意思是：

```text
即使 replica0 没有真实请求，
它也构造一个 dummy / IDLE batch，
继续跑一次 forward，
目的只是陪 replica1 进入同样的 collectives。
```

对应关系：

```text
replica1: 真实 batch B forward
replica0: IDLE batch forward，只参与 GLOBAL MoE collective
```

它不生成真实 token，也不占用真实请求状态；它只是保证 collective participation。

### 8.3 为什么 manager 解决不了这个？

manager 是几秒 poll 一次：

```text
poll /kunserve/status
判断是否 enter/restore balloon
写日志
```

但 MoE collective 是每个 decode step、每一层都发生。manager 不可能每一层去同步两个实例。

所以 idle keepalive 必须在 SGLang scheduler 内部做。当前代码已经有完整脚手架：

```text
scheduler.py: _kunserve_phase_e_active / negotiate_balloon_step_bs /
              _build_balloon_keepalive_batch / _stop_balloon_keepalive
model_runner.py: negotiate_balloon_step_bs (cross-replica all_gather on
                  runtime_group, returns (max, min) of local_bs)
                 _kunserve_keepalive_dummy_kv_slot (在 commit_balloon 预留的
                  KV scratch slot，让 keepalive batch 写 KV 不撞到真实请求)
schedule_batch.py: prepare_for_idle(target_bs > 0) 构造 dummy DECODE-shape batch
model_runner.py: forward_idle() 走标准 forward 路径，触发 lane collective
```

### 8.4 当前实现状态（2026-05-24）

**状态**：⏳ Phase E 部分实现，存在两个 bug 阻断完整落地。

- **外层 bug（控制流）**：`scheduler.py:event_loop_overlap` 里 keepalive 构造
  放在 `elif batch is None:` 分支末尾，**只在 `last_batch` 也为 None 时**才触发；
  转换步（last_batch 是上一个真实 batch 在 result_queue 里，本步队列空）漏过去 →
  此步本副本不发起 lane collective → 对端 lane.all_gather 死锁。`event_loop_normal`
  路径没有这个问题（keepalive 在 run_batch 之前构造）。
- **内层 latent bug（kernel 安全）**：`prepare_for_idle(target_bs > 0)` 把
  `req_pool_indices = zeros(n)`，让所有 dummy token 都指向 req_pool[0]。在没有
  真实请求时（或者真实请求刚被回收），req_pool[0] 对应的 kv 槽位可能不可读，
  attention kernel 读时触发 `cudaErrorIllegalAddress`。这条路径**从未被实际触发过**
  （因为外层 bug 总是先死锁），所以 latent bug 之前一直没暴露。修外层 → 内层立刻炸。

完整诊断与推荐修复方案见 [phase_e_keepalive_dilemma.md](phase_e_keepalive_dilemma.md)。
首选方案：在 `commit_balloon` 预分配一个永久的 "phantom" req_pool entry，把它的
`req_to_token` 整行写成 `dummy_kv_slot`，让 keepalive batch 用
`req_pool_indices = full(n, phantom_idx)` 而不是 zeros。

### 8.5 临时缓解：每请求流式 dump prompt-answer

在 Phase E 完整修好之前，rollout 一旦遇到不均衡负载就会 hang，导致 trainer 走
不到 `exit_after_rollout` 的 prompt-answer.txt 写出步骤——所有已完成请求的文本都
看不到。为了在 hang 也能验证 dispatcher 正确性，`async_sglang_server.py` 现在
在每个请求完成时立刻把 prompt+decoded answer 追加写到
`${SGLANG_STREAMING_PROMPT_ANSWER_LOG}_r${replica}.txt`。`compare_kunserve_vs_baseline.sh`
默认设这个变量为 `${rundir}/prompt_answer_streaming.txt`。即使 run 最终被 SIGTERM，
已完成请求的文本仍然保留在文件里可供检查。

---

## 9. pre-capture 用这个例子怎么理解？

> **状态更新（Phase D + Phase F 已实现）**：
> - **Phase D（fixed_padded GLOBAL graph capture）** 已落地。触发：`KUNSERVE_CAPTURE_POLICY=fixed_padded`。dispatcher 双路径：capture 流走静态 buffer + `all_gather_into_tensor`。
> - **Phase F（lane subgroup 优化）** 已落地。dispatcher 构造函数接受 `lane_group` + `local_tp_group`，dispatch 用 lane all_gather 消除 `[A,A,B,B]` 冗余，combine 用 `lane.reduce_scatter_tensor + local_tp.all_reduce` 替代 global all_reduce。PG 通过 manager 的 `kunserve_pg_names` 传入。
> - 默认仍 `disabled` 不破坏旧 smoke。实现细节见 [kunserve_implementation_detail.md §6.6](kunserve_implementation_detail.md)。
> 本节保留原始动机描述以便后续接手理解为什么需要 pre-capture。

### 9.1 没有 pre-capture 会怎样？

如果等到请求已经因为 KV 不够触发 BALLOON 时才做 GLOBAL graph capture：

```text
1. 发现需要 BALLOON
2. 注册 GLOBAL bundle
3. capture GLOBAL cuda graph
4. commit offload experts
5. 切到 GLOBAL
```

capture 可能要几十秒甚至几分钟。这时请求已经在等待，非常慢。

### 9.2 pre-capture 的做法

在 rollout 开始后、真正 BALLOON 前，manager 预测“这轮可能用 KunServe”，提前做：

```text
1. 创建 global process group [0,1,2,3]
2. 构造未来 BALLOON 的 expert layout
3. 注册 GLOBAL bundle
4. 暂时切到 GLOBAL bundle
5. capture GLOBAL cuda graph
6. 切回 LOCAL bundle
7. 继续正常 LOCAL 推理
```

真正 BALLOON 时：

```text
1. prepare_balloon 发现 GLOBAL bundle/graph 已经 ready
2. commit_balloon 做 VMM expert offload + KV expand
3. switch_runtime_bundle("global")
```

这样 BALLOON entry latency 小很多。

### 9.3 为什么 SGLang Standard-like pre-capture 必须 fixed padded？

CUDA graph capture 要求 shape 稳定。

如果每一步：

```text
A token 数 = 30, 28, 33, ...
B token 数 = 45, 41, 50, ...
```

那么 all-gather 和 union batch shape 都在变，不适合 graph。

所以 `sglang` backend 要 graph capture，就要固定 padding。例如 capture bs=64：

```text
每个 replica 每步都按 [64,H] 参与 MoE exchange
两个 replica union 固定 [128,H]
```

真实 token 少于 64 的部分用 dummy mask 掉。

这就是：

```text
kunserve_capture_policy=fixed_padded
```

### 9.4 当前实现（Phase D, 已落地）

CrossReplicaStandardDispatcher 被改成**双路径**：

```text
dynamic 路径：
  - 保留原 correctness-first 实现
  - 每步 dist.all_gather(list) + torch.full/empty_like + .item() host sync
  - 不能 capture，但 BALLOON 期间任何 shape 都能跑（prefill、不匹配 batch）

static 路径（capture-only）：
  - 构造时按 capture_max_m = graph_runner.max_num_token 预分配持久 buffer
  - dispatch 用 dist.all_gather_into_tensor（不是 list 形式）
  - 用静态 slice copy 做 lane select
  - remap topk_ids 走 torch.clamp + torch.where 无 Python 分支
  - combine 用 dist.all_reduce 直接在静态 buffer 上 in-place
  - 无 .item() / .cpu() / bool() 等 host sync
```

路径选择：进入 `dispatch()` / `combine()` 时通过 `torch.cuda.is_current_stream_capturing()` 判断。capture stream 上必走 static；eager 必走 dynamic；replay 时 Python 不执行，graph 直接重放，问题不存在。

预分配的 buffer（每个 FusedMoE 层一套）：

```text
_buf_padded_hidden[M, H]              # 本 rank all-gather 源
_buf_padded_topk_ids[M, K]            #  ↑
_buf_padded_topk_weights[M, K]        #  ↑
_buf_gathered_hidden[W·M, H]          # all-gather 目标
_buf_gathered_topk_ids[W·M, K]        #  ↑
_buf_gathered_topk_weights[W·M, K]    #  ↑
_buf_union_hidden[NR·M, H]            # lane select 输出 / runner 输入
_buf_union_topk_ids[NR·M, K]          #  ↑
_buf_union_topk_weights[NR·M, K]      #  ↑
_buf_union_topk_ids_remapped[NR·M, K] # remap 后的 topk_ids（独立 buffer）
_neg_one_int32                        # 常量标量，给 torch.where 用
```

其中 `M = capture_max_m`、`W = world_size`（=4，单节点 2 replica × TP2）、`NR = num_replicas`（=2）。

**关键设计决定**：

1. 用单独的 `_buf_union_topk_ids_remapped` 作为 remap 输出，**绝对不能** `self._buf_union_topk_ids = torch.where(...)` 这样重新绑定属性——会把后续 capture 的 buffer 指针污染掉（指向上一次 capture 的 graph 私有 pool）。
2. `_allocate_static_buffers()` 在构造时立即调用 `_mapping_on(device)` 把 `local_expert_mapping` move 到 cuda，避免 capture 内 `.to(device)` 重新分配。
3. 所有 buffer 在 default cuda pool 分配，**不在** graph capture context 里——这样每个 `(variant, batch_size)` 的 graph 共享同一组静态地址。

### 9.5 NCCL communicator 预热（避开 capture 内 init）

NCCL communicator 是 per-group 的，第一次任何 collective 在 `runtime_group` 上跑都会触发 bootstrap。如果这个 bootstrap 发生在 `with torch.cuda.graph(...)` capture context **里**，NCCL 会报"init not allowed in graph mode"或者把 per-launch 元数据烤进 graph 导致 replay 出错。

`model_runner._warmup_balloon_global_runtime` 在进 capture 之前做一次 dummy collective：

```python
# 仅 sglang + fixed_padded 路径触发；deepep 路径自己的 DeepEP buffer setup 已处理
if backend_lower == "sglang" and policy_lower == "fixed_padded":
    preheat_group = self._resolve_balloon_process_group(process_group_name)
    if preheat_group is not None:
        ag_in  = torch.zeros(1, device=cuda)
        ag_out = torch.zeros(world, device=cuda)
        dist.all_gather_into_tensor(ag_out, ag_in, group=preheat_group)
        ar_buf = torch.zeros(1, device=cuda)
        dist.all_reduce(ar_buf, op=SUM, group=preheat_group)
        torch.cuda.synchronize()
        _kunserve_ms("[KUNSERVE-MS] NCCL communicator preheat done ...")
```

两个 op（`all_gather_into_tensor` + `all_reduce`）就够 —— communicator init 是 per-group 不是 per-shape，一旦建好后续任何 shape 都直接复用。`torch.cuda.synchronize()` 确保 bootstrap 完成再进 capture。

失败被 wrap 在 try/except 里：如果 communicator 已经从别的路径（manager 的 `init_weights_update_group`、之前的 warmup forward）建好了，preheat 会成功；如果真没建好且 preheat 也失败，capture 会自己报错，那时这条 milestone 是定位首要线索。

### 9.6 形状不变量与 idle keepalive 的耦合

static 路径有一条**硬约束**：capture 和 replay 时所有 global rank 必须看到同一个 `local_m`。

- capture 阶段：`CudaGraphRunner` 按 `capture_bs` 同步遍历，两个 replica 在同一时刻执行 warmup_balloon RPC，都进 `_capture_one_stream("global")`，每个 bs 同时 capture。manager 已经用 `asyncio.gather` 并行调用两边的 `warmup_balloon`，这一步天然 lockstep。
- replay 阶段：BALLOON forward 时如果两个 replica `forward_batch.batch_size` 不一致（例如 replicaA decode 64 个请求，replicaB 已经 drain 完），`all_gather_into_tensor` 会因为 input shape 不匹配 hang/报错。

**Phase D 没有解决 replay-time 的 lockstep 问题**——这是 §8 idle keepalive 的范围。Phase D 跑通后，必须配套实现 Phase E（idle keepalive 正式化），否则长尾不均衡场景会 NCCL hang。短期 smoke 测试两 replica 工作量大致相同时不触发。

---

## 10. manager 架构应该怎么重构？

当前 `controller.py` 太大，因为它同时做：

```text
HTTP client
layout plan
process group init
warmup/prepare/commit 状态机
backend payload 构造
metrics log
asymmetric drain log
```

下一步加 `deepep|sglang` 后，如果继续塞在一个文件，会很难维护。

建议拆成：

```text
kunserve_manager/
  cli.py                    # argparse/env，只负责启动
  config.py                 # ManagerConfig / enum
  models.py                 # LayoutPlan / PGSpec / status dataclass
  http_client.py            # HTTP RPC client + retry + response unwrap
  layout.py                 # complementary expert layout 计算
  process_groups.py         # PG init/destroy 编排
  metrics.py                # bw_throughput.jsonl / lifecycle log
  controller.py             # 只保留状态机
  backends/
    base.py                 # BackendStrategy 协议
    deepep.py               # DeepEP backend payload 和策略
    sglang_standard.py      # CrossReplicaStandard backend payload 和策略
```

### 10.1 controller 应该只像这样工作

```text
start():
  statuses = 等两个 replica ready
  plan = layout.build(statuses)
  strategy = backend_factory(comm_backend)
  pg_registry = process_groups.ensure(strategy.required_groups(plan))
  if pre_capture:
      strategy.warmup(plan, pg_registry)
  start polling loop

tick():
  statuses = fetch_statuses()
  metrics.write(statuses)
  if should_enter_balloon(statuses):
      strategy.prepare(plan, pg_registry)
      strategy.commit(plan)
  if should_restore(statuses):
      strategy.restore(plan)
```

这样 controller 不需要知道 DeepEP 怎么 dispatch，也不需要知道 SGLang backend 是 P0 global dense 还是 P1 lane subgroup。

### 10.2 BackendStrategy 负责差异

```python
class KunServeBackendStrategy:
    def required_groups(plan): ...
    def build_warmup_payloads(plan, pg_registry): ...
    def build_prepare_payloads(plan, pg_registry): ...
    def build_commit_payloads(plan): ...
```

DeepEPStrategy payload 里：

```text
kunserve_comm_backend=deepep
process_group_name=global_pg
capture_policy=auto
```

SGLangStandardStrategy payload 里：

```text
kunserve_comm_backend=sglang
process_group_name=global_pg
kunserve_backend_config.exchange_mode=global_dense_v0
capture_policy=fixed_padded
```

---

## 11. SGLang server 侧代码改动方案

### 11.1 扩展 RPC 输入结构

修改：

```text
python/sglang/srt/managers/io_struct.py
```

给 `PrepareBalloonReqInput` 和 `WarmupBalloonReqInput` 增加：

```python
kunserve_comm_backend: str = "deepep"      # deepep | sglang
capture_policy: str = "auto"              # auto | fixed_padded | disabled
kunserve_pg_names: Optional[Dict[str, str]] = None
kunserve_backend_config: Optional[Dict[str, Any]] = None
```

### 11.2 拆分 ModelRunner 的 GLOBAL bundle 注册

当前：

```text
register_balloon_global_runtime_bundle()
```

同时包含 DeepEP guard、metadata、dispatcher、runner、bundle 注册。建议拆成：

```python
def register_balloon_global_runtime_bundle(..., kunserve_comm_backend="deepep", ...):
    common = self._build_balloon_runtime_common(...)
    if kunserve_comm_backend == "deepep":
        return self._register_balloon_global_runtime_bundle_deepep(common, ...)
    if kunserve_comm_backend == "sglang":
        return self._register_balloon_global_runtime_bundle_sglang(common, ...)
```

共享 common 部分：

```text
1. 校验 ep_dispatch_algorithm='static'
2. normalize active_local_expert_mapping
3. resolve process group
4. build GLOBAL expert location metadata
5. build dispatcher_local_expert_mapping
6. build global_runner_config
```

DeepEP 分支保留现在逻辑。

SGLang 分支新增：

```text
CrossReplicaStandardDispatcher + Triton runner + reduce_results=False
```

### 11.3 新增 CrossReplicaStandardDispatcher

新增文件：

```text
python/sglang/srt/layers/moe/token_dispatcher/kunserve_standard.py
```

它返回 `StandardDispatchOutput`，这样可以复用已有 Triton Standard runner。

伪代码：

```python
class CrossReplicaStandardDispatcher(BaseDispatcher):
    def dispatch(self, hidden_states, topk_output):
        sizes = all_gather(local_num_tokens)
        hidden_pad, topk_pad, weight_pad, mask = pad_to_max_or_fixed(...)
        gathered = global_all_gather(...)
        union = select_same_lane_segments(gathered)
        remapped_topk = self.local_expert_mapping[union_topk]
        remapped_topk[invalid_mask] = -1
        save_state(local_m, max_m, replica_idx)
        return StandardDispatchOutput(union_hidden, None, union_topk_output)

    def combine(self, combine_input):
        partial_union = combine_input.hidden_states
        full_union = global_all_reduce_sum(partial_union)
        return slice_back_to_local_replica(full_union)
```

第一版 runner 建议用 Triton，不直接上 DeepGEMM。原因是当前代码里已经有 Standard -> DeepGEMM permute，但之前 LOCAL Standard + DeepGEMM 出现过 corruption 风险；先 correctness，后性能。

---

## 12. verl 侧最小改动

verl 不应该知道内部是 DeepEP 还是 Standard-like。

只加配置字段：

```python
kunserve_comm_backend: str = "deepep"
kunserve_pre_capture: bool = True
kunserve_capture_policy: str = "auto"
```

然后在启动 manager 时转成 CLI：

```text
--comm-backend deepep|sglang
--pre-capture / --no-pre-capture
--capture-policy auto|fixed_padded|disabled
```

不要把 layout、process group、dispatcher 细节放回 verl。

---

## 13. 分阶段实施计划

### Phase A：先重构 manager，不改变 DeepEP 行为

目标：把大 controller 拆层，但默认 `comm_backend=deepep` 的行为不变。

验收：

```text
python -m kunserve_manager --help
现有 deepep smoke 仍能启动 manager
```

### Phase B：加统一 backend 参数

目标：

```text
kunserve_comm_backend=deepep
```

走原逻辑；payload/status 里能看到 backend 字段。

### Phase C：实现 SGLang Standard-like eager correctness

参数：

```text
kunserve_comm_backend=sglang
kunserve_capture_policy=disabled
```

先不 capture graph，只跑 eager。

验收：

```text
能进入 BALLOON
不 NaN
不 device assert
不 collective hang
输出 stop reason 大多正常
```

### Phase D：实现 fixed padded pre-capture【已完成】

参数：

```text
kunserve_comm_backend=sglang
kunserve_pre_capture=True
kunserve_capture_policy=fixed_padded
```

**实现状态**：
- ✅ `kunserve_standard.py` 增加 static 路径 + 静态 buffer 预分配 + `all_gather_into_tensor` + 无 host sync
- ✅ `runtime_config.py` 的 `should_capture_global_graph()` 允许 sglang+fixed_padded
- ✅ `model_runner.register_balloon_global_runtime_bundle` 传 `capture_max_m=graph_runner.max_num_token`
- ✅ `_warmup_balloon_global_runtime` 的 `skip_capture` 按 backend 分支
- ✅ NCCL communicator 预热（dummy all_gather_into_tensor + all_reduce）

**剩余风险**：replay 时两 replica `local_m` 必须一致，依赖 Phase E。

验收命令：

```bash
RUN=/workspace/verl/outputs/<ts>/kunserve
grep -aE "CrossReplicaStandardDispatcher static buffers ready" \
     "${RUN}"/kunserve_*.log "${RUN}"/verl_training.log
grep -aE "NCCL communicator preheat done" \
     "${RUN}"/kunserve_*.log
grep -aE "Capturing batches \(variant='?global'?" \
     "${RUN}"/verl_training.log
grep -aE "\[KUNSERVE-MS\] skip GLOBAL cuda graph capture" \
     "${RUN}"/kunserve_*.log   # 必须为空
jq '.internal_states[] | .balloon_status.captured_graph_variants' \
     "${RUN}"/sglang_snapshot/server_info_*.json
# 期望进 BALLOON 后包含 ["local","global"]
```

### Phase E：正式化 idle keepalive【脚手架已就绪，blocked on latent bug】

构造不均衡请求：

```text
replica0 先 drain
replica1 还剩长请求
```

验收：

```text
replica0 balloon_keepalive_steps 持续增长
replica1 不 hang
最后请求能正常结束
```

**当前状态（2026-05-24）**：脚手架已实现（`_kunserve_phase_e_active`、
`negotiate_balloon_step_bs`、`_build_balloon_keepalive_batch`、
`_kunserve_keepalive_dummy_kv_slot` 都已落地），但 `event_loop_overlap` 里 keepalive
构造时机有外层 bug、`prepare_for_idle(target_bs > 0)` 里的 attention 读路径有
未被触发过的内层 latent bug。两者都必须修才能完成 Phase E。详细诊断见
[phase_e_keepalive_dilemma.md](phase_e_keepalive_dilemma.md)。

推荐修复路径（按风险从低到高）：
1. **phantom req_pool entry**：在 `commit_balloon` 预分配 1 个永久 req_pool 槽位，
   将其 `req_to_token` 整行填 `dummy_kv_slot`；keepalive batch 用 `req_pool_indices = phantom_idx`
   而不是 zeros。修内层 latent bug。
2. **早绑定 keepalive**：把 `event_loop_overlap` 里的 keepalive 构造从底部 `elif` 提到
   Phase E 协商之后、`run_batch` 之前；修外层控制流 bug。需要同时做 (1) 否则会暴露 latent bug。
3. **不均衡负载验证**：跑 `MAX_RESPONSE_LENGTH=10000 TRAIN_BATCH_SIZE=2` 这种容易让
   一个副本先 drain 的配置，确认 `start balloon keepalive` 日志出现且 lane collective
   持续匹配。

**当前临时缓解**：`async_sglang_server.py` 在请求完成时立刻流式 dump prompt+answer 到
`${SGLANG_STREAMING_PROMPT_ANSWER_LOG}_r${replica}.txt`（详见 § 8.5）。Phase E hang
即使发生，已完成请求的文本仍可保留供检查。

### Phase F：实现 lane subgroup 优化【已完成】

**实现状态**：✅ 已落地。dispatcher 构造函数接受 `lane_group` + `local_tp_group`。dispatch 用 `lane_group.all_gather_into_tensor` 直接得到 `[A, B]`，无 `[A,A,B,B]` 冗余。combine 用 `lane_group.reduce_scatter_tensor + local_tp_group.all_reduce` 替代 global all_reduce，每个 rank 只收 replica 维度的切片。lane group PG 通过 manager 的 `kunserve_pg_names`（`lane_0`/`lane_1`）传入，由 `init_weights_update_group` 的 `lane_only_tp_rank` 模式创建。

Phase F 同时优化 dispatch 和 combine 两侧：
- **dispatch 侧**：消除 `[A,A,B,B]` 网络冗余。compared to global all-gather，dispatch 阶段字节数从 `world * M * H` 降到 `num_replicas * M * H`
- **combine 侧**：`reduce_scatter_tensor`（lane subgroup 内 reduce + 按 replica 切分）+ `local_tp_group.all_reduce`（合并两条 lane 贡献）。等价于 all_reduce 但网络字节数 ≈ 减半

回退策略：当 `lane_group` 或 `local_tp_group` 为 None，dispatcher 透明回退到 Phase D global group 路径。

### Phase G（可选）：token-level all-to-all（DeepEP 思路，不依赖 DeepEP 实现）

进一步用 "token 只发到拥有它 top_k expert 的 rank、算完结果送回 origin replica" 的精确路由替代 dense all-gather + all-reduce。通信量 O(M·top_k·H) 而不是 O(M·world_size·H)。

这一阶段在数学上和 DeepEP 等价，但用 `torch.distributed` 原语而不是 DeepEP/NVSHMEM 实现。优势是可移植性（不依赖 IB、NVSHMEM、IBGDA）；劣势是工程复杂度高，需要 token 分桶 + 静态 padding + 反向路由表。**只有 Phase F 的 reduce_scatter 优化收益榨干之后才考虑做**。

---

## 14. 最终推荐参数

### verl Hydra

```text
actor_rollout_ref.rollout.kunserve_enable=True
actor_rollout_ref.rollout.kunserve_comm_backend=sglang
actor_rollout_ref.rollout.kunserve_pre_capture=True
actor_rollout_ref.rollout.kunserve_capture_policy=fixed_padded   # Phase D 已支持
```

或者通过 smoke 脚本环境变量：

```bash
export KUNSERVE_COMM_BACKEND=sglang
export KUNSERVE_CAPTURE_POLICY=fixed_padded   # 默认 disabled
```

### manager CLI

```bash
python -m kunserve_manager \
  --replica 198.18.0.1:PORT0 \
  --replica 198.18.0.1:PORT1 \
  --model-path /workspace/verl/models/... \
  --comm-backend sglang \
  --pre-capture \
  --capture-policy fixed_padded \
  --group-name kunserve_global_ep \
  --backend nccl
```

### SGLang warmup/prepare payload 示例

```json
{
  "target_variant": "global",
  "runtime_ep_size": 4,
  "runtime_rank_offset": 0,
  "dispatch_rank_offset": 0,
  "active_local_expert_mapping": [0, 1, 2, 3],
  "physical_to_logical_map": [[...]],
  "process_group_name": "kunserve_global_ep_v...",
  "capture_cuda_graph": true,
  "kunserve_comm_backend": "sglang",
  "capture_policy": "fixed_padded",
  "kunserve_backend_config": {
    "num_replicas": 2,
    "local_ep_size": 2,
    "replica_idx": 0,
    "exchange_mode": "global_dense_v0"
  }
}
```

---

## 15. 最重要的原则

1. **DeepEP 通路保留，但不能当作新 SGLang backend 的 correctness oracle。** 新 backend 要用数学 reference 验证：dense Standard partial sum + global all-reduce。
2. **不要污染 baseline StandardDispatcher。** 新逻辑应该是 KunServe-only 的 `CrossReplicaStandardDispatcher`。
3. **P0 先只建一个 global PG。** 这样可以复用当前 `/init_weights_update_group`，先把 correctness 跑通。
4. **P1 再做 lane subgroup。** lane subgroup 是性能优化，不是第一版 correctness 必需项。
5. **GLOBAL all-reduce 必须用 KunServe global group。** 不能误用 local `get_tp_group()`。
6. **pre-capture 必须 fixed padded。** dynamic all-gather 可以 eager 跑，但不能承诺 CUDA graph。**Phase D 已实现这条**。
7. **idle keepalive 是跨 replica collective 的生存条件。** 只要一个 replica 还在 BALLOON decode，其他 replica 即使没真实请求也必须继续参与 GLOBAL collectives。**Phase D 的 graph replay 同样依赖这条**——两 replica 的 `local_m` 必须 lockstep，否则 `all_gather_into_tensor` 会 hang。
8. **初版不要做 persistent union hidden。** MoE 内部可以临时 union，但离开 `FusedMoE.forward_impl()` 时必须 slice 回 local batch；否则会牵连 attention/KV/ForwardBatch/scheduler，变成”合并实例”级别改造。
9. **静态 buffer 的属性指针不能在 dispatch 内部被重新绑定。** capture 之间共享同一组持久 buffer 地址；`self._buf_X = torch.where(...)` 这样的写法会把后续 capture 的 record 钉死到上一次 capture 的 graph 私有 pool。所有 in-place 修改必须走 `copy_` / `.zero_()` / `.fill_()` / `dist.*_into_tensor`。
10. **NCCL communicator 必须在 capture 之外完成 bootstrap。** 进入 `with torch.cuda.graph(...)` 之前对目标 PG 跑一次 dummy `all_gather_into_tensor + all_reduce`；否则 NCCL 在 capture 里 init 会失败或把 per-launch 元数据烤进 graph。
