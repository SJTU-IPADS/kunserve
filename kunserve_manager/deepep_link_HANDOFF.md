# `feat/deepep-comm` 交接说明（DeepEP 跨实例通信）

> 简短交接文档。详细方案见同目录 `deepep_link_implementation_plan.md`。
> 最后更新：2026-06-03

## 1. 这条分支在干什么
用 DeepEP 的 token-routed dispatch/combine 替换当前 dense all-gather，抹掉跨实例通信开销。
**离线能做的(M0 + 基础设施)已完成；M1 起需要真实 H20 + NVSHMEM 环境。**

## 2. 两个仓库各自的 `feat/deepep-comm`

### sglang（`git@github.com:SJTU-IPADS/kunserve.git`）
分支基点：从 `kunserve-4-rl` 的**已提交 HEAD** `60b9a947e` 切出。
| commit | 内容 |
|---|---|
| `c4b38cd18` / `4f548fabe` | 方案文档（最终版：传输直接用 DeepEP，不走 PyTorch NCCL）|
| `23f0b53f0` | **M0**：`kunserve_routing_ref.py` 路由参考 + 5 测试（static-remap/owner 正确性、数值等价 dense）|
| `259c87878` | **PrecisionPolicy**：拆掉 `deepep⇒deep_gemm` 硬 assert → `kunserve_precision.py`（fp8↔deep_gemm / bf16↔triton）|
| `31eac92aa` | `KUNSERVE_DISPATCH_DTYPE` env 读取/校验 + **M4 适配骨架** `kunserve_runner_adapter.py`（grouped↔sorted，未实现）|
| `268e39516` / `b268a184b` | 文档 §11（env 现状 + verl 透传记录）|

新增文件：`python/sglang/srt/model_executor/kunserve_precision.py`、`python/sglang/srt/layers/moe/token_dispatcher/kunserve_routing_ref.py`、`.../kunserve_runner_adapter.py`、`python/sglang/test/kunserve/test_*.py`。
改动文件：`python/sglang/srt/model_executor/model_runner.py`（assert→policy）。

### verl（`git@github.com:imchangyue/verl.git`）
分支基点：从 `main` 的**已提交 HEAD** `266d761d` 切出。
| commit | 内容 |
|---|---|
| `c1a38530` | `KUNSERVE_DISPATCH_DTYPE` 4 点透传到 SGLang scheduler（脚本 export ×2 + `constants_ppo` job env + `async_sglang_server` actor env）|

## 3. worktree（开发位置，不要在主 checkout 上切分支）
```
/workspace/sglang-deepep   -> sglang feat/deepep-comm
/workspace/verl-deepep     -> verl   feat/deepep-comm
```
主 checkout（`/workspace/sglang`=kunserve-4-rl、`/workspace/verl`=main）**留给 codex 做 H20/VMM/NCCL perf，勿动**。

## 4. ⚠️ 合并注意事项（最重要）
两个 `feat/deepep-comm` 都**缺 main / kunserve-4-rl 上"未提交"的修复**作底，因为分支是从已提交 HEAD 切的：
- **verl**：缺 main 上未提交的 **NCCL 网卡 forward**（`constants_ppo` 的 NCCL passthrough、`async_sglang_server` 的 NCCL actor env、脚本的 `NCCL_SOCKET_IFNAME` pin / `attention_backend`）。
- **sglang**：缺 kunserve-4-rl 上未提交的 **VMM 改动**（`cuda_vmm.py` 的 `vmm_map_chunk_bytes` 等，codex 在做）。

**合并策略**：等 codex 把那些修复在各自主分支**提交**后，把 `feat/deepep-comm` rebase/merge 上去：
```bash
# 提交落地后：
git -C /workspace/sglang  checkout feat/deepep-comm && git merge kunserve-4-rl
git -C /workspace/verl     checkout feat/deepep-comm && git merge main
```
- verl 的 `KUNSERVE_DISPATCH_DTYPE` 透传与 NCCL forward **代码结构刻意对齐**（同样的 `for passthrough in (...)` 元组 / `**{k:os.environ[k] for k in (...)}` 推导式）→ 合并就是把变量名**并进同一个列表**，不是真冲突。
- ⚠️ **M1 上机前必须先合 verl 的 NCCL forward**，否则 H20 上 NCCL 会卡（本会话已踩过）。

## 5. 验证
```bash
cd /workspace/sglang-deepep/python
PYTHONPATH=$PWD python3 -m pytest sglang/test/kunserve/ -q      # 19 passed, 1 xfailed(M4占位)
```

## 6. 里程碑进度
- ✅ M0 路由正确性参考+测试
- ✅ PrecisionPolicy（fp8↔deep_gemm / bf16↔triton，向后兼容默认 fp8）
- ✅ `KUNSERVE_DISPATCH_DTYPE` 登记 + verl 透传
- ✅ M4 L3 适配骨架（签名+契约，未实现）
- ⏳ **M1**（点亮 fp8 DeepEP 端到端）—— 需 H20+NVSHMEM，等 codex 环境就绪 + 合并修复后开始
- ⏳ M2 graph capture / M3 量收益 / M4 bf16 适配实现

## 7. 关键约束（实现时勿违反）
- 精度/传输解耦：bf16 是今后一等公民，默认 fp8；expert runner == `moe_runner_backend`（**不**加 `KUNSERVE_EXPERT_RUNNER`）。
- `reduce_results=False`（DeepEP combine 已聚合，勿再 TP all-reduce）。
- static remap 正确性是头号风险（M0 已用 CPU 测试钉住，运行时仍要对拍 dense，temp=0）。
- DeepEP 现有代码**没跑通过**，当脚手架看，逐段验证。
