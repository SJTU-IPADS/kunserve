# kunserve_manager

Standalone control plane for a group of KunServe-enabled `sglang` replicas.

## Why a separate package?

KunServe needs **one** process that:

1. Polls `/kunserve/status` on every replica in the group.
2. Decides when to enter / leave BALLOON state (KV expansion via expert
   weight donation across replicas).
3. Dispatches `/kunserve/{warmup,prepare,commit,restore}_balloon` RPCs to
   the replicas in the right order, using their HTTP endpoints.

It deliberately **does not** live inside the sglang inference processes.
There is no leader election, no "every replica also tries to start a
manager and one wins via file lock". The deployer (a verl training
script, a manual CLI invocation, a Kubernetes sidecar, ...) is
responsible for running exactly one manager per group and pointing it at
the replicas.

## Install

From this directory:

```bash
pip install -e .
```

Editable install drops two things on `$PATH`:
- the importable `kunserve_manager` package (for tests / programmatic use)
- a `kunserve-manager` CLI shim

If you don't want to install, you can also run it directly:

```bash
PYTHONPATH=/path/to/sglang/kunserve_manager python -m kunserve_manager ...
```

## Run (standalone)

```bash
python -m kunserve_manager \
    --replica 198.18.0.1:43747 \
    --replica 198.18.0.1:35403 \
    --model-path /workspace/Qwen3-30B-A3B-Thinking-2507 \
    --poll-interval 2.0 \
    --group-name kunserve_global_ep \
    --log-level INFO
```

`--replica` is repeatable; you can also pass them via env:

```bash
KUNSERVE_MANAGER_REPLICAS=198.18.0.1:43747,198.18.0.1:35403 \
KUNSERVE_MANAGER_MODEL_PATH=/path/to/model \
python -m kunserve_manager
```

The manager runs until it receives `SIGINT` / `SIGTERM`, at which point
it tears down BALLOON state on all replicas and exits.

### Required arguments

| flag                                 | env                                          | required | notes |
|--------------------------------------|----------------------------------------------|----------|-------|
| `--replica HOST:PORT`                | `KUNSERVE_MANAGER_REPLICAS=a:p,b:p`          | yes      | exactly 2 entries |
| `--model-path PATH`                  | `KUNSERVE_MANAGER_MODEL_PATH=PATH`           | yes      | used to derive expert layout |

### Optional knobs

| flag                                       | default        | env                                              |
|--------------------------------------------|----------------|--------------------------------------------------|
| `--poll-interval SEC`                      | 2.0            | `KUNSERVE_MANAGER_POLL_INTERVAL`                 |
| `--min-running-requests-per-replica N`     | 1              | `KUNSERVE_MANAGER_MIN_RUNNING_REQUESTS_PER_REPLICA` |
| `--offload-local-experts N`                | layout-derived | `KUNSERVE_MANAGER_OFFLOAD_LOCAL_EXPERTS`         |
| `--group-name NAME`                        | kunserve_global_ep | `KUNSERVE_MANAGER_GROUP_NAME`                |
| `--backend nccl|gloo|...`                  | nccl           | `KUNSERVE_MANAGER_BACKEND`                       |
| `--enable-restore`                         | off            | `KUNSERVE_MANAGER_ENABLE_RESTORE=1`              |
| `--no-eager-warmup`                        | warmup is on   | `KUNSERVE_MANAGER_EAGER_WARMUP=0`                |
| `--log-level INFO`                         | INFO           | `KUNSERVE_MANAGER_LOG_LEVEL`                     |

## Run (verl integration)

verl spawns this manager as a subprocess after both sglang HTTP servers
are up; see `verl/workers/rollout/sglang_rollout/async_sglang_server.py`.
Nothing in this package needs to know it is being run from verl.
