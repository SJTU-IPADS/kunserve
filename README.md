# KunServe

This repository hosts **KunServe** (Python orchestration, Ray, scheduling experiments) and **FlashTransformer** (high-performance dense Transformer inference kernels and model execution in C++/CUDA), used together for **LLM serving research and reproducible evaluation**.

## Disclaimer (read first)

- **Academic and evaluation focus.** The stack is intended for **research, benchmarking, and prototyping**, not as a drop-in replacement for commercial managed inference. Operability, security hardening, and long-term API stability are not goals of this release on their own.

- **Dense models.** We currently focus on **dense** architectures in the supported code paths. For new topologies, start from the existing Llama-oriented implementation under `FlashTransformer/src/csrc/model/sota/` and extend thoughtfully.

- **Upstream lineage.** The C++ backend was forked and renamed to **FlashTransformer** in this tree; Python tooling is branded **KunServe**. Historical ties to earlier research systems are acknowledged in [kunserve/README.md](kunserve/README.md).

## Quick start

```shell
git clone <this-repository> kunserve && cd kunserve

# Submodules: FlashTransformer is declared in the repo-root `.gitmodules`.
# Nested deps (flash-attention, flashinfer) stay in `FlashTransformer/.gitmodules`
# with paths relative to that directory — use recursive update from the root:
git submodule update --init --recursive

# Conda environment (see environment.yml at repo root)
conda env create -f environment.yml && conda activate kunserve

# Build FlashTransformer (CUDA toolchain required)
cmake -S FlashTransformer -B FlashTransformer/build -DBUILD_MODE=RELEASE && cmake --build FlashTransformer/build -j"$(nproc)"

pip install -e .
```

After a successful build you should have `libflash_pybinding.so` under `FlashTransformer/build/lib` (exact layout may depend on your CMake preset).

The Git submodule URL for `FlashTransformer` is set in `.gitmodules` to the **FlashTransformer** project name. If your hosting still serves the repository under an older remote name, override it locally, for example: `git config submodule.FlashTransformer.url <your-clone-url>`.

## Documentation

- **KunServe** (Python layer, evaluation scope, acknowledgements): [kunserve/README.md](kunserve/README.md)
- **FlashTransformer** (native backend): [FlashTransformer/README.md](FlashTransformer/README.md)
- **Scripts** (benchmarks, evaluation): [scripts/README.md](scripts/README.md)

## Roadmap

Planned directions (non-binding; order and timing may change):

- **Weight loading from host memory / SSD** — broader storage tiers beyond today’s assumptions, for large models and flexible deployment.
- **Integrate SOTA inference engines (e.g. vLLM, SGLang)** — optional backends alongside KunServe’s research-oriented execution stack, so benchmarks and scheduling ideas can target widely used runtimes.
- **CUDA graphs for efficient decoding** — capture and replay steady-state decode for lower launch overhead where the graph constraints are acceptable.
- **Extend KunServe to RL rollout** — reuse orchestration, batching, and tracing for rollout-heavy RL training loops, not only static serving benchmarks.

## Citation

If this codebase helps your research, please cite the publications that match the components you build on (for example, the **DistServe** paper if you compare against or extend disaggregated prefill/decode serving), **vLLM**, **FlashAttention**, **FlashInfer**, **Llumnix**, and your own work as appropriate.
