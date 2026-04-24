# KunServe

KunServe is the Python-side **inference evaluation and experimentation stack** in this repository. It orchestrates Ray workers, scheduling, KV-cache / ballooning experiments, and benchmark drivers on top of the native C++ execution backend shipped as **FlashTransformer** (under `FlashTransformer/` in this tree).

## Disclaimer

- **Purpose.** We built KunServe primarily as a **research and evaluation tool** for studying LLM serving behavior (throughput, latency, memory, disaggregation, scheduling, and related ideas). It is **not** positioned or supported as an industrial-grade, fully productized inference service (HA, multi-tenant isolation, SRE playbooks, compliance, and so on are out of scope unless you add them yourself).

- **Model coverage.** The current codebase targets **dense Transformer models** on the paths exercised in our configs and scripts. **MoE, sparse experts, and other architectures are not first-class here.** To add structures beyond what is already wired, extend the C++ model path (see `FlashTransformer/src/csrc/model/sota/llama.cc` and headers there) following the same patterns, then hook weights and configs through the Python stack as needed.

- **No warranty.** Software is provided as-is for academic and experimental use. Validate correctness and performance in your own environment before any production-like deployment.

## Relationship to FlashTransformer

Weights are converted to the internal layout expected by FlashTransformer (see `kunserve/downloader/` and `FlashTransformer/scripts/`). Build FlashTransformer first, then install the Python package from the repository root (`pip install -e .`).

## Roadmap

- Weight loading from host memory / SSD.
- Support SOTA inference engines such as **vLLM** / **SGLang**, in addition to KunServe’s self-made execution stack.
- **CUDA graphs** for efficient decoding.
- Extend KunServe to **RL rollout** workloads.

## Acknowledgements

We learned a great deal from open-source communities and prior systems, including **Llumnix**, **DistServe** (and its **SwiftTransformer** execution stack, which this tree evolved away from while retaining engineering lineage), **vLLM**, **FlashInfer**, and **FlashAttention**, among others. Their ideas and code organization informed parts of this project; remaining bugs and design choices are our own.

## Further reading

- Repository overview and clone/build flow: see the [top-level README](../README.md).
- Benchmark and evaluation scripts: `scripts/benchmark/`, `scripts/evaluation/`, and `evaluation/`.
