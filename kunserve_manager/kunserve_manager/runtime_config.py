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
            # The sglang backend's eager path uses dynamic padded
            # all-gather/all-reduce and is NOT graph-safe; only the
            # fixed_padded path is.  Allow capture only when the user has
            # explicitly opted into the static-buffer dispatcher path via
            # capture_policy=fixed_padded.  auto/<unset> stays eager to
            # preserve historical correctness-first behavior.
            return self.capture_policy == "fixed_padded"
        return True
