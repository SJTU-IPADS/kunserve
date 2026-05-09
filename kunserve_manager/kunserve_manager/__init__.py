"""KunServe sidecar manager — standalone control plane for grouped sglang replicas.

Usage:
    python -m kunserve_manager --replica HOST:PORT --replica HOST:PORT \
        --model-path /path/to/model

This package is intentionally decoupled from sglang's inference processes:
each replica only needs to expose the /kunserve/* HTTP endpoints; a single
manager process polls them, decides BALLOON state, and dispatches RPCs.

Importable for tests / programmatic use:

    from kunserve_manager import KunServeController
"""
from kunserve_manager.controller import KunServeController

__all__ = ["KunServeController"]
__version__ = "0.1.0"
