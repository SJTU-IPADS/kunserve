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
