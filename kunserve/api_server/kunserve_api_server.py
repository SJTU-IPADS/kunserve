"""CLI module invoked as ``python -m kunserve.api_server.kunserve_api_server``.

Some deployment scripts still shell out to this name. The maintained path for
benchmarks in this repository is ``scripts/benchmark/evaluation.py`` together
with ``kunserve.llm.AsyncLLM``. If you need an OpenAPI-compatible HTTP server,
implement it against ``AsyncLLM`` or restore your internal server module here.
"""

from __future__ import annotations

import sys


def main() -> int:
    sys.stderr.write(
        "kunserve.api_server.kunserve_api_server: HTTP server is not shipped in "
        "this tree. Use `scripts/benchmark/evaluation.py` for evaluation runs, "
        "or integrate `kunserve.llm.AsyncLLM` from Python.\n"
        "See kunserve/README.md.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
