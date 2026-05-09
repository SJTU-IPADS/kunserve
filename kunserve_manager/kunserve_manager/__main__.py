"""Allow ``python -m kunserve_manager`` to launch the standalone manager."""
from kunserve_manager.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
