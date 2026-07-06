"""PID evaluation entrypoint."""
from __future__ import annotations

import sys

from control.MiniController.main import main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "eval"])
    main()
