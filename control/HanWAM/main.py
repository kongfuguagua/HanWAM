"""HanWAM command dispatcher."""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="HanWAM train/eval dispatcher")
    parser.add_argument("command", choices=["train", "eval"], help="Subcommand to run")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.command == "train":
        from . import train

        train.main()
    else:
        from . import eval as eval_module

        eval_module.main()


if __name__ == "__main__":
    # Prefer `python -m control.HanWAM.train` and `python -m control.HanWAM.eval`.
    main()
