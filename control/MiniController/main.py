"""Unified MiniController experiment entrypoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config_schema import CONFIG_PATH, load_config, stage_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniController train/eval stages from YAML config")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--stage", default=None, help="Comma-separated stages; defaults to experiment.stages")
    parser.add_argument("--mode", default=None, help="Override experiment/train/eval mode: 1, 3, or all")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.mode is not None:
        config["experiment"]["mode"] = args.mode
        config["train"]["mode"] = args.mode
        config["eval"]["mode"] = args.mode
    if args.experiment_name is not None:
        config["experiment"]["name"] = args.experiment_name
    if args.checkpoint is not None:
        config["method"]["checkpoint_template"] = args.checkpoint
    if args.device is not None:
        config["train"]["device"] = args.device
    return config


def main() -> dict:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    stages = stage_list(config, args.stage)
    results = {}
    if "train" in stages:
        if config["method"]["name"] != "hanwam":
            raise ValueError("Only method.name=hanwam supports train stage.")
        from control.HanWAM.train import main as train_main

        results["train"] = train_main(config)
    if "eval" in stages:
        from .experiment import run_eval_from_config

        results["eval"] = run_eval_from_config(config)
    return results


if __name__ == "__main__":
    main()
