"""Prepare and run both graph studies from a pinned local catalog."""

import argparse
from pathlib import Path

from epgnn.dataset_build.storage import ROOT
from epgnn.relational_forecast.catalog import read_json

from . import inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "smoke", "run", "validate", "report"]
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/combined_graph_study.json",
    )
    parser.add_argument(
        "--family", choices=["all", "xgb", "relational", "adaptive"], default="all"
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    prepared = inputs.preparation(config)
    if args.command == "prepare":
        print(prepared, flush=True)
    else:
        from . import runner

        runner.dispatch(args, config, prepared)


if __name__ == "__main__":
    main()
