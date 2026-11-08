"""Command-line entry point for GPU checks, training and saved-run inspection."""

import argparse
import json
from pathlib import Path

from epgnn.dataset_build.storage import ROOT
from .runner import DEFAULT_CONFIG, gpu_smoke, run, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["gpu-smoke", "france", "full", "status"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--saved",
        action="store_true",
        help="Inspect the last saved full result (status only)",
    )
    args = parser.parse_args(argv)
    if args.saved and args.command != "status":
        parser.error(
            "--saved applies only to status; training always uses current code"
        )
    if args.command == "status":
        status(args.config, saved=args.saved)
    elif args.command == "gpu-smoke":
        config = json.loads(Path(args.config).read_text())
        gpu_smoke(config, ROOT / "runs/vibing_gpu_smoke")
    else:
        run(args.config, smoke=args.command == "france")


if __name__ == "__main__":
    main()
