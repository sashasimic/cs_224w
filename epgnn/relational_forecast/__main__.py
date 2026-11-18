"""Build the local catalog or run the conditional CORE forecast experiment."""

import argparse
from pathlib import Path

from . import catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "validate", "smoke", "run"])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/relational_forecast.json"),
    )
    args = parser.parse_args()
    settings = catalog.load_settings(args.config)
    if args.command == "build":
        print(catalog.build(settings), flush=True)
    elif args.command == "validate":
        print(catalog.validate(settings), flush=True)
    else:
        from .experiment import run

        run(settings, smoke=args.command == "smoke")


if __name__ == "__main__":
    main()
