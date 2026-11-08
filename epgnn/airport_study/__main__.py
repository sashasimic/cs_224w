"""python -m epgnn.airport_study {fetch,audit,smoke,full,status,report,verify}."""

import argparse
import json

from epgnn.dataset_build.storage import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["fetch", "audit", "smoke", "full", "status", "report", "verify"],
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs/experiments/airport_temperature.json")
    )
    args = parser.parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    if args.command == "fetch":
        from .sources import fetch

        fetch(config)
    elif args.command == "audit":
        from .data import prepare

        path, metadata, _ = prepare(config)
        print(
            json.dumps(
                {"cache": str(path), "coverage": metadata["coverage_summary"]}, indent=2
            )
        )
    else:
        from .runner import report_saved, run, status, verify_saved

        if args.command == "status":
            status(config)
        elif args.command == "report":
            report_saved(config)
        elif args.command == "verify":
            verify_saved(config)
        else:
            run(config, smoke=args.command == "smoke")


if __name__ == "__main__":
    main()
