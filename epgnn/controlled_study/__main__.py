"""python -m epgnn.controlled_study {dry-run,smoke,full,status}"""

import argparse
import json
from pathlib import Path

from .design import jobs
from .preflight import DEFAULT_CONFIG
from .runner import run, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["dry-run", "smoke", "full", "status"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "dry-run":
        planned = jobs(config)
        print(
            json.dumps(
                {
                    "job_count": len(planned),
                    "jobs": [{"name": job.name, **job.record()} for job in planned],
                },
                indent=2,
            )
        )
    elif args.command == "status":
        status(config)
    else:
        run(config, smoke=args.command == "smoke")


if __name__ == "__main__":
    main()
