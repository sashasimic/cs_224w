import argparse
import json

from .pipeline import audit, build, validate


def main():
    parser = argparse.ArgumentParser(description="Auditable noon-Brussels forecasting dataset construction")
    parser.add_argument("command", choices=["audit", "build", "validate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--download", action="store_true", help="audit only: fetch missing explicit releases with checksum verification; never pin/upload")
    args = parser.parse_args()
    if args.download and args.command != "audit":
        parser.error("--download is available only with audit")
    try:
        if args.command == "audit":
            result = audit(args.config, download=args.download)
            print(json.dumps({k: result[k] for k in ["status", "fingerprint", "findings"]}, indent=2))
            return 2 if result["status"] == "blocked" else 0
        result = build(args.config) if args.command == "build" else validate(args.config)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"dataset_build: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
