"""Freeze, rebuild, and verify the deterministic V10 test-A submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bf_tap.artifacts import atomic_write_json
from bf_tap.optimization.v10_rebuild import build, freeze, verify_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "build", "verify"))
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.output is None:
            parser.error("freeze requires --output")
        print(freeze(args.data_config, args.output))
        return
    if args.command == "verify":
        if args.package is None:
            parser.error("verify requires --package")
        print(json.dumps(verify_package(args.package, args.data_config), indent=2))
        return
    required = {
        "--output": args.output,
        "--manifest": args.manifest,
        "--authorization": args.authorization,
        "--ledger": args.ledger,
        "--worker-python": args.worker_python,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("build missing " + ", ".join(missing))
    try:
        authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
        archive, release = build(
            args.manifest,
            authorization,
            args.ledger,
            args.worker_python,
            args.output,
        )
        print(json.dumps({"archive": str(archive), **release}, indent=2))
    except Exception as exc:
        failure = args.output / "failure.json"
        if not failure.exists():
            atomic_write_json(
                failure,
                {"status": "FAILED_PRESERVED", "error_type": type(exc).__name__, "error": str(exc)},
            )
        raise


if __name__ == "__main__":
    main()

