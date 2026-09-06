from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .audit import csv_manifest, write_json
from .config import load_yaml, validate_data_paths
from .exceptions import BFTapError, ContractError
from .offline import run_prediction, run_training
from .protection import load_protection_policy, record_protected_access
from .submission import pack_submission, validate_submission
from .validation import run_development_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bf-tap", description="Leakage-safe BF tap baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="write structural manifests without label statistics")
    audit.add_argument("--data-config", required=True)
    audit.add_argument("--output", required=True)

    validate = sub.add_parser("validate", help="run frozen temporal validation")
    validate.add_argument("--data-config", required=True)
    validate.add_argument("--config", required=True, dest="baseline_config")
    validate.add_argument("--feature-config", default="configs/features.yaml")
    validate.add_argument("--split-config", required=True)
    validate.add_argument("--data-contract", default="configs/data_contract.yaml")
    validate.add_argument("--protection-policy", default="configs/protection.yaml")
    validate.add_argument("--protection-ledger", default="local/manifests/protected_access.json")
    validate.add_argument("--suite", required=True, choices=("development",))
    validate.add_argument("--output", required=True)

    train = sub.add_parser("train", help="train and freeze a self-contained model bundle")
    train.add_argument("--data-config", required=True)
    train.add_argument("--data-contract", default="configs/data_contract.yaml")
    train.add_argument("--protection-policy", default="configs/protection.yaml")
    train.add_argument("--config", required=True, dest="baseline_config")
    train.add_argument("--feature-config", default="configs/features.yaml")
    train.add_argument("--mode", required=True, choices=("development", "official-release"))
    train.add_argument("--train-start", required=True)
    train.add_argument("--fit-cutoff")
    train.add_argument("--protection-ledger")
    train.add_argument("--frozen-manifest")
    train.add_argument("--output", required=True)

    predict = sub.add_parser("predict", help="predict from raw stage files using bundle semantics")
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--data-config", required=True)
    predict.add_argument("--stage", required=True, choices=("test_a", "test_b", "test_c"))
    predict.add_argument("--output", required=True)

    check = sub.add_parser("check-submission", help="validate a result against one stage")
    check.add_argument("--data-config", required=True)
    check.add_argument("--stage", required=True, choices=("test_a", "test_b", "test_c"))
    check.add_argument("--path", required=True)

    pack = sub.add_parser("pack", help="create an official-format ZIP")
    pack.add_argument("--stage", required=True, choices=("test_a", "test_b", "test_c"))
    pack.add_argument("--team-name", required=True)
    pack.add_argument("--result", required=True)
    pack.add_argument("--output-dir", required=True)
    pack.add_argument("--data-config", required=True)

    access = sub.add_parser(
        "register-protected-access",
        help="record an explicit protected-label lifecycle grant without reading labels",
    )
    access.add_argument("--protection-policy", default="configs/protection.yaml")
    access.add_argument("--ledger", required=True)
    access.add_argument(
        "--lifecycle", required=True, choices=("holdout_scoring", "final_training")
    )
    access.add_argument("--frozen-manifest", required=True)
    return parser


def _audit(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.data_config)
    validate_data_paths(cfg, command="audit")
    manifests = {}
    for name, raw_path in cfg.get("paths", {}).items():
        if raw_path is None or not str(raw_path).endswith(".csv"):
            continue
        path = Path(raw_path)
        protected = (
            ["tap_iron", "tap_time_len"]
            if name in {"train_samples", "tap_history_train"}
            else []
        )
        times = {
            "train_samples": ["reference_time"],
            "test_a_samples": ["reference_time"],
            "test_b_samples": ["reference_time"],
            "test_c_samples": ["reference_time"],
            "operation_hourly": ["clock"],
            "burden_change": ["cal_time"],
            "tap_history_train": ["reference_time", "tap_end_time"],
        }.get(name, [])
        keys = {
            "train_samples": ["sample_id"],
            "test_a_samples": ["sample_id"],
            "test_b_samples": ["sample_id"],
            "test_c_samples": ["sample_id"],
            "operation_hourly": ["clock"],
            "burden_change": ["cal_time"],
            "tap_history_train": ["sample_id"],
        }.get(name)
        manifests[name] = csv_manifest(
            path,
            key_columns=keys,
            time_columns=times,
            protected_columns=protected,
        )
    write_json(args.output, {"status": "STRUCTURAL_AUDIT_ONLY", "files": manifests})
    print(json.dumps({"output": args.output, "files": len(manifests)}, ensure_ascii=False))
    return 0


def _check(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.data_config)
    validate_data_paths(cfg, command="check-submission", stage=args.stage)
    samples_path = cfg["paths"].get(f"{args.stage}_samples")
    if not samples_path:
        raise ContractError(f"missing path for {args.stage}_samples")
    expected = pd.read_csv(samples_path, usecols=["sample_id"], dtype="string")["sample_id"]
    result = pd.read_csv(args.path, dtype={"sample_id": "string"})
    validate_submission(result, expected)
    print(json.dumps({"status": "PASS", "rows": len(result)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit":
            return _audit(args)
        if args.command == "validate":
            path = run_development_validation(
                data_config_path=args.data_config,
                baseline_config_path=args.baseline_config,
                feature_config_path=args.feature_config,
                split_config_path=args.split_config,
                semantic_contract_path=args.data_contract,
                protection_policy_path=args.protection_policy,
                protection_ledger_path=args.protection_ledger,
                output=args.output,
            )
            print(path)
            return 0
        if args.command == "train":
            path = run_training(
                data_config_path=args.data_config,
                semantic_contract_path=args.data_contract,
                protection_policy_path=args.protection_policy,
                baseline_config_path=args.baseline_config,
                feature_config_path=args.feature_config,
                output=args.output,
                mode=args.mode,
                train_start=args.train_start,
                fit_cutoff=args.fit_cutoff,
                protection_ledger_path=args.protection_ledger,
                frozen_manifest_path=args.frozen_manifest,
            )
            print(path)
            return 0
        if args.command == "predict":
            path = run_prediction(
                bundle_path=args.bundle,
                data_config_path=args.data_config,
                stage=args.stage,
                output=args.output,
            )
            print(path)
            return 0
        if args.command == "check-submission":
            return _check(args)
        if args.command == "pack":
            cfg = load_yaml(args.data_config)
            validate_data_paths(cfg, command="pack", stage=args.stage)
            expected = pd.read_csv(
                cfg["paths"][f"{args.stage}_samples"],
                usecols=["sample_id"],
                dtype={"sample_id": "string"},
            )["sample_id"]
            path = pack_submission(
                args.result,
                stage=args.stage,
                team_name=args.team_name,
                output_dir=args.output_dir,
                expected_ids=expected,
            )
            print(path)
            return 0
        if args.command == "register-protected-access":
            from .audit import sha256_file

            policy = load_protection_policy(args.protection_policy)
            result = record_protected_access(
                args.ledger,
                policy,
                lifecycle=args.lifecycle,
                frozen_manifest_sha256=sha256_file(args.frozen_manifest),
            )
            print(json.dumps(result, ensure_ascii=False))
            return 0
    except BFTapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable")
