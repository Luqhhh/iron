"""Cold-process feature-profile verification; reads sample metadata, never targets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import (atomic_write_json, file_identities, file_sha256,
                         validate_inference_source_contract, verify_file_identities)
from ..config import load_yaml
from ..data import normalize_event_source
from ..exceptions import ContractError
from ..io import read_csv
from .indexed_features import load_known_index
from .v3_followup import read, snapshot_sources
from .v3_remaining_features import ProfilePredictor


def verify_profiles(source: Path, output: Path, data_config: str, index_paths: list[str]):
    if read(source / "final_status.json").get("engineering_status") != "G0_EXECUTION_PASS":
        raise ContractError("profile verification requires a completed development run")
    data = load_yaml(data_config)
    identities = file_identities({key: data["paths"][key] for key in ("operation_hourly", "burden_change")})
    source_identity = file_identities({str(p): p for p in source.rglob("*") if p.is_file()})
    index, index_audit = load_known_index(index_paths)
    if index_audit != read(source / "known_index_manifest.json"):
        raise ContractError("profile verification index sources differ from training")
    checks = []
    metrics = read(source / "candidate_metrics.json")
    for bundle in sorted((source / "bundles").glob("*/*")):
        origin, profile = bundle.parent.name, bundle.name
        predictor = ProfilePredictor(bundle)
        metadata = predictor.metadata
        validate_inference_source_contract(metadata["inference_source_contract"], identities,
                                            semantic_contract_sha256=metadata["contract_digests"]["semantic_contract_sha256"])
        expected_parts = []
        for unit_id, unit in metrics.items():
            if unit["origin_id"] == origin:
                expected_parts.append(pd.read_csv(source / "units" / unit_id / profile / "predictions.csv",
                                                 dtype={"sample_id": "string"}, float_precision="round_trip"))
        if not expected_parts:
            raise ContractError("profile bundle has no matching recorded evaluation units")
        expected = pd.concat(expected_parts, ignore_index=True)
        if expected.sample_id.duplicated().any():
            raise ContractError("profile evaluation samples unexpectedly overlap within an origin")
        queries = index.loc[index.sample_id.isin(expected.sample_id)].groupby("spout_no", sort=True).head(2)
        if queries.empty or len(index.loc[index.sample_id.isin(expected.sample_id)]) != len(expected):
            raise ContractError("recorded profile predictions and explicit metadata do not match")
        frames = {}
        for kind, role in (("operation", "operation_hourly"), ("burden", "burden_change")):
            mapping = metadata["semantic_contract"]["sources"][kind]
            frames[kind] = normalize_event_source(read_csv(data["paths"][role]),
                event_time_column=mapping["event_time_column"], available_at_column=mapping["available_at_column"],
                value_columns=metadata["feature_config"][kind]["value_columns"], missing_markers=mapping["missing_markers"])
        predictions = predictor.predict(queries, frames["operation"], frames["burden"], index)
        joined = predictions.merge(expected, on="sample_id", suffixes=("_restored", "_recorded"), validate="one_to_one")
        difference = max(float(np.max(np.abs(joined[f"pred_{t}_restored"] - joined[f"pred_{t}_recorded"])))
                         for t in ("tap_iron", "tap_time_len"))
        if difference > 1e-12 or len(joined) != len(queries):
            raise ContractError("cold-process profile prediction differs")
        checks.append({"origin": origin, "profile": profile, "samples": len(queries),
                       "bundle_sha256": file_sha256(bundle / "bundle.json"), "max_absolute_difference": difference})
    if len(checks) != 14:
        raise ContractError("expected two profiles at each of seven development origins")
    verify_file_identities(identities)
    verify_file_identities(index_audit["sources"])
    verify_file_identities(source_identity)
    atomic_write_json(output / "checks.json", checks)
    atomic_write_json(output / "final_status.json", {
        "engineering_status": "G0_14_PROFILE_BUNDLE_COLD_PROCESS_PASS", "bundles_checked": len(checks),
        "max_absolute_difference": max(c["max_absolute_difference"] for c in checks),
        "targets_read": False, "protected_labels_read": False,
        "scope": "two_samples_per_observed_spout_per_origin_not_full_three_stage_release"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", nargs="+", required=True)
    parser.add_argument("--data-config", default="configs/data.local.yaml")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = snapshot_sources(args.output)
        verify_profiles(args.run, args.output, args.data_config, args.sample_index)
        verify_file_identities(snapshot["files"])
    except Exception as exc:
        atomic_write_json(args.output / "final_status.json", {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
                          overwrite=(args.output / "final_status.json").exists())
        raise


if __name__ == "__main__":
    main()
