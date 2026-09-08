"""Pair existing development predictions across adjacent training origins.

This analysis opens only an existing development reference run, never official
label files. It does not fit models or select a release candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, file_sha256, stable_digest
from ..exceptions import ContractError

TARGETS = ("tap_iron", "tap_time_len")


def paired_month(older: pd.DataFrame, newer: pd.DataFrame, month: str) -> dict:
    start = pd.Timestamp(month + "-01", tz="Asia/Shanghai")
    end = start + pd.offsets.MonthBegin(1)
    if start < pd.Timestamp("2024-07-01", tz="Asia/Shanghai") or end > pd.Timestamp("2024-11-01", tz="Asia/Shanghai"):
        raise ContractError("refresh analysis is restricted to July–October development")
    parts = []
    for frame in (older, newer):
        frame = frame.copy()
        required = {"sample_id", "reference_time", *TARGETS, *[f"pred_{t}" for t in TARGETS]}
        if required - set(frame) or frame.empty or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ContractError("refresh requires unique, non-null sample IDs and complete predictions")
        frame["sample_id"] = frame.sample_id.astype(str)
        if frame.sample_id.duplicated().any():
            raise ContractError("sample IDs collide after normalization")
        times = pd.to_datetime(frame.reference_time, utc=True).dt.tz_convert("Asia/Shanghai")
        if not ((times >= start) & (times < end)).all():
            raise ContractError("refresh rows are outside the registered evaluation month")
        frame["reference_time"] = times
        numeric = frame[[*TARGETS, *[f"pred_{t}" for t in TARGETS]]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ContractError("refresh requires finite nonnegative targets and predictions")
        parts.append(frame.set_index("sample_id").sort_index())
    old, new = parts
    if not old.index.equals(new.index):
        raise ContractError("same-month origins must have exactly the same sample IDs")
    if not old[["reference_time", *TARGETS]].equals(new[["reference_time", *TARGETS]]):
        raise ContractError("paired sample times or targets disagree")
    result = {"status": "PAIRED", "samples": len(old),
              "sample_ids_sha256": stable_digest(old.index.tolist())}
    losses = {}
    for name, frame in (("older", old), ("newer", new)):
        ratios = []
        for target in TARGETS:
            denominator = float(frame[target].sum())
            if denominator <= 0:
                raise ContractError("refresh target denominator must be positive")
            error = float((frame[f"pred_{target}"] - frame[target]).abs().sum())
            result.update({f"{name}_{target}_abs_error_sum": error,
                           f"{name}_{target}_denominator": denominator,
                           f"{name}_{target}_wmape": error / denominator})
            ratios.append(error / denominator)
        losses[name] = result[f"{name}_E"] = float(np.mean(ratios))
    result["refresh_gain"] = losses["older"] - losses["newer"]
    return result


def run(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    try:
        status = json.loads((source / "final_status.json").read_text())
        if (status.get("protected_labels_read") is not False
                or status.get("suite") != "reference"
                or status.get("engineering_status") != "G0_EXECUTION_PASS"):
            raise ContractError("source must be a successful development reference run")
        identities = {name: file_sha256(source / name) for name in
                      ("final_status.json", "resolved_config.json", "registry.jsonl")}
        registry = [json.loads(line) for line in (source / "registry.jsonl").read_text().splitlines()]
        rows = []
        for candidate in ("E09", "E12-raw"):
            for month in range(7, 11):
                old_origin, new_origin = f"O2024{month-1:02d}", f"O2024{month:02d}"
                row = {"candidate": candidate, "month": f"2024-{month:02d}",
                       "older_cell": old_origin + "_H2", "newer_cell": new_origin + "_H1"}
                for side, origin, m in (("older", old_origin, month-1), ("newer", new_origin, month)):
                    cutoff = pd.Timestamp(f"2024-{m:02d}-01", tz="Asia/Shanghai")
                    components = ("E09",) if candidate == "E09" else ("E09", "E04")
                    for component in components:
                        fits = [r for r in registry if r.get("candidate") == component and r.get("context") == origin]
                        if len(fits) != 1 or pd.Timestamp(fits[0]["cutoff"]) != cutoff:
                            raise ContractError("source fit registry does not match refresh origins")
                    row[f"{side}_fit_cutoff"] = str(cutoff)
                    row[f"{side}_history_cutoff"] = str(cutoff)
                paths = [source / "units" / row[f"{side}_cell"] / candidate / "errors.csv"
                         for side in ("older", "newer")]
                if not all(p.is_file() for p in paths):
                    row.update(status="MISSING", missing_files=[str(p.relative_to(source)) for p in paths if not p.is_file()])
                else:
                    for p in paths:
                        identities[str(p.relative_to(source))] = file_sha256(p)
                    row.update(paired_month(*(pd.read_csv(p, dtype={"sample_id": str}) for p in paths), row["month"]))
                rows.append(row)
        # Catch concurrent changes; do not silently attach a stale input digest.
        if any(file_sha256(source / name) != digest for name, digest in identities.items()):
            raise ContractError("source evidence changed during refresh analysis")
        table = pd.DataFrame(rows)
        table.to_csv(output / "training_refresh_comparison.csv", index=False)
        summary = {}
        for candidate in ("E09", "E12-raw"):
            paired = table.loc[(table.candidate == candidate) & (table.status == "PAIRED")]
            summary[candidate] = {"paired_months": len(paired), "missing_months": 4-len(paired),
                                  "mean_gain": float(paired.refresh_gain.mean()) if len(paired) else None,
                                  "median_gain": float(paired.refresh_gain.median()) if len(paired) else None}
        report = {"schema_version": "training-refresh-v1", "summary": summary,
                  "source_run": str(source), "source_sha256": identities,
                  "comparison_csv_sha256": file_sha256(output / "training_refresh_comparison.csv"),
                  "protected_labels_read": False, "new_model_fits": 0,
                  "scope": "joint_training_and_history_refresh_not_platform_forecast",
                  "history_cutoff_evidence": "source reference runner freezes history at registered fit cutoff"}
        atomic_write_json(output / "refresh_summary.json", report)
        atomic_write_json(output / "final_status.json", {"G0": "PASS_ANALYSIS", "G1": "NOT_EVALUATED", "protected_labels_read": False})
        return report
    except Exception as exc:
        atomic_write_json(output / "final_status.json", {"status": "FAILED", "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.reference_run, args.output)["summary"], indent=2))


if __name__ == "__main__":
    main()
