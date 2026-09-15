"""Build a black-box group-test challenger for the time correction.

TGATE improved V10 while TGATE600 regressed.  This package isolates the
spout-2 portion of TGATE (effective_neighbors < 500, same 25% recent-60-day
median shrink) and leaves spout 1 at the V10 control.  It is intended for one
platform probe, followed by a final re-submission of the best observed package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V10_ROOT = ROOT / "local/runs/optimization-v0.17-v10-final-rebuild-r2"
V10_RESULT_SHA256 = "d430da7399c3125db1768b1d7d236ec6eb95e13c633e5389b138dfe39c97b3c6"
ROWS = 335
CUTOFF = pd.Timestamp("2024-12-01 01:44:00", tz="Asia/Shanghai")
EFF_THRESHOLD = 500.0
SHRINK = 0.25
SPOUT = "2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(frame: pd.DataFrame, column: str) -> None:
    value = pd.to_datetime(frame[column])
    frame[column] = value.dt.tz_localize("Asia/Shanghai") if value.dt.tz is None else value.dt.tz_convert("Asia/Shanghai")


def recent_spout_medians(test: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
    result: list[float] = []
    spout = history.spout_no.astype(str).to_numpy()
    for row in test.itertuples(index=False):
        mask = (
            (spout == str(row.spout_no))
            & (history.reference_time < row.reference_time)
            & (history.reference_time >= row.reference_time - pd.Timedelta(days=60))
            & (history.tap_end_time <= CUTOFF)
        )
        local = history.loc[mask]
        if local.empty:
            local = history.loc[(spout == str(row.spout_no)) & (history.tap_end_time <= CUTOFF)]
        if local.empty:
            raise ValueError(f"no train-only history for spout {row.spout_no}")
        result.append(float(local.tap_time_len.median()))
    return np.asarray(result, dtype=float)


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    submission = output / "submission"
    submission.mkdir()

    v10_result = V10_ROOT / "submission/result.csv"
    if sha256(v10_result) != V10_RESULT_SHA256:
        raise ValueError("frozen V10 result identity changed")
    result = pd.read_csv(v10_result, dtype={"sample_id": str})
    test = pd.read_csv(ROOT / "初赛数据集/test/test_a_samples.csv", dtype={"sample_id": str})
    history = pd.read_csv(ROOT / "初赛数据集/train/tap_history_train.csv")
    parse_time(test, "reference_time")
    parse_time(history, "reference_time")
    parse_time(history, "tap_end_time")
    if len(result) != ROWS or len(test) != ROWS:
        raise ValueError("test-A row count differs")
    if result.sample_id.tolist() != test.sample_id.tolist():
        raise ValueError("V10 result IDs/order differ from test-A metadata")
    if (test.reference_time < CUTOFF).any():
        raise ValueError("test-A metadata precedes the frozen cutoff")

    bundle_path = V10_ROOT / "qrf/model/bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle["evaluation_input"]["ids"] != test.sample_id.tolist():
        raise ValueError("QRF diagnostics are not aligned with test-A IDs")
    neighbors = pd.DataFrame(bundle["neighbors"])
    if len(neighbors) != ROWS:
        raise ValueError("QRF neighbor diagnostics row count differs")

    recent = recent_spout_medians(test, history)
    current = result["pred_tap_time_len"].to_numpy(dtype=float)
    effective = neighbors["effective_neighbors"].to_numpy(dtype=float)
    spout2 = test.spout_no.astype(str).to_numpy() == SPOUT
    gate = (effective < EFF_THRESHOLD) & spout2
    adjusted = current.copy()
    adjusted[gate] = (1.0 - SHRINK) * current[gate] + SHRINK * recent[gate]
    if not np.isfinite(adjusted).all() or (adjusted < 0).any():
        raise ValueError("nonfinite or negative challenger time")

    candidate = pd.DataFrame({
        "sample_id": result.sample_id,
        "pred_tap_iron": result.pred_tap_iron.astype(float),
        "pred_tap_time_len": adjusted,
    })
    csv_path = submission / "result.csv"
    candidate.to_csv(csv_path, index=False, float_format="%.6f", lineterminator="\n")
    archive = submission / "Luqhhh_bf_tap_predict_prelim_TGATE_SPOUT2.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as handle:
        handle.write(csv_path, arcname="result.csv")
    with ZipFile(archive) as handle:
        if handle.namelist() != ["result.csv"] or hashlib.sha256(handle.read("result.csv")).hexdigest() != sha256(csv_path):
            raise ValueError("challenger archive payload mismatch")

    summary = {
        "status": "PASS_READY_TO_SUBMIT_EXPLORATORY",
        "candidate": "T_GATE_V10_QRF_MEDIAN_RECENT60_SPOUT2_ONLY",
        "stage": "test_a",
        "rows": ROWS,
        "base_result_sha256": V10_RESULT_SHA256,
        "result_sha256": sha256(csv_path),
        "archive_sha256": sha256(archive),
        "rule": {
            "effective_neighbors_lt": EFF_THRESHOLD,
            "spout": SPOUT,
            "shrink_toward_train_only_spout_recent60_median": SHRINK,
            "iron": "exact V10 copy",
            "test_targets_read": False,
            "test_distribution_used_for_selection": False,
        },
        "changed_rows": int(gate.sum()),
        "changed_fraction": float(gate.mean()),
        "time_delta_summary": {
            "mean": float(np.mean(adjusted - current)),
            "median": float(np.median(adjusted - current)),
            "min": float(np.min(adjusted - current)),
            "max": float(np.max(adjusted - current)),
        },
        "inputs": {
            "v10_result": str(v10_result),
            "qrf_bundle": str(bundle_path),
            "test_samples": str(ROOT / "初赛数据集/test/test_a_samples.csv"),
            "train_history": str(ROOT / "初赛数据集/train/tap_history_train.csv"),
        },
        "fit_counts": {"new_model": 0, "new_preprocessor": 0, "new_calibration_fit": 0},
        "platform_uploads": 0,
    }
    (output / "challenger_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate.assign(effective_neighbors=effective, gate=gate, recent60=recent).to_csv(
        output / "prediction_audit.csv", index=False, float_format="%.6f", lineterminator="\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
