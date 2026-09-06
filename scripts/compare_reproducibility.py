from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bf_tap.artifacts import atomic_write_json, file_sha256


def _raw_difference(left: Path, right: Path) -> dict[str, object]:
    a = pd.read_csv(left, dtype={"sample_id": "string"})
    b = pd.read_csv(right, dtype={"sample_id": "string"})
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise ValueError("prediction shapes or columns differ")
    ids_equal = a["sample_id"].equals(b["sample_id"])
    values = ["pred_tap_iron", "pred_tap_time_len"]
    difference = np.abs(a[values].to_numpy() - b[values].to_numpy())
    return {
        "rows": len(a),
        "sample_ids_equal": ids_equal,
        "raw_max_abs_diff": float(difference.max()) if len(difference) else 0.0,
        "left_sha256": file_sha256(left),
        "right_sha256": file_sha256(right),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain-left", required=True, type=Path)
    parser.add_argument("--retrain-right", required=True, type=Path)
    parser.add_argument("--predict-left", required=True, type=Path)
    parser.add_argument("--predict-right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    retrain = {}
    for fold in ("DEV_LONG", "DEV_SHORT"):
        left = args.retrain_left / fold
        right = args.retrain_right / fold
        retrain[fold] = {
            **_raw_difference(
                left / "predictions_raw.csv", right / "predictions_raw.csv"
            ),
            "final_csv_byte_equal": (left / "predictions.csv").read_bytes()
            == (right / "predictions.csv").read_bytes(),
            "model_binary_sha256_equal": {
                target: file_sha256(left / "bundle" / f"{target}.cbm")
                == file_sha256(right / "bundle" / f"{target}.cbm")
                for target in ("tap_iron", "tap_time_len")
            },
        }

    prediction = {
        **_raw_difference(
            args.predict_left / "predictions_raw.csv",
            args.predict_right / "predictions_raw.csv",
        ),
        "result_csv_byte_equal": (args.predict_left / "result.csv").read_bytes()
        == (args.predict_right / "result.csv").read_bytes(),
        "runs": [
            json.loads((path / "prediction_manifest.json").read_text(encoding="utf-8"))
            for path in (args.predict_left, args.predict_right)
        ],
    }
    report = {
        "schema_version": 1,
        "retrain": retrain,
        "same_bundle_independent_processes": prediction,
        "interpretation": (
            "CatBoost binary files may contain run metadata; acceptance is based on "
            "raw prediction tolerance and final CSV bytes. Every bundle independently "
            "verifies its recorded component digests."
        ),
    }
    atomic_write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
