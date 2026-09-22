"""Retrain the frozen M0 rule on V2; keep inputs and release evidence private."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from .audit import digest, write_json
from .data import FEATURES, TARGETS, SUBMISSION_COLUMNS, read_table
from .metrics import score_targets
from .splits import make_folds
from .submission import deny_training_reads, package, predict_m0, validate_result


def load_v2(folder: Path, stage: str, rows: int):
    samples = read_table(folder / f"{stage}_samples.csv",
                         ("sample_id", "spout_no") + (TARGETS if stage == "train" else ()))
    features = read_table(folder / f"{stage}_features.csv", ("sample_id",) + FEATURES)
    pattern = rf"R2S2_{stage.upper()}_[0-9A-F]{{12}}"
    for frame in (samples, features):
        ids = frame.sample_id
        if len(frame) != rows or ids.isna().any() or ids.duplicated().any() or not ids.str.fullmatch(pattern).all():
            raise ValueError("Invalid V2 sample identity")
        if not np.isfinite(frame.drop(columns="sample_id").to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite V2 inputs")
    if set(samples.sample_id) != set(features.sample_id):
        raise ValueError("V2 feature ID mismatch")
    if not samples.spout_no.isin([1, 2, 3, 4]).all():
        raise ValueError("Invalid spout")
    if stage == "train" and (samples[list(TARGETS)] < 0).any().any():
        raise ValueError("Negative target")
    merged = samples.merge(features, on="sample_id", validate="one_to_one", sort=False)
    if stage == "test":
        template = pd.read_csv(folder / "result_template.csv", dtype={"sample_id": "string"})
        if list(template.columns) != list(SUBMISSION_COLUMNS) or template.sample_id.tolist() != samples.sample_id.tolist():
            raise ValueError("V2 template identity mismatch")
    return merged


def payload_v2(model, ids):
    ids = list(ids)
    if len(ids) != 322 or len(set(ids)) != 322 or not all(re.fullmatch(r"R2S2_TEST_[0-9A-F]{12}", s) for s in ids):
        raise ValueError("Expected 322 unique V2 IDs")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(SUBMISSION_COLUMNS)
    writer.writerows((sid, *(format(v, ".17g") for v in pair)) for sid, pair in zip(ids, predict_m0(model, ids)))
    payload = stream.getvalue().encode()
    validate_result(payload, ids)
    return payload


def run(root: Path, output: Path):
    if not output.resolve().is_relative_to(root.resolve() / "local/runs"):
        raise ValueError("Private run directory required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        paths = [root / f"复赛_{s}" / f"{s}_{kind}.csv" for s in ("train", "test") for kind in ("samples", "features")]
        paths += [root / "复赛_test/result_template.csv"]
        identity = {str(p.relative_to(root)): digest(p) for p in paths}
        write_json(output / "input_manifest.json", identity)
        train = load_v2(root / "复赛_train", "train", 2754)
        test = load_v2(root / "复赛_test", "test", 322)
        metrics = {}
        for seed in (42, 3407):
            folds = make_folds(train, seed).set_index("sample_id").loc[train.sample_id, "fold"].to_numpy()
            pred = np.empty((len(train), 2))
            for fold in range(5):
                pred[folds == fold] = train.loc[folds != fold, list(TARGETS)].median().to_numpy()
            metrics[str(seed)] = score_targets(train[TARGETS[0]], pred[:, 0], train[TARGETS[1]], pred[:, 1])
            oof = train[["sample_id", *TARGETS]].copy()
            oof["fold"] = folds
            oof[list(SUBMISSION_COLUMNS[1:])] = pred
            oof.to_csv(output / f"oof-{seed}.csv", index=False, mode="x")
        model = {"candidate": "R2_M0_GLOBAL_MEDIAN", "data_version": "synthetic_round2_v2",
                 "training_rows": len(train), "medians": {t: float(np.median(train[t])) for t in TARGETS},
                 "rule": "Full V2 public-training median per target; no scaling, bias or clipping",
                 "data_sha256": identity}
        write_json(output / "model.json", model)
        payload = payload_v2(model, test.sample_id)
        package(output, payload, test.sample_id)
        subprocess.run([sys.executable, "-m", "bf_tap_r2.v2_release", "--infer", "--output", str(output)], check=True, cwd=root)
        if (output / "rebuilt.csv").read_bytes() != payload:
            raise ValueError("Cold inference mismatch")
        if identity != {str(p.relative_to(root)): digest(p) for p in paths}:
            raise ValueError("Inputs changed during run")
        write_json(output / "verification.json", {
            "G0": "PASS", "rows": 322, "unique_ids": 322, "missing_ids": 0, "extra_ids": 0,
            "old_ids": 0, "test_order": True, "cold_inference_byte_identical": True,
            "inference_training_reads_prohibited": True,
            "G1": {"model": "M0 retrained on V2", "v2_oof_metrics": metrics, "platform_score": None},
            "result_sha256": digest(output / "result.csv"),
            "zip_sha256": digest(output / "Luqhhh_bf_tap_predict_round2.zip"),
            "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": digest(Path(__file__)), "uv_lock_sha256": digest(root / "uv.lock"),
            "platform_uploads": 0})
        print((output / "verification.json").read_text())
    except Exception as exc:
        write_json(output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()
    if args.infer:
        sys.addaudithook(deny_training_reads)
        model = json.loads((args.output / "model.json").read_text())
        test = load_v2(Path("复赛_test"), "test", 322)
        with (args.output / "rebuilt.csv").open("xb") as handle:
            handle.write(payload_v2(model, test.sample_id))
    else:
        run(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
