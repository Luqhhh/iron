"""Deterministic round-two release packaging and label-free M0 inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

from .audit import digest, write_json
from .data import SUBMISSION_COLUMNS, TARGETS, load_config, load_snapshot, read_table

ZIP_NAME = "Luqhhh_bf_tap_predict_round2.zip"


def csv_bytes(ids, predictions):
    ids = list(ids)
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(ids), 2) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Predictions must be finite nonnegative pairs")
    if len(ids) != len(set(ids)) or not all(isinstance(s, str) and s.startswith("R2S_TEST_") for s in ids):
        raise ValueError("Invalid round-two test IDs")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(SUBMISSION_COLUMNS)
    writer.writerows((sid, format(i, ".17g"), format(t, ".17g")) for sid, (i, t) in zip(ids, values))
    return stream.getvalue().encode("utf-8")


def validate_result(payload, expected_ids):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if reader.fieldnames != list(SUBMISSION_COLUMNS):
        raise ValueError("Submission column order mismatch")
    rows = list(reader)
    ids = [row["sample_id"] for row in rows]
    if len(rows) != 322 or len(set(ids)) != 322 or ids != list(expected_ids):
        raise ValueError("Submission IDs or sample order mismatch")
    if any(set(row) != set(SUBMISSION_COLUMNS) for row in rows):
        raise ValueError("Unexpected extra CSV values")
    values = np.array([[float(row[name]) for name in SUBMISSION_COLUMNS[1:]] for row in rows])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid submission numbers")
    return rows


def package(output, payload, ids):
    validate_result(payload, ids)
    with (output / "result.csv").open("xb") as handle:
        handle.write(payload)
    info = zipfile.ZipInfo("result.csv", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(output / ZIP_NAME, "x") as archive:
        archive.writestr(info, payload)
    with zipfile.ZipFile(output / ZIP_NAME) as archive:
        if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload or archive.testzip():
            raise ValueError("ZIP identity failure")


def predict_m0(model, ids):
    if model["candidate"] != "R2_M0_GLOBAL_MEDIAN":
        raise ValueError("Not an M0 model")
    return np.tile([model["medians"][t] for t in TARGETS], (len(ids), 1))


def deny_training_reads(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = str(args[0])
        if "复赛_train" in path or any(name in path for name in ("train_samples.csv", "train_features.csv", "tap_history_train.csv")):
            raise RuntimeError("Inference must not read training inputs")


def build_m0(root, output):
    from .cv import verify_audit
    if not output.resolve().is_relative_to(root.resolve() / "local/runs/round2-v0.2"):
        raise ValueError("Release must remain private")
    identity = verify_audit(root, root / "local/runs/round2-v0.1/p0-audit-r2")
    cfg = load_config(root / "configs/round2_v0_1/data.yaml")
    train, _ = load_snapshot(root, cfg, "train")
    test, _ = load_snapshot(root, cfg, "test")
    output.mkdir(parents=True, exist_ok=False)
    try:
        medians = {t: float(np.median(train[t].to_numpy())) for t in TARGETS}
        model = {"candidate": "R2_M0_GLOBAL_MEDIAN", "medians": medians, "training_rows": len(train),
                 "rule": "numpy.median over all public round2 training rows, independently per target; no clipping or bias",
                 "data_sha256": {k: v["sha256"] for k, v in identity["files"].items()}}
        write_json(output / "model.json", model)
        ids = test.sample_id.tolist()
        payload = csv_bytes(ids, predict_m0(model, ids))
        package(output, payload, ids)
        subprocess.run([sys.executable, "-m", "bf_tap_r2.submission", "infer", "--model", str(output / "model.json"),
                        "--samples", str(root / cfg["test_samples"]), "--output", str(output / "rebuilt.csv")], check=True, cwd=root)
        if (output / "rebuilt.csv").read_bytes() != payload:
            raise ValueError("Independent-process CSV byte mismatch")
        shuffled = test.sample(frac=1, random_state=3407)
        shuffled_predictions = pd.DataFrame(predict_m0(model, shuffled.sample_id), index=shuffled.sample_id).loc[ids].to_numpy()
        chunked = np.concatenate([predict_m0(model, part.sample_id) for part in (test.iloc[:1], test.iloc[1:111], test.iloc[111:])])
        if csv_bytes(ids, shuffled_predictions) != payload or csv_bytes(ids, chunked) != payload:
            raise ValueError("Input order or chunk invariance failed")
        manifest = {"candidate": model["candidate"], "state": "BASELINE_READY_FOR_PLATFORM_CHECK",
                    "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    "source_sha256": digest(Path(__file__)), "uv_lock_sha256": digest(root / "uv.lock"),
                    "metric_contract_sha256": digest(root / "configs/round2_v0_2/metric_contract.yaml"),
                    "data_sha256": model["data_sha256"], "model_sha256": digest(output / "model.json"),
                    "result_sha256": digest(output / "result.csv"), "zip_sha256": digest(output / ZIP_NAME),
                    "platform_submission_id": None, "platform_score": None, "platform_uploads": 0}
        write_json(output / "manifest.json", manifest)
        write_json(output / "verification.json", {"status": "pass", "rows": 322, "unique_ids": 322,
                   "columns": list(SUBMISSION_COLUMNS), "finite_nonnegative": True, "test_order_restored": True,
                   "zip_only_result_csv": True, "independent_process_byte_identical": True,
                   "inference_training_reads_prohibited": True, "shuffle_and_chunk_byte_identical": True,
                   "result_sha256": manifest["result_sha256"], "zip_sha256": manifest["zip_sha256"], "new_regressor_fits": 0})
        print(json.dumps(manifest, ensure_ascii=False))
    except Exception as exc:
        write_json(output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-m0")
    build.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--samples", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-m0":
        build_m0(Path.cwd(), args.output)
    else:
        sys.addaudithook(deny_training_reads)
        model = json.loads(args.model.read_text())
        samples = read_table(args.samples, ("sample_id", "spout_no"))
        payload = csv_bytes(samples.sample_id, predict_m0(model, samples.sample_id))
        validate_result(payload, samples.sample_id)
        with args.output.open("xb") as handle:
            handle.write(payload)


if __name__ == "__main__":
    main()
