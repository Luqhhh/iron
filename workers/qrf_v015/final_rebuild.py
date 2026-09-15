"""Fit one frozen final QRF from hashed train/evaluation handoffs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import joblib
import numpy as np

from preprocessing import Preprocessor
from qrf_model import PARAMETERS, PROTOCOL, QRF
from worker import environment, payload, sha, source_identity, write


def _predict_checks(model, preprocessor, arrays, info):
    x, diagnostic = preprocessor.transform(
        arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"]
    )
    median, mean, neighbors = model.predict(x, model.training_months)
    sequences = {
        "reverse": np.arange(len(x))[::-1],
        "subset": np.arange(0, len(x), max(1, len(x) // 7)),
        "single": np.array([len(x) // 2]),
    }
    checks = {}
    for name, indices in sequences.items():
        transformed, _ = preprocessor.transform(
            arrays["numeric"][indices],
            arrays["spout"][indices],
            arrays["ids"][indices].tolist(),
            info["numeric_columns"],
        )
        q, m, _ = model.predict(transformed)
        if not np.array_equal(transformed, x[indices]):
            raise ValueError("QRF preprocessing changes with caller order or subset")
        if not np.array_equal(q, median[indices]) or not np.array_equal(m, mean[indices]):
            raise ValueError("QRF prediction changes with caller order or subset")
        checks[name] = True
    chunks = [model.predict(x[i : i + 127])[:2] for i in range(0, len(x), 127)]
    if not np.array_equal(np.concatenate([v[0] for v in chunks]), median):
        raise ValueError("QRF median changes with chunking")
    if not np.array_equal(np.concatenate([v[1] for v in chunks]), mean):
        raise ValueError("QRF mean changes with chunking")
    checks["chunks"] = True
    return median, mean, neighbors, diagnostic, checks


def run(train_path: Path, evaluation_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    train, train_info = payload(train_path)
    evaluation, evaluation_info = payload(evaluation_path)
    if set(train) != {
        "ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months"
    }:
        raise ValueError("unexpected final QRF training payload")
    if set(evaluation) != {"ids", "numeric", "spout", "reference_ns"}:
        raise ValueError("unexpected final QRF evaluation payload")
    if not train_info["training"] or evaluation_info["training"]:
        raise ValueError("training/evaluation handoff roles differ")
    if (
        train_info["cutoff_ns"] != evaluation_info["cutoff_ns"]
        or train_info["raw_schema_sha256"] != evaluation_info["raw_schema_sha256"]
        or train_info["numeric_columns"] != evaluation_info["numeric_columns"]
    ):
        raise ValueError("training/evaluation cutoff or raw schema differs")
    if set(train_info["ids"]) & set(evaluation_info["ids"]):
        raise ValueError("QRF training and evaluation IDs overlap")
    cutoff = train_info["cutoff_ns"]
    if (
        (train["reference_ns"] >= cutoff).any()
        or (train["available_ns"] > cutoff).any()
        or (evaluation["reference_ns"] < cutoff).any()
    ):
        raise ValueError("QRF temporal boundary violation")

    started = time.perf_counter()
    preprocessor = Preprocessor().fit(
        train["numeric"], train["spout"], train_info["ids"], train_info["numeric_columns"]
    )
    x_train, train_diagnostic = preprocessor.transform(
        train["numeric"], train["spout"], train_info["ids"], train_info["numeric_columns"]
    )
    model = QRF().fit(x_train, train["y"], train_info["ids"])
    model.training_months = train["training_months"]
    model_path = output / "forest.joblib"
    joblib.dump(model, model_path, compress=3)
    write(output / "preprocessor.json", preprocessor.metadata())

    median, mean, neighbors, evaluation_diagnostic, checks = _predict_checks(
        model, preprocessor, evaluation, evaluation_info
    )
    predictions_path = output / "predictions.npz"
    with predictions_path.open("xb") as handle:
        np.savez(handle, ids=evaluation["ids"], median=median, mean=mean)
    sources = source_identity()
    sources["final_rebuild.py"] = sha(Path(__file__))
    bundle = {
        "kind": "V8_FINAL_QRF_RECONSTRUCTION_v1",
        "protocol": PROTOCOL,
        "parameters": PARAMETERS,
        "environment": environment(),
        "sources": sources,
        "train_input": train_info,
        "evaluation_input": evaluation_info,
        "forest_sha256": sha(model_path),
        "preprocessor_sha256": sha(output / "preprocessor.json"),
        "predictions_sha256": sha(predictions_path),
        "support": [float(model.y.min()), float(model.y.max())],
        "fit_counts": {"forest": 1, "preprocessor": 1, "internal_trees": 256},
        "training_diagnostic": train_diagnostic,
        "evaluation_diagnostic": evaluation_diagnostic,
        "transformed_schema_sha256": hashlib.sha256(
            repr(preprocessor.transformed_columns).encode("utf-8")
        ).hexdigest(),
        "checks": checks,
        "neighbors": neighbors,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write(output / "bundle.json", bundle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.train.resolve(), args.evaluation.resolve(), args.output.resolve())
