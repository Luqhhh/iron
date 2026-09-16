"""Fresh QRF median/support inference from a root-certified V22 model manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from worker import payload, restore, sha, write, zero_fit


def _model_manifest(path: Path, expected_sha256: str) -> dict:
    if sha(path) != expected_sha256:
        raise ValueError("V22 model manifest identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "V22_CERTIFIED_QRF_MODEL_v1" or value.get("protocol") != "QRF_FULLTRAIN_LEAF_v1":
        raise ValueError("unknown V22 model manifest")
    required = {"model_path", "bundle_sha256", "training_identity_sha256", "unique_training_rows", "cutoff_ns"}
    if required - set(value):
        raise ValueError("incomplete V22 model manifest")
    if not isinstance(value["unique_training_rows"], int) or value["unique_training_rows"] < 1:
        raise ValueError("invalid certified unique training row count")
    return value


def infer(
    model_manifest_path: Path,
    model_manifest_sha256: str,
    input_path: Path,
    input_sha256: str,
    output: Path,
    expected_output: Path | None = None,
) -> None:
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError(output)
    if sha(input_path) != input_sha256:
        raise ValueError("V22 feature handoff identity changed")
    certified = _model_manifest(model_manifest_path, model_manifest_sha256)
    started = time.perf_counter()
    with zero_fit() as counts:
        model, preprocessor, bundle = restore(
            Path(certified["model_path"]), certified["bundle_sha256"]
        )
        if len(set(model.ids)) != certified["unique_training_rows"] or len(model.y) != certified["unique_training_rows"]:
            raise ValueError("loaded QRF unique response count differs from certified N")
        arrays, info = payload(input_path)
        if set(arrays) != {"ids", "numeric", "spout", "reference_ns"} or info["training"]:
            raise ValueError("V22 inference accepts label-free query features only")
        if (
            info["cutoff_ns"] != certified["cutoff_ns"]
            or info["cutoff_ns"] != bundle["input"]["cutoff_ns"]
            or info["raw_schema_sha256"] != bundle["input"]["raw_schema_sha256"]
            or set(info["ids"]) & set(model.ids)
            or (arrays["reference_ns"] < info["cutoff_ns"]).any()
        ):
            raise ValueError("V22 cutoff/schema/training-ID boundary differs")
        x, diagnostic = preprocessor.transform(
            arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"]
        )
        q, _, neighbors = model.predict(x, model.training_months)
        neff = np.asarray([row["effective_neighbors"] for row in neighbors], dtype=np.float64)
        n = certified["unique_training_rows"]
        tolerance = max(1e-8, 1e-12 * n)
        if not np.isfinite(q).all() or (q < 0).any() or (neff < 1 - tolerance).any() or (neff > n + tolerance).any():
            raise ValueError("invalid QRF median/effective support")
        checks = {}
        sequences = {
            "reverse": np.arange(len(x))[::-1],
            "subset": np.arange(0, len(x), max(1, len(x) // 7)),
            "single": np.asarray([len(x) // 2]),
        }
        for name, indices in sequences.items():
            transformed, _ = preprocessor.transform(
                arrays["numeric"][indices], arrays["spout"][indices],
                arrays["ids"][indices].tolist(), info["numeric_columns"]
            )
            part_q, _, part_neighbors = model.predict(transformed)
            part_neff = np.asarray([row["effective_neighbors"] for row in part_neighbors])
            if not np.array_equal(transformed, x[indices]) or not np.array_equal(part_q, q[indices]) or not np.array_equal(part_neff, neff[indices]):
                raise ValueError("V22 QRF median/support changes with caller subset/order")
            checks[name] = True
        chunks = [model.predict(x[i:i + 127]) for i in range(0, len(x), 127)]
        chunk_q = np.concatenate([part[0] for part in chunks])
        chunk_neff = np.concatenate([
            np.asarray([row["effective_neighbors"] for row in part[2]]) for part in chunks
        ])
        if not np.array_equal(chunk_q, q) or not np.array_equal(chunk_neff, neff):
            raise ValueError("V22 QRF median/support changes with chunking")
        checks["chunks"] = True
        if expected_output is not None:
            with np.load(expected_output, allow_pickle=False) as expected:
                if set(expected.files) != {"ids", "Q", "effective_neighbors", "N"}:
                    raise ValueError("unexpected saved V22 inference payload")
                if not all(np.array_equal(expected[name], value) for name, value in {
                    "ids": arrays["ids"], "Q": q, "effective_neighbors": neff,
                    "N": np.full(len(q), n, dtype=np.int64),
                }.items()):
                    raise ValueError("cold V22 QRF inference differs")
            checks["saved_output_exact"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez(
            handle,
            ids=arrays["ids"],
            Q=q,
            effective_neighbors=neff,
            N=np.full(len(q), n, dtype=np.int64),
        )
    write(
        Path(str(output) + ".json"),
        {
            "status": "PASS",
            "model_manifest_sha256": model_manifest_sha256,
            "input_sha256": input_sha256,
            "output_sha256": sha(output),
            "rows": len(q),
            "unique_training_rows": n,
            "training_identity_sha256": certified["training_identity_sha256"],
            "zero_fit": counts,
            "checks": checks,
            "preprocessor_diagnostic": diagnostic,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-output", type=Path)
    args = parser.parse_args()
    infer(
        args.model_manifest.resolve(), args.model_manifest_sha256,
        args.input.resolve(), args.input_sha256, args.output.resolve(),
        None if args.expected_output is None else args.expected_output.resolve(),
    )
