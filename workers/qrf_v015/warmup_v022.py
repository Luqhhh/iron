"""V22-only April/May adapter around the frozen QRF numerical core."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np

from preprocessing import Preprocessor
from qrf_model import PARAMETERS, PROTOCOL, QRF
from worker import environment, payload, sha, source_identity, write


ADAPTER_PROTOCOL = "QRF_V022_WARMUP_ADAPTER_v1"
EXPECTED_ROWS = {4: 297, 5: 588}


def run(train_path: Path, manifest_path: Path, manifest_sha256: str, month: int, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if month not in EXPECTED_ROWS:
        raise ValueError("V22 warmup adapter permits only April/May cutoffs")
    if sha(manifest_path) != manifest_sha256:
        raise ValueError("warmup registration manifest identity changed")
    registration = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        registration.get("kind") != "V22_QRF_WARMUP_REGISTRATION_v1"
        or registration.get("adapter_protocol") != ADAPTER_PROTOCOL
        or registration.get("qrf_protocol") != PROTOCOL
        or registration.get("parameters") != PARAMETERS
        or registration.get("allowed_months") != [4, 5]
        or registration.get("expected_training_rows", {}).get(str(month)) != EXPECTED_ROWS[month]
        or registration.get("train_inputs", {}).get(str(month), {}).get("sha256") != sha(train_path)
        or registration.get("reuse_audit", {}).get(str(month)) != "NO_EQUIVALENT_CERTIFIED_MODEL"
        or Path(registration.get("output_paths", {}).get(str(month), "")).resolve() != output.resolve()
    ):
        raise ValueError("unregistered warmup protocol, parameters, budget, or input")
    arrays, info = payload(train_path, registration["train_inputs"][str(month)])
    if set(arrays) != {"ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months"}:
        raise ValueError("unexpected V22 warmup training payload")
    if len(set(info["ids"])) != EXPECTED_ROWS[month] or len(arrays["y"]) != EXPECTED_ROWS[month]:
        raise ValueError("V22 warmup unique training row count differs")
    cutoff = int(info["cutoff_ns"])
    if (arrays["reference_ns"] >= cutoff).any() or (arrays["available_ns"] > cutoff).any():
        raise ValueError("V22 warmup temporal boundary violation")
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "kind": "V22_QRF_WARMUP_BUNDLE_v1",
        "adapter_protocol": ADAPTER_PROTOCOL,
        "adapter_sha256": sha(Path(__file__)),
        "registration_manifest_sha256": manifest_sha256,
        "input": info,
        "sources": source_identity(),
        "environment": environment(),
        "parameters": PARAMETERS,
        "protocol": PROTOCOL,
        "month": month,
    }
    write(output / "preprocessor_intent.json", common)
    started = time.perf_counter()
    preprocessor = Preprocessor().fit(
        arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"]
    )
    write(output / "preprocessor.json", preprocessor.metadata())
    x, diagnostic = preprocessor.transform(
        arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"]
    )
    write(output / "forest_intent.json", common)
    model = QRF().fit(x, arrays["y"], info["ids"])
    model.training_months = arrays["training_months"]
    joblib.dump(model, output / "forest.joblib", compress=3)
    write(
        output / "bundle.json",
        {
            **common,
            "forest_sha256": sha(output / "forest.joblib"),
            "preprocessor_sha256": sha(output / "preprocessor.json"),
            "support": [float(model.y.min()), float(model.y.max())],
            "unique_training_rows": len(model.ids),
            "training_ids_sha256": hashlib.sha256(
                json.dumps(model.ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "training_diagnostic": diagnostic,
            "transformed_schema_sha256": hashlib.sha256(
                json.dumps(preprocessor.transformed_columns, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "fit_counts": {"forest": 1, "preprocessor": 1, "internal_trees": 256},
            "fit_seconds": time.perf_counter() - started,
            "model_bytes": (output / "forest.joblib").stat().st_size,
            "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--registration-manifest", required=True, type=Path)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--month", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(
        args.train.resolve(),
        args.registration_manifest.resolve(),
        args.registration_sha256,
        args.month,
        args.output.resolve(),
    )
