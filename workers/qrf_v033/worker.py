#!/usr/bin/env python3
"""Locked Python 3.12 worker for v0.33 forest training and OOB inference."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
FROZEN = HERE.parent / "qrf_v015"
V26_WORKER = HERE.parent / "qrf_v026"
sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(V26_WORKER))
sys.path.insert(0, str(HERE))

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from preprocessing import Preprocessor, validate
from qrf_model import QRF
from oob_sampled import PROTOCOL as OOB_PROTOCOL, derive_attachment, load_npz, predict as predict_oob, validate_attachment
from sampled_forest import CANDIDATES, PARAMETERS, PROTOCOL as FOREST_PROTOCOL, SampledTimeForest, expected_draws


V26 = REPOSITORY / "local/runs/optimization-v0.26-qrf-partition-tests-r1"
EXPECTED_ROWS = {6: 888, 7: 1180, 8: 1490, 9: 1803, 10: 2091, 11: 2424, 12: 2754}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_identity(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": array.shape}, sort_keys=True).encode())
    digest.update(array.tobytes())
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest.hexdigest()}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)


def environment():
    return {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__,
    }


def source_identity():
    files = {
        "worker": HERE / "worker.py", "sampled_forest": HERE / "sampled_forest.py",
        "oob_sampled": HERE / "oob_sampled.py", "qrf_model": FROZEN / "qrf_model.py",
        "preprocessing": FROZEN / "preprocessing.py", "lock": FROZEN / "uv.lock",
    }
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in files.items()}


def registration():
    return {
        "forest_protocol": FOREST_PROTOCOL, "oob_protocol": OOB_PROTOCOL,
        "candidates": CANDIDATES, "parameters": PARAMETERS,
        "expected_training_rows": EXPECTED_ROWS,
        "expected_draws": {candidate: {slot: expected_draws(candidate, rows) for slot, rows in EXPECTED_ROWS.items()} for candidate in CANDIDATES},
    }


def _feature_path(slot: int, kind: str):
    return V26 / "features" / str(slot) / f"{kind}.npz"


def payload(slot: int, kind: str):
    path = _feature_path(slot, kind)
    info = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if sha(path) != info["sha256"]:
        raise ValueError("certified v0.26 handoff identity changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if arrays["ids"].tolist() != info["ids"]:
        raise ValueError("certified handoff ID identity changed")
    validate(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    if len(info["numeric_columns"]) != 209 or len(info["raw_schema"]) != 210:
        raise ValueError("v0.33 requires the original 210-column handoff")
    return arrays, info, path


def preprocessor(info, arrays, *, training):
    path = (REPOSITORY / info["original_preprocessor_path"]).resolve()
    if not path.is_relative_to(REPOSITORY.resolve()) or sha(path) != info["original_preprocessor_sha256"]:
        raise ValueError("original preprocessor identity differs")
    value = Preprocessor.restore(json.loads(path.read_text(encoding="utf-8")))
    if value.columns != info["numeric_columns"]:
        raise ValueError("original preprocessor columns differ")
    if training and value.training_ids != arrays["ids"].tolist():
        raise ValueError("original preprocessor training IDs differ")
    transformed, diagnostic = value.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    digest = hashlib.sha256(json.dumps(value.transformed_columns).encode()).hexdigest()
    if digest != info["original_transformed_schema_sha256"] or transformed.shape[1] != 213 or transformed.dtype != np.float32:
        raise ValueError("certified 213-column float32 transformed schema differs")
    return value, transformed, diagnostic, path


def _registered(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered worker environment/source differs")
    expected = json.loads(json.dumps(registration(), sort_keys=True))
    if manifest["worker_registration"] != expected:
        raise ValueError("registered v0.33 worker contract differs")
    return manifest


def _v26_parent(slot: int):
    completion = V26 / ("final_models_complete.json" if slot == 12 else "development_models_complete.json")
    trusted = json.loads(completion.read_text(encoding="utf-8"))["A"][str(slot)]
    folder = V26 / "models/A" / str(slot)
    if sha(folder / "bundle.json") != trusted:
        raise ValueError("v0.26 parent bundle identity differs")
    metadata = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if sha(folder / "forest.joblib") != metadata["forest_sha256"]:
        raise ValueError("v0.26 parent forest identity differs")
    model = joblib.load(folder / "forest.joblib")
    return model, metadata


def audit(root: Path, slot: int):
    _registered(root)
    result = {}
    for kind in ("train", "evaluation"):
        arrays, info, path = payload(slot, kind)
        prep, transformed, diagnostic, prep_path = preprocessor(info, arrays, training=kind == "train")
        result[kind] = {
            "rows": len(arrays["ids"]), "handoff_path": str(path.relative_to(REPOSITORY)), "handoff_sha256": sha(path),
            "ids": array_identity(arrays["ids"]), "numeric": array_identity(arrays["numeric"]),
            "spout": array_identity(arrays["spout"]), "transformed": array_identity(transformed),
            "transformed_columns": len(prep.transformed_columns), "preprocessor_sha256": sha(prep_path),
            "transform_diagnostic": diagnostic,
        }
    if result["train"]["rows"] != EXPECTED_ROWS[slot]:
        raise ValueError("registered training row count differs")
    parent, metadata = _v26_parent(slot)
    if parent.ids != payload(slot, "train")[0]["ids"].tolist() or len(parent.forest.estimators_) != 256:
        raise ValueError("v0.26 parent training/tree identity differs")
    return {
        "slot": slot, "inputs": result, "parent_forest_sha256": metadata["forest_sha256"],
        "candidate_draws": {candidate: expected_draws(candidate, EXPECTED_ROWS[slot]) for candidate in CANDIDATES},
        "zero_fit": True,
    }


@contextmanager
def zero_fit():
    counter = {"sampled_forest": 0, "sklearn_forest": 0, "preprocessor": 0, "parent_qrf": 0}
    def reject(name):
        def inner(*args, **kwargs):
            counter[name] += 1
            raise ValueError("cold inference forbids fit")
        return inner
    with (
        patch.object(SampledTimeForest, "fit", reject("sampled_forest")),
        patch.object(RandomForestRegressor, "fit", reject("sklearn_forest")),
        patch.object(ExtraTreesRegressor, "fit", reject("sklearn_forest")),
        patch.object(Preprocessor, "fit", reject("preprocessor")),
        patch.object(QRF, "fit", reject("parent_qrf")),
    ):
        yield counter


def fit(root: Path, slot: int, candidate: str):
    manifest = _registered(root)
    if candidate not in CANDIDATES or slot not in EXPECTED_ROWS:
        raise ValueError("unregistered v0.33 fit")
    attempted = len(list((root / "models" / candidate).glob("*/fit_intent.json")))
    if attempted >= 7:
        raise ValueError("seven-fit per-candidate budget exhausted")
    folder = root / "models" / candidate / str(slot)
    folder.mkdir(parents=True, exist_ok=False)
    arrays, info, source_path = payload(slot, "train")
    if set(arrays) != {"ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months"}:
        raise ValueError("v0.33 training handoff fields differ")
    if len(arrays["ids"]) != EXPECTED_ROWS[slot] or (arrays["reference_ns"] >= info["cutoff_ns"]).any() or (arrays["available_ns"] > info["cutoff_ns"]).any():
        raise ValueError("v0.33 training boundary differs")
    prep, transformed, diagnostic, prep_path = preprocessor(info, arrays, training=True)
    response = np.asarray(arrays["y"], dtype=np.float64)
    common = {
        "slot": slot, "candidate": candidate, "candidate_id": CANDIDATES[candidate],
        "forest_protocol": FOREST_PROTOCOL, "oob_protocol": OOB_PROTOCOL,
        "parameters": PARAMETERS[candidate], "training_rows": len(response),
        "draws_per_tree": expected_draws(candidate, len(response)),
        "source_handoff_path": str(source_path.relative_to(REPOSITORY)), "source_handoff_sha256": sha(source_path),
        "training_ids": array_identity(arrays["ids"]), "raw_matrix": array_identity(arrays["numeric"]),
        "transformed_float32_matrix": array_identity(transformed), "raw_target": array_identity(response),
        "preprocessor_path": str(prep_path.relative_to(REPOSITORY)), "preprocessor_sha256": sha(prep_path),
        "preprocessor_fit": False, "worker_sources": source_identity(), "environment": environment(),
        "manifest_sha256": sha(root / "manifest.json"),
    }
    write(folder / "fit_intent.json", common)
    start = time.perf_counter()
    model = SampledTimeForest(candidate).fit(transformed, response, arrays["ids"].tolist())
    model.training_months = arrays["training_months"]
    fit_seconds = time.perf_counter() - start
    joblib.dump(model, folder / "forest.joblib", compress=3)
    attachment_start = time.perf_counter()
    attachment, certificate = derive_attachment(model, transformed)
    np.savez_compressed(folder / "oob_attachment.npz", **attachment)
    attachment_seconds = time.perf_counter() - attachment_start
    if candidate == "A":
        parent, _ = _v26_parent(slot)
        parent_draws = np.asarray(parent.forest.estimators_samples_, dtype=np.int32)
        draw_identity = bool(np.array_equal(attachment["draws"], parent_draws))
        if not draw_identity:
            raise ValueError("candidate A bootstrap draws differ from V26A under equal draw parameters")
    else:
        draw_identity = None
    bundle = {
        **common, "forest_sha256": sha(folder / "forest.joblib"),
        "attachment_sha256": sha(folder / "oob_attachment.npz"), "attachment_certificate": certificate,
        "support": [float(response.min()), float(response.max())],
        "training_partition_audit": model.training_partition_audit,
        "training_transform_diagnostic": diagnostic, "fit_seconds": fit_seconds,
        "attachment_seconds": attachment_seconds, "model_bytes": (folder / "forest.joblib").stat().st_size,
        "attachment_bytes": (folder / "oob_attachment.npz").stat().st_size,
        "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "v26a_draws_exact": draw_identity,
    }
    write(folder / "bundle.json", bundle)
    write(folder / "fit_record.json", {
        "status": "COMPLETED", "candidate": candidate, "candidate_id": CANDIDATES[candidate], "slot": slot,
        "bundle_sha256": sha(folder / "bundle.json"), "forest_sha256": bundle["forest_sha256"],
        "attachment_sha256": bundle["attachment_sha256"], "fit_seconds": fit_seconds,
        "attachment_seconds": attachment_seconds, "internal_trees": 256, "preprocessor_fits": 0,
    })


def restore(root: Path, slot: int, candidate: str):
    completion = root / ("final_models_complete.json" if slot == 12 else "development_models_complete.json")
    trusted = json.loads(completion.read_text(encoding="utf-8"))[candidate][str(slot)]
    folder = root / "models" / candidate / str(slot)
    if sha(folder / "bundle.json") != trusted:
        raise ValueError("untrusted v0.33 bundle")
    metadata = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if metadata["environment"] != environment() or metadata["worker_sources"] != source_identity():
        raise ValueError("v0.33 restored environment/source differs")
    if metadata["parameters"] != PARAMETERS[candidate] or metadata["candidate_id"] != CANDIDATES[candidate]:
        raise ValueError("v0.33 restored candidate definition differs")
    if sha(folder / "forest.joblib") != metadata["forest_sha256"] or sha(folder / "oob_attachment.npz") != metadata["attachment_sha256"]:
        raise ValueError("v0.33 model/attachment identity differs")
    model = joblib.load(folder / "forest.joblib")
    attachment = load_npz(folder / "oob_attachment.npz")
    arrays, info, _ = payload(slot, "train")
    prep, transformed, _, _ = preprocessor(info, arrays, training=True)
    if model.candidate != candidate or model.ids != arrays["ids"].tolist():
        raise ValueError("v0.33 restored model training identity differs")
    validate_attachment(model, transformed, attachment, rederive=True)
    if candidate == "A":
        parent, _ = _v26_parent(slot)
        if not np.array_equal(attachment["draws"], np.asarray(parent.forest.estimators_samples_, dtype=np.int32)):
            raise ValueError("restored A draws differ from V26A")
    return model, attachment, prep, metadata


def _prediction(model, attachment, transformed):
    median, mean, diagnostic = predict_oob(model, transformed, attachment)
    effective = np.asarray([row["effective_neighbors"] for row in diagnostic], dtype=np.float64)
    fallback = np.asarray([row["fallback_tree_count"] for row in diagnostic], dtype=np.int16)
    minimum = np.asarray([row["minimum_selected_count"] for row in diagnostic], dtype=np.int32)
    maximum = np.asarray([row["maximum_selected_count"] for row in diagnostic], dtype=np.int32)
    return median, mean, effective, fallback, minimum, maximum


def _invariance(model, attachment, transformed, expected):
    index_sets = (
        np.arange(len(transformed))[::-1],
        np.asarray([0, len(transformed) // 2, len(transformed) - 1]),
        np.asarray([len(transformed) // 2]),
    )
    for indices in index_sets:
        actual = _prediction(model, attachment, transformed[indices])
        if any(not np.array_equal(value, base[indices]) for value, base in zip(actual, expected, strict=True)):
            raise ValueError("v0.33 reverse/subset/single invariance failed")
    chunks = [_prediction(model, attachment, transformed[index:index + 127]) for index in range(0, len(transformed), 127)]
    for field, base in enumerate(expected):
        if not np.array_equal(np.concatenate([part[field] for part in chunks]), base):
            raise ValueError("v0.33 chunk invariance failed")


def predict(root: Path, slot: int, candidate: str, output: Path, cold: bool):
    _registered(root)
    with zero_fit() as counter:
        start = time.perf_counter()
        model, attachment, prep, metadata = restore(root, slot, candidate)
        load_seconds = time.perf_counter() - start
        arrays, info, source_path = payload(slot, "evaluation")
        if set(arrays) != {"ids", "numeric", "spout", "reference_ns"} or set(arrays["ids"].tolist()) & set(model.ids):
            raise ValueError("v0.33 evaluation handoff fields/IDs differ")
        if (arrays["reference_ns"] < info["cutoff_ns"]).any():
            raise ValueError("v0.33 evaluation as-of boundary differs")
        transformed, diagnostic = prep.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
        start = time.perf_counter()
        values = _prediction(model, attachment, transformed)
        prediction_seconds = time.perf_counter() - start
        median, mean, effective, fallback, minimum, maximum = values
        if (median < metadata["support"][0]).any() or (median > metadata["support"][1]).any():
            raise ValueError("v0.33 OOB median left raw target support")
        if cold:
            with np.load(root / "worker_predictions" / candidate / f"{slot}.npz", allow_pickle=False) as saved:
                expected_saved = (saved["median"], saved["mean"], saved["effective_neighbors"], saved["fallback_tree_count"], saved["minimum_selected_count"], saved["maximum_selected_count"])
                if saved["ids"].tolist() != arrays["ids"].tolist() or any(not np.array_equal(a, b) for a, b in zip(values, expected_saved, strict=True)):
                    raise ValueError("v0.33 cold prediction differs")
            _invariance(model, attachment, transformed, values)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ValueError("never overwrite v0.33 worker prediction")
            np.savez_compressed(output, ids=arrays["ids"], median=median, mean=mean, effective_neighbors=effective,
                                fallback_tree_count=fallback, minimum_selected_count=minimum, maximum_selected_count=maximum)
        receipt = {
            "slot": slot, "candidate": candidate, "candidate_id": CANDIDATES[candidate], "rows": len(transformed),
            "training_rows": metadata["training_rows"], "draws_per_tree": metadata["draws_per_tree"],
            "model_sha256": metadata["forest_sha256"], "attachment_sha256": metadata["attachment_sha256"],
            "tree_sha256": metadata["attachment_certificate"]["tree_sha256"],
            "draw_sha256": metadata["attachment_certificate"]["draw_sha256"],
            "full_mapping_sha256": metadata["attachment_certificate"]["full_mapping_sha256"],
            "oob_mapping_sha256": metadata["attachment_certificate"]["oob_mapping_sha256"],
            "selected_mapping_sha256": metadata["attachment_certificate"]["selected_mapping_sha256"],
            "load_seconds": load_seconds, "prediction_seconds": prediction_seconds,
            "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "effective_neighbors": {"minimum": float(effective.min()), "median": float(np.median(effective)), "maximum": float(effective.max())},
            "fallback_tree_fraction_mean": float(fallback.mean() / 256),
            "selected_leaf_size": {"minimum": int(minimum.min()), "median_of_minimum": float(np.median(minimum)), "maximum": int(maximum.max())},
            "evaluation_handoff_path": str(source_path.relative_to(REPOSITORY)), "evaluation_handoff_sha256": sha(source_path),
            "transform_diagnostic": diagnostic, "preprocessor_fit": False, "zero_fit": counter,
            "exact_checks": "full_reverse_chunk_subset_single" if cold else "full",
        }
        write(Path(str(output) + ".json"), receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("environment", "identity", "registration", "audit", "fit", "predict", "cold"))
    parser.add_argument("--root", type=Path); parser.add_argument("--slot", type=int)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES)); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "environment": print(json.dumps(environment(), sort_keys=True)); return
    if args.command == "identity": print(json.dumps(source_identity(), sort_keys=True)); return
    if args.command == "registration": print(json.dumps(registration(), sort_keys=True)); return
    root = args.root.resolve()
    if args.command == "audit": print(json.dumps(audit(root, args.slot), sort_keys=True)); return
    if args.command == "fit": fit(root, args.slot, args.candidate); return
    receipt = predict(root, args.slot, args.candidate, args.output.resolve(), cold=args.command == "cold")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
