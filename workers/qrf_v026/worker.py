"""Locked v0.26 adapter for two partition forests and recomputed old-QRF support."""
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
sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(HERE))

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from preprocessing import Preprocessor, validate
from qrf_model import PARAMETERS as PARENT_PARAMETERS
from qrf_model import PROTOCOL as PARENT_MODEL_PROTOCOL
from qrf_model import QRF
from partition_forest import CANDIDATES, PARAMETERS, PARENT_PROTOCOL, PROTOCOL, PartitionForest


DEVELOPMENT = REPOSITORY / "local/runs/optimization-v0.15-opt32-r1"
FINAL = REPOSITORY / "local/runs/optimization-v0.15-v8-user-test-a-r1"


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
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }


def parent_source_identity():
    return {name: sha(FROZEN / name) for name in ("worker.py", "qrf_model.py", "preprocessing.py", "pyproject.toml", "uv.lock")}


def source_identity():
    files = {
        "adapter": HERE / "worker.py",
        "partition_forest": HERE / "partition_forest.py",
        "parent_qrf_model": FROZEN / "qrf_model.py",
        "preprocessing": FROZEN / "preprocessing.py",
        "pyproject": FROZEN / "pyproject.toml",
        "lock": FROZEN / "uv.lock",
    }
    return {key: {"path": str(path), "sha256": sha(path)} for key, path in files.items()}


def payload(path):
    path = Path(path)
    info = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if sha(path) != info["sha256"]:
        raise ValueError("worker input identity changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if arrays["ids"].tolist() != info["ids"]:
        raise ValueError("worker input IDs differ")
    validate(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    return arrays, info


def original_preprocessor(info, arrays, *, require_training_ids):
    path = (REPOSITORY / info["original_preprocessor_path"]).resolve()
    if not path.is_relative_to(REPOSITORY.resolve()) or sha(path) != info["original_preprocessor_sha256"]:
        raise ValueError("original preprocessor identity differs")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    preprocessor = Preprocessor.restore(metadata)
    if preprocessor.columns != info["numeric_columns"]:
        raise ValueError("original preprocessor raw schema differs")
    if require_training_ids and preprocessor.training_ids != arrays["ids"].tolist():
        raise ValueError("original preprocessor training IDs differ")
    transformed_schema_sha256 = hashlib.sha256(json.dumps(preprocessor.transformed_columns).encode()).hexdigest()
    if transformed_schema_sha256 != info["original_transformed_schema_sha256"]:
        raise ValueError("original transformed schema identity differs")
    return preprocessor, path


def audit_inputs(root: Path, slot: int):
    result = {}
    for kind in ("train", "evaluation"):
        arrays, info = payload(root / "features" / str(slot) / f"{kind}.npz")
        preprocessor, path = original_preprocessor(info, arrays, require_training_ids=kind == "train")
        transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
        if transformed.dtype != np.float32:
            raise ValueError("original transformed matrix is not float32")
        result[kind] = {
            "rows": len(arrays["ids"]),
            "ids": array_identity(arrays["ids"]),
            "numeric": array_identity(arrays["numeric"]),
            "spout": array_identity(arrays["spout"]),
            "reference_ns": array_identity(arrays["reference_ns"]),
            "available_ns": array_identity(arrays["available_ns"]) if kind == "train" else None,
            "raw_target": array_identity(arrays["y"]) if kind == "train" else None,
            "transformed_float32": array_identity(transformed),
            "preprocessor_path": str(path.relative_to(REPOSITORY)),
            "preprocessor_sha256": sha(path),
            "preprocessor_medians": array_identity(preprocessor.medians),
            "spout_vocabulary": preprocessor.vocabulary,
            "transformed_columns": len(preprocessor.transformed_columns),
            "transform_diagnostic": diagnostic,
        }
    if result["train"]["preprocessor_sha256"] != result["evaluation"]["preprocessor_sha256"]:
        raise ValueError("train/evaluation preprocessor identity differs")
    return result


@contextmanager
def zero_fit():
    counter = {"partition_forest_fit_attempts": 0, "parent_qrf_fit_attempts": 0, "sklearn_forest_fit_attempts": 0, "preprocessor_fit_attempts": 0}

    def reject_partition(*args, **kwargs):
        counter["partition_forest_fit_attempts"] += 1
        raise ValueError("cold inference forbids partition-forest fit")

    def reject_parent(*args, **kwargs):
        counter["parent_qrf_fit_attempts"] += 1
        raise ValueError("cold inference forbids parent-QRF fit")

    def reject_sklearn(*args, **kwargs):
        counter["sklearn_forest_fit_attempts"] += 1
        raise ValueError("cold inference forbids sklearn forest fit")

    def reject_preprocessor(*args, **kwargs):
        counter["preprocessor_fit_attempts"] += 1
        raise ValueError("cold inference forbids preprocessor fit")

    with (
        patch.object(PartitionForest, "fit", reject_partition),
        patch.object(QRF, "fit", reject_parent),
        patch.object(RandomForestRegressor, "fit", reject_sklearn),
        patch.object(ExtraTreesRegressor, "fit", reject_sklearn),
        patch.object(Preprocessor, "fit", reject_preprocessor),
    ):
        yield counter


def _registered(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered worker environment/source differs")
    if manifest["forest_parameters"] != PARAMETERS or manifest["protocol"] != PROTOCOL:
        raise ValueError("registered v0.26 forest protocol differs")
    return manifest


def fit(root: Path, slot: int, candidate: str):
    manifest = _registered(root)
    if candidate not in CANDIDATES or slot not in range(6, 13):
        raise ValueError("unregistered v0.26 fit candidate/slot")
    attempted = len(list((root / "models" / candidate).glob("*/fit_intent.json")))
    if attempted >= 7:
        raise ValueError("seven-fit per-candidate forest budget exhausted")
    folder = root / "models" / candidate / str(slot)
    folder.mkdir(parents=True, exist_ok=False)
    arrays, info = payload(root / "features" / str(slot) / "train.npz")
    expected = {"ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months"}
    if set(arrays) != expected or (arrays["reference_ns"] >= info["cutoff_ns"]).any() or (arrays["available_ns"] > info["cutoff_ns"]).any():
        raise ValueError("v0.26 training boundary/payload differs")
    response = np.asarray(arrays["y"], dtype=np.float64)
    if not np.isfinite(response).all() or (response < 0).any():
        raise ValueError("raw tap_time_len must remain finite nonnegative")
    preprocessor, preprocessor_path = original_preprocessor(info, arrays, require_training_ids=True)
    transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    definition = CANDIDATES[candidate]
    common = {
        "slot": slot,
        "candidate": candidate,
        "candidate_id": definition["candidate_id"],
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "estimator_class": definition["estimator_class"],
        "splitter": definition["splitter"],
        "criterion": definition["criterion"],
        "parameters": PARAMETERS[candidate],
        "raw_target": "tap_time_len_minutes_finite_nonnegative_untransformed",
        "input": info,
        "input_sha256": info["sha256"],
        "training_ids": array_identity(arrays["ids"]),
        "raw_matrix": array_identity(arrays["numeric"]),
        "transformed_float32_matrix": array_identity(transformed),
        "raw_target_identity": array_identity(response),
        "worker_sources": source_identity(),
        "environment": environment(),
        "manifest_sha256": sha(root / "manifest.json"),
        "original_preprocessor_path": str(preprocessor_path.relative_to(REPOSITORY)),
        "original_preprocessor_sha256": sha(preprocessor_path),
        "preprocessor_fit": False,
    }
    write(folder / "fit_intent.json", common)
    start = time.perf_counter()
    model = PartitionForest(candidate).fit(transformed, response, info["ids"])
    model.training_months = arrays["training_months"]
    joblib.dump(model, folder / "forest.joblib", compress=3)
    bundle = {
        **common,
        "forest_sha256": sha(folder / "forest.joblib"),
        "support": [float(model.y.min()), float(model.y.max())],
        "training_partition_audit": model.training_partition_audit,
        "training_transform_diagnostic": diagnostic,
        "fit_seconds": time.perf_counter() - start,
        "model_bytes": (folder / "forest.joblib").stat().st_size,
        "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write(folder / "bundle.json", bundle)
    write(folder / "fit_record.json", {
        "status": "COMPLETED",
        "candidate": candidate,
        "candidate_id": definition["candidate_id"],
        "slot": slot,
        "bundle_sha256": sha(folder / "bundle.json"),
        "forest_sha256": bundle["forest_sha256"],
        "fit_seconds": bundle["fit_seconds"],
        "peak_memory_kib": bundle["peak_memory_kib"],
        "preprocessor_fits": 0,
        "internal_trees": len(model.forest.estimators_),
    })


def restore(folder: Path, trusted_sha: str, candidate: str):
    if candidate not in CANDIDATES or sha(folder / "bundle.json") != trusted_sha:
        raise ValueError("untrusted v0.26 partition bundle")
    metadata = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    definition = CANDIDATES[candidate]
    if (
        metadata["environment"] != environment()
        or metadata["worker_sources"] != source_identity()
        or metadata["parameters"] != PARAMETERS[candidate]
        or metadata["protocol"] != PROTOCOL
        or metadata["parent_protocol"] != PARENT_PROTOCOL
        or metadata["candidate"] != candidate
        or metadata["candidate_id"] != definition["candidate_id"]
        or metadata["estimator_class"] != definition["estimator_class"]
        or metadata["criterion"] != definition["criterion"]
        or metadata["splitter"] != definition["splitter"]
        or metadata["preprocessor_fit"] is not False
    ):
        raise ValueError("v0.26 bundle source/protocol/estimator differs")
    if sha(folder / "forest.joblib") != metadata["forest_sha256"]:
        raise ValueError("v0.26 forest identity differs")
    preprocessor_path = REPOSITORY / metadata["original_preprocessor_path"]
    if sha(preprocessor_path) != metadata["original_preprocessor_sha256"]:
        raise ValueError("restored original preprocessor identity differs")
    model = joblib.load(folder / "forest.joblib")
    preprocessor = Preprocessor.restore(json.loads(preprocessor_path.read_text(encoding="utf-8")))
    if model.candidate != candidate or model.ids != preprocessor.training_ids or len(model.forest.estimators_) != 256:
        raise ValueError("restored v0.26 training/tree identity differs")
    if type(model.forest).__name__ != definition["estimator_class"]:
        raise ValueError("restored estimator class differs")
    if any(tree.criterion != definition["criterion"] or tree.splitter != definition["splitter"] for tree in model.forest.estimators_):
        raise ValueError("restored tree splitter/criterion differs")
    return model, preprocessor, metadata


def _check_invariance(model, transformed, median, mean):
    index_sets = (
        np.arange(len(transformed))[::-1],
        np.asarray([0, len(transformed) // 2, len(transformed) - 1]),
        np.asarray([len(transformed) // 2]),
    )
    for indices in index_sets:
        current_median, current_mean, _ = model.predict(transformed[indices])
        if not np.array_equal(current_median, median[indices]) or not np.array_equal(current_mean, mean[indices]):
            raise ValueError("partition forest subset/order/single invariance failed")
    chunks = [model.predict(transformed[index:index + 127])[:2] for index in range(0, len(transformed), 127)]
    if not np.array_equal(np.concatenate([chunk[0] for chunk in chunks]), median):
        raise ValueError("partition forest chunk median invariance failed")
    if not np.array_equal(np.concatenate([chunk[1] for chunk in chunks]), mean):
        raise ValueError("partition forest chunk mean invariance failed")


def predict(root: Path, slot: int, candidate: str, output: Path, cold: bool):
    _registered(root)
    completion_name = "final_models_complete.json" if slot == 12 else "development_models_complete.json"
    completed = json.loads((root / completion_name).read_text(encoding="utf-8"))
    with zero_fit() as counter:
        start = time.perf_counter()
        model, preprocessor, metadata = restore(root / "models" / candidate / str(slot), completed[candidate][str(slot)], candidate)
        load_seconds = time.perf_counter() - start
        arrays, info = payload(root / "features" / str(slot) / "evaluation.npz")
        expected = {"ids", "numeric", "spout", "reference_ns"}
        if set(arrays) != expected or set(info["ids"]) & set(model.ids) or (arrays["reference_ns"] < info["cutoff_ns"]).any():
            raise ValueError("v0.26 evaluation payload/boundary differs")
        transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
        start = time.perf_counter()
        median, mean, neighbors = model.predict(transformed, model.training_months)
        prediction_seconds = time.perf_counter() - start
        support = metadata["support"]
        if (median < support[0]).any() or (median > support[1]).any():
            raise ValueError("partition forest prediction left raw training support")
        if cold:
            with np.load(root / "worker_predictions" / candidate / f"{slot}.npz", allow_pickle=False) as saved:
                if saved["ids"].tolist() != arrays["ids"].tolist() or not np.array_equal(saved["median"], median) or not np.array_equal(saved["mean"], mean):
                    raise ValueError("cold partition prediction differs")
            _check_invariance(model, transformed, median, mean)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ValueError("never overwrite v0.26 worker prediction")
            np.savez(output, ids=arrays["ids"], median=median, mean=mean)
        write(Path(str(output) + ".json"), {
            "slot": slot,
            "candidate": candidate,
            "candidate_id": metadata["candidate_id"],
            "rows": len(transformed),
            "load_seconds": load_seconds,
            "prediction_seconds": prediction_seconds,
            "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "support": support,
            "lower_boundary": int(np.count_nonzero(median == support[0])),
            "upper_boundary": int(np.count_nonzero(median == support[1])),
            "neighbors": neighbors,
            "input": info,
            "preprocessor_diagnostic": diagnostic,
            "preprocessor_fit": False,
            "zero_fit": counter,
            "exact_cold_checks": cold,
        })


def _parent_root(slot: int):
    return FINAL if slot == 12 else DEVELOPMENT


def restore_parent(slot: int):
    root = _parent_root(slot)
    completed = json.loads((root / "models_complete.json").read_text(encoding="utf-8"))
    folder = root / "models" / str(slot)
    if sha(folder / "bundle.json") != completed[str(slot)]:
        raise ValueError("untrusted parent QRF bundle")
    metadata = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if (
        metadata["environment"] != environment()
        or metadata["sources"] != parent_source_identity()
        or metadata["parameters"] != PARENT_PARAMETERS
        or metadata["protocol"] != PARENT_MODEL_PROTOCOL
        or metadata["protocol"] != PARENT_PROTOCOL
    ):
        raise ValueError("parent QRF environment/source/protocol differs")
    if sha(folder / "forest.joblib") != metadata["forest_sha256"] or sha(folder / "preprocessor.json") != metadata["preprocessor_sha256"]:
        raise ValueError("parent QRF serialized identity differs")
    model = joblib.load(folder / "forest.joblib")
    preprocessor = Preprocessor.restore(json.loads((folder / "preprocessor.json").read_text(encoding="utf-8")))
    if model.ids != preprocessor.training_ids or len(model.forest.estimators_) != 256:
        raise ValueError("parent QRF restored identity differs")
    return model, preprocessor, metadata


def predict_parent(root: Path, slot: int, output: Path, cold: bool):
    _registered(root)
    with zero_fit() as counter:
        model, preprocessor, metadata = restore_parent(slot)
        arrays, info = payload(root / "features" / str(slot) / "evaluation.npz")
        transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
        median, mean, neighbors = model.predict(transformed, model.training_months)
        effective_neighbors = np.asarray([item["effective_neighbors"] for item in neighbors], dtype=np.float64)
        old_saved_path = _parent_root(slot) / "worker_predictions" / f"{slot}.npz"
        with np.load(old_saved_path, allow_pickle=False) as old_saved:
            if old_saved["ids"].tolist() != arrays["ids"].tolist() or not np.array_equal(old_saved["median"], median) or not np.array_equal(old_saved["mean"], mean):
                raise ValueError("recomputed parent QRF prediction differs from original saved output")
        if cold:
            with np.load(root / "old_qrf_predictions" / f"{slot}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["median"], median) or not np.array_equal(saved["mean"], mean) or not np.array_equal(saved["effective_neighbors"], effective_neighbors):
                    raise ValueError("cold parent support recomputation differs")
            _check_invariance(model, transformed, median, mean)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ValueError("never overwrite recomputed old-QRF prediction")
            np.savez(output, ids=arrays["ids"], median=median, mean=mean, effective_neighbors=effective_neighbors)
        write(Path(str(output) + ".json"), {
            "slot": slot,
            "rows": len(transformed),
            "source_bundle_sha256": sha(_parent_root(slot) / "models" / str(slot) / "bundle.json"),
            "source_saved_prediction_sha256": sha(old_saved_path),
            "support": metadata["support"],
            "gate_rows_spout1_neff_lt_500": int(np.count_nonzero((arrays["spout"].astype(str) == "1") & (effective_neighbors < 500.0))),
            "effective_neighbors": array_identity(effective_neighbors),
            "preprocessor_diagnostic": diagnostic,
            "zero_fit": counter,
            "recomputed_from_original_forest": True,
            "source_saved_prediction_exact": True,
            "exact_cold_checks": cold,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("environment", "identity", "audit", "fit", "predict", "cold", "old-predict", "old-cold"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--candidate", choices=("A", "B"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "environment":
        print(json.dumps(environment(), sort_keys=True))
    elif arguments.command == "identity":
        print(json.dumps(source_identity(), sort_keys=True))
    elif arguments.command == "audit":
        print(json.dumps(audit_inputs(arguments.root.resolve(), arguments.slot), sort_keys=True))
    elif arguments.command == "fit":
        fit(arguments.root.resolve(), arguments.slot, arguments.candidate)
    elif arguments.command in ("predict", "cold"):
        predict(arguments.root.resolve(), arguments.slot, arguments.candidate, arguments.output.resolve(), arguments.command == "cold")
    else:
        predict_parent(arguments.root.resolve(), arguments.slot, arguments.output.resolve(), arguments.command == "old-cold")


if __name__ == "__main__":
    main()
