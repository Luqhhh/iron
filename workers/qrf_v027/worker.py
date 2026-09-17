"""Locked v0.27 worker for direct-iron QRF and frozen-time leaf recency."""
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
from partition_forest import PARAMETERS as PARENT_PARAMETERS
from partition_forest import PROTOCOL as PARENT_PROTOCOL
from partition_forest import PartitionForest
from leaf_recency import (
    HALF_LIFE_DAYS,
    PROTOCOL as LEAF_PROTOCOL,
    assert_all_ones_reproduces,
    predict_with_recency,
    recency60_weights,
)
from target_forest import PARAMETERS, PROTOCOL, TARGET, UNIT, IronTargetForest


V26 = REPOSITORY / "local/runs/optimization-v0.26-qrf-partition-tests-r1"


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


def source_identity():
    files = {
        "adapter": HERE / "worker.py",
        "target_forest": HERE / "target_forest.py",
        "leaf_recency": HERE / "leaf_recency.py",
        "parent_partition_forest": V26_WORKER / "partition_forest.py",
        "parent_adapter": V26_WORKER / "worker.py",
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
        raise ValueError("v0.27 worker input identity changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if arrays["ids"].tolist() != info["ids"]:
        raise ValueError("v0.27 worker input IDs differ")
    validate(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    return arrays, info


def original_preprocessor(info, arrays, *, require_training_ids):
    path = (REPOSITORY / info["original_preprocessor_path"]).resolve()
    if not path.is_relative_to(REPOSITORY.resolve()) or sha(path) != info["original_preprocessor_sha256"]:
        raise ValueError("original v0.15 preprocessor identity differs")
    preprocessor = Preprocessor.restore(json.loads(path.read_text(encoding="utf-8")))
    if preprocessor.columns != info["numeric_columns"]:
        raise ValueError("original v0.15 raw schema differs")
    if require_training_ids and preprocessor.training_ids != arrays["ids"].tolist():
        raise ValueError("original v0.15 preprocessor training IDs differ")
    digest = hashlib.sha256(json.dumps(preprocessor.transformed_columns).encode()).hexdigest()
    if digest != info["original_transformed_schema_sha256"]:
        raise ValueError("original v0.15 transformed schema differs")
    return preprocessor, path


def audit_inputs(root: Path, slot: int):
    result = {}
    for kind in ("train", "evaluation"):
        arrays, info = payload(root / "features" / str(slot) / f"{kind}.npz")
        preprocessor, path = original_preprocessor(info, arrays, require_training_ids=kind == "train")
        transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
        result[kind] = {
            "rows": len(arrays["ids"]),
            "ids": array_identity(arrays["ids"]),
            "numeric": array_identity(arrays["numeric"]),
            "spout": array_identity(arrays["spout"]),
            "reference_ns": array_identity(arrays["reference_ns"]),
            "available_ns": array_identity(arrays["available_ns"]) if kind == "train" else None,
            "raw_time_target": array_identity(arrays["y"]) if kind == "train" else None,
            "transformed_float32": array_identity(transformed),
            "preprocessor_path": str(path.relative_to(REPOSITORY)),
            "preprocessor_sha256": sha(path),
            "transformed_columns": len(preprocessor.transformed_columns),
            "transform_diagnostic": diagnostic,
        }
    return result


@contextmanager
def zero_fit():
    counter = {
        "iron_target_forest_fit_attempts": 0,
        "parent_partition_forest_fit_attempts": 0,
        "parent_qrf_fit_attempts": 0,
        "sklearn_forest_fit_attempts": 0,
        "preprocessor_fit_attempts": 0,
    }

    def reject(name):
        def inner(*args, **kwargs):
            counter[name] += 1
            raise ValueError("cold inference forbids fit")
        return inner

    with (
        patch.object(IronTargetForest, "fit", reject("iron_target_forest_fit_attempts")),
        patch.object(PartitionForest, "fit", reject("parent_partition_forest_fit_attempts")),
        patch.object(QRF, "fit", reject("parent_qrf_fit_attempts")),
        patch.object(RandomForestRegressor, "fit", reject("sklearn_forest_fit_attempts")),
        patch.object(ExtraTreesRegressor, "fit", reject("sklearn_forest_fit_attempts")),
        patch.object(Preprocessor, "fit", reject("preprocessor_fit_attempts")),
    ):
        yield counter


def _registered(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered v0.27 worker environment/source differs")
    if manifest["iron_forest_parameters"] != PARAMETERS or manifest["protocols"] != {"A": PROTOCOL, "B": LEAF_PROTOCOL}:
        raise ValueError("registered v0.27 model protocols differ")
    return manifest


def _target(root: Path, slot: int, ids):
    path = root / "targets" / str(slot) / "tap_iron.npz"
    info = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if sha(path) != info["sha256"] or info["target"] != TARGET or info["unit"] != UNIT:
        raise ValueError("registered iron target identity differs")
    with np.load(path, allow_pickle=False) as source:
        target_ids, response = source["ids"], source["y"]
    if target_ids.tolist() != list(ids) or info["ids"] != list(ids):
        raise ValueError("iron target/sample IDs differ")
    response = np.asarray(response, dtype=np.float64)
    if not np.isfinite(response).all() or (response < 0).any():
        raise ValueError("iron targets must be finite nonnegative tonnes")
    return response, info


def fit_a(root: Path, slot: int):
    _registered(root)
    if slot not in range(6, 13):
        raise ValueError("unregistered v0.27 iron slot")
    attempted = len(list((root / "models" / "A").glob("*/fit_intent.json")))
    if attempted >= 7:
        raise ValueError("seven-fit v0.27 iron forest budget exhausted")
    folder = root / "models" / "A" / str(slot)
    folder.mkdir(parents=True, exist_ok=False)
    arrays, info = payload(root / "features" / str(slot) / "train.npz")
    expected = {"ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months"}
    if set(arrays) != expected or (arrays["reference_ns"] >= info["cutoff_ns"]).any() or (arrays["available_ns"] > info["cutoff_ns"]).any():
        raise ValueError("v0.27 iron training boundary/payload differs")
    response, target_info = _target(root, slot, arrays["ids"].tolist())
    preprocessor, preprocessor_path = original_preprocessor(info, arrays, require_training_ids=True)
    transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    common = {
        "slot": slot,
        "candidate": "A",
        "candidate_id": "V27I_ABS_QRF_DIRECT_IRON",
        "protocol": PROTOCOL,
        "target": TARGET,
        "unit": UNIT,
        "estimator_class": "RandomForestRegressor",
        "splitter": "best",
        "criterion": "absolute_error",
        "parameters": PARAMETERS,
        "raw_target": "tap_iron_tonnes_finite_nonnegative_untransformed",
        "input": info,
        "input_sha256": info["sha256"],
        "target_input": target_info,
        "target_input_sha256": target_info["sha256"],
        "training_ids": array_identity(arrays["ids"]),
        "transformed_float32_matrix": array_identity(transformed),
        "raw_target_identity": array_identity(response),
        "worker_sources": source_identity(),
        "environment": environment(),
        "manifest_sha256": sha(root / "manifest.json"),
        "original_preprocessor_path": str(preprocessor_path.relative_to(REPOSITORY)),
        "original_preprocessor_sha256": sha(preprocessor_path),
        "preprocessor_fit": False,
        "sample_weight": False,
    }
    write(folder / "fit_intent.json", common)
    start = time.perf_counter()
    model = IronTargetForest().fit(transformed, response, info["ids"])
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
        "candidate": "A",
        "candidate_id": common["candidate_id"],
        "slot": slot,
        "bundle_sha256": sha(folder / "bundle.json"),
        "forest_sha256": bundle["forest_sha256"],
        "fit_seconds": bundle["fit_seconds"],
        "peak_memory_kib": bundle["peak_memory_kib"],
        "preprocessor_fits": 0,
        "internal_trees": len(model.forest.estimators_),
    })


def restore_a(folder: Path, trusted_sha: str):
    if sha(folder / "bundle.json") != trusted_sha:
        raise ValueError("untrusted v0.27 iron bundle")
    metadata = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if (
        metadata["environment"] != environment()
        or metadata["worker_sources"] != source_identity()
        or metadata["parameters"] != PARAMETERS
        or metadata["protocol"] != PROTOCOL
        or metadata["target"] != TARGET
        or metadata["unit"] != UNIT
        or metadata["candidate"] != "A"
        or metadata["candidate_id"] != "V27I_ABS_QRF_DIRECT_IRON"
        or metadata["preprocessor_fit"] is not False
        or metadata["sample_weight"] is not False
    ):
        raise ValueError("v0.27 iron bundle source/protocol/target differs")
    if sha(folder / "forest.joblib") != metadata["forest_sha256"]:
        raise ValueError("v0.27 iron forest identity differs")
    preprocessor_path = REPOSITORY / metadata["original_preprocessor_path"]
    if sha(preprocessor_path) != metadata["original_preprocessor_sha256"]:
        raise ValueError("restored iron preprocessor identity differs")
    model = joblib.load(folder / "forest.joblib")
    preprocessor = Preprocessor.restore(json.loads(preprocessor_path.read_text(encoding="utf-8")))
    if model.target != TARGET or model.unit != UNIT or model.ids != preprocessor.training_ids:
        raise ValueError("restored v0.27 iron target/training identity differs")
    return model, preprocessor, metadata


def _parent_completion(slot):
    name = "final_models_complete.json" if slot == 12 else "development_models_complete.json"
    return json.loads((V26 / name).read_text(encoding="utf-8"))["A"][str(slot)]


def restore_parent(slot):
    folder = V26 / "models" / "A" / str(slot)
    if sha(folder / "bundle.json") != _parent_completion(slot):
        raise ValueError("untrusted V26A parent bundle")
    metadata = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if (
        metadata["environment"] != environment()
        or metadata["parameters"] != PARENT_PARAMETERS["A"]
        or metadata["protocol"] != PARENT_PROTOCOL
        or metadata["candidate"] != "A"
        or metadata["candidate_id"] != "V26A_QRF_ABSOLUTE_SPLIT_TIME"
        or metadata["criterion"] != "absolute_error"
        or metadata["splitter"] != "best"
    ):
        raise ValueError("V26A parent model protocol differs")
    for item in metadata["worker_sources"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("V26A parent source identity differs")
    if sha(folder / "forest.joblib") != metadata["forest_sha256"]:
        raise ValueError("V26A parent forest identity differs")
    preprocessor_path = REPOSITORY / metadata["original_preprocessor_path"]
    if sha(preprocessor_path) != metadata["original_preprocessor_sha256"]:
        raise ValueError("V26A parent preprocessor identity differs")
    model = joblib.load(folder / "forest.joblib")
    preprocessor = Preprocessor.restore(json.loads(preprocessor_path.read_text(encoding="utf-8")))
    if model.candidate != "A" or model.ids != preprocessor.training_ids or len(model.forest.estimators_) != 256:
        raise ValueError("V26A parent restored identity differs")
    return model, preprocessor, metadata


def _check_invariance(predictor, transformed, median, mean):
    index_sets = (
        np.arange(len(transformed))[::-1],
        np.asarray([0, len(transformed) // 2, len(transformed) - 1]),
        np.asarray([len(transformed) // 2]),
    )
    for indices in index_sets:
        current = predictor(transformed[indices])
        if not np.array_equal(current[0], median[indices]) or not np.array_equal(current[1], mean[indices]):
            raise ValueError("v0.27 subset/order/single invariance failed")
    chunks = [predictor(transformed[index:index + 127])[:2] for index in range(0, len(transformed), 127)]
    if not np.array_equal(np.concatenate([chunk[0] for chunk in chunks]), median):
        raise ValueError("v0.27 chunk median invariance failed")
    if not np.array_equal(np.concatenate([chunk[1] for chunk in chunks]), mean):
        raise ValueError("v0.27 chunk mean invariance failed")


def predict_a(root: Path, slot: int, output: Path, cold: bool):
    _registered(root)
    completion = json.loads((root / ("final_models_complete.json" if slot == 12 else "development_models_complete.json")).read_text(encoding="utf-8"))
    with zero_fit() as counter:
        model, preprocessor, metadata = restore_a(root / "models" / "A" / str(slot), completion["A"][str(slot)])
        arrays, info = payload(root / "features" / str(slot) / "evaluation.npz")
        if set(info["ids"]) & set(model.ids) or (arrays["reference_ns"] < info["cutoff_ns"]).any():
            raise ValueError("v0.27 iron evaluation boundary differs")
        transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
        median, mean, neighbors = model.predict(transformed, model.training_months)
        if cold:
            with np.load(root / "worker_predictions" / "A" / f"{slot}.npz", allow_pickle=False) as saved:
                if saved["ids"].tolist() != arrays["ids"].tolist() or not np.array_equal(saved["median"], median) or not np.array_equal(saved["mean"], mean):
                    raise ValueError("cold v0.27 iron prediction differs")
            _check_invariance(lambda value: model.predict(value, model.training_months), transformed, median, mean)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ValueError("never overwrite v0.27 iron prediction")
            np.savez(output, ids=arrays["ids"], median=median, mean=mean)
        write(Path(str(output) + ".json"), {
            "slot": slot,
            "candidate": "A",
            "candidate_id": metadata["candidate_id"],
            "rows": len(transformed),
            "support": metadata["support"],
            "lower_boundary": int(np.count_nonzero(median == metadata["support"][0])),
            "upper_boundary": int(np.count_nonzero(median == metadata["support"][1])),
            "neighbors": neighbors,
            "input": info,
            "preprocessor_diagnostic": diagnostic,
            "preprocessor_fit": False,
            "zero_fit": counter,
            "exact_cold_checks": cold,
        })


def _weights(root, slot, model, train, info, cold):
    path = root / "weights" / str(slot) / "recency60.npz"
    metadata_path = Path(str(path) + ".json")
    values = recency60_weights(train["reference_ns"], info["cutoff_ns"])
    if model.ids != train["ids"].tolist():
        raise ValueError("V26A model/training IDs do not align for recency weights")
    if (train["available_ns"] > info["cutoff_ns"]).any():
        raise ValueError("recency donor label was unavailable at cutoff")
    if cold:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha(path) != metadata["sha256"]:
            raise ValueError("saved recency60 attachment identity differs")
        with np.load(path, allow_pickle=False) as source:
            if source["ids"].tolist() != model.ids or not np.array_equal(source["weights"], values):
                raise ValueError("cold recency60 attachment differs")
        return values, metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or metadata_path.exists():
        raise ValueError("never overwrite v0.27 recency attachment")
    np.savez(path, ids=train["ids"], reference_ns=train["reference_ns"], weights=values)
    metadata = {
        "sha256": sha(path),
        "slot": slot,
        "cutoff": info["cutoff"],
        "cutoff_ns": info["cutoff_ns"],
        "ids": model.ids,
        "target": "tap_time_len",
        "unit": "minutes",
        "protocol": LEAF_PROTOCOL,
        "half_life_days": HALF_LIFE_DAYS,
        "age_time": "reference_time",
        "availability_only_for_legality": True,
        "binary64_weights": array_identity(values),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "model_training_ids_aligned": True,
        "fit": False,
    }
    write(metadata_path, metadata)
    return values, metadata


def predict_b(root: Path, slot: int, output: Path, cold: bool):
    _registered(root)
    with zero_fit() as counter:
        model, preprocessor, parent_metadata = restore_parent(slot)
        train, train_info = payload(root / "features" / str(slot) / "train.npz")
        recency, weight_info = _weights(root, slot, model, train, train_info, cold)
        arrays, info = payload(root / "features" / str(slot) / "evaluation.npz")
        transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
        assert_all_ones_reproduces(model, transformed[: min(17, len(transformed))])
        predictor = lambda value: predict_with_recency(model, value, recency, model.training_months)
        median, mean, neighbors = predictor(transformed)
        parent_path = V26 / "worker_predictions" / "A" / f"{slot}.npz"
        with np.load(parent_path, allow_pickle=False) as parent:
            if parent["ids"].tolist() != arrays["ids"].tolist():
                raise ValueError("V26A parent prediction IDs differ")
        if cold:
            with np.load(root / "worker_predictions" / "B" / f"{slot}.npz", allow_pickle=False) as saved:
                if saved["ids"].tolist() != arrays["ids"].tolist() or not np.array_equal(saved["median"], median) or not np.array_equal(saved["mean"], mean):
                    raise ValueError("cold v0.27 leaf-recency prediction differs")
            _check_invariance(predictor, transformed, median, mean)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ValueError("never overwrite v0.27 leaf-recency prediction")
            np.savez(output, ids=arrays["ids"], median=median, mean=mean)
        write(Path(str(output) + ".json"), {
            "slot": slot,
            "candidate": "B",
            "candidate_id": "V27T_ABS_QRF_LEAF_RECENCY60",
            "rows": len(transformed),
            "support": parent_metadata["support"],
            "lower_boundary": int(np.count_nonzero(median == parent_metadata["support"][0])),
            "upper_boundary": int(np.count_nonzero(median == parent_metadata["support"][1])),
            "neighbors": neighbors,
            "weight_attachment": weight_info,
            "parent_bundle_sha256": _parent_completion(slot),
            "parent_tree_response_and_leaf_identity_frozen": True,
            "each_tree_mass_equal": True,
            "global_posthoc_reweighting": False,
            "all_ones_parent_distribution_check": True,
            "input": info,
            "preprocessor_diagnostic": diagnostic,
            "preprocessor_fit": False,
            "zero_fit": counter,
            "exact_cold_checks": cold,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("environment", "identity", "audit", "fit-a", "predict-a", "cold-a", "predict-b", "cold-b"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "environment":
        print(json.dumps(environment(), sort_keys=True))
    elif arguments.command == "identity":
        print(json.dumps(source_identity(), sort_keys=True))
    elif arguments.command == "audit":
        print(json.dumps(audit_inputs(arguments.root.resolve(), arguments.slot), sort_keys=True))
    elif arguments.command == "fit-a":
        fit_a(arguments.root.resolve(), arguments.slot)
    elif arguments.command in ("predict-a", "cold-a"):
        predict_a(arguments.root.resolve(), arguments.slot, arguments.output.resolve(), arguments.command == "cold-a")
    else:
        predict_b(arguments.root.resolve(), arguments.slot, arguments.output.resolve(), arguments.command == "cold-b")


if __name__ == "__main__":
    main()
