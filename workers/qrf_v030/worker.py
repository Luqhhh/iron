"""Locked v0.30 adapter: append 768 warm-start trees to the certified V26A time forest.

Only candidate B is produced here.  Every append fit is registered once per
cutoff, verifies the 256-tree prefix byte-for-byte, and is rejected outright
during cold inference.  OOB leaf responses still come from the v0.29 core;
this adapter only registers the 1024-tree protocol around it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
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
V27_WORKER = HERE.parent / "qrf_v027"
V29_WORKER = HERE.parent / "qrf_v029"
sys.path[:0] = [str(HERE), str(V29_WORKER), str(V27_WORKER), str(V26_WORKER), str(FROZEN)]

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

import append_forest as append_module
from append_forest import (
    APPENDED_PARAMETERS,
    NEW_TREES,
    PARENT_CANDIDATE_ID,
    PARENT_PARAMETERS,
    PARENT_PROTOCOL,
    PARENT_TREES,
    PROTOCOL,
    TARGET,
    TOTAL_TREES,
    UNIT,
    AppendedTimeForest,
    bootstrap_draws,
    bootstrap_sha256,
    forest_state_sha256,
    full_leaf_mapping,
    ordered_training_identity,
    partition_sha256,
)
from oob_append import (
    ATTACHMENT_PROTOCOL,
    CORE_PROTOCOL,
    REGRESSION_TREES,
    RestrictedView,
    assert_prefix_equal,
    full_leaf_arrays,
    require_1024_identity,
)
from oob_response import array_identity as core_array_identity
from oob_response import derive_attachment, load_npz, predict, validate_attachment
from partition_forest import PARAMETERS as V26_PARAMETERS
from partition_forest import PartitionForest
from preprocessing import Preprocessor
from qrf_model import QRF


V26 = REPOSITORY / "local/runs/optimization-v0.26-qrf-partition-tests-r1"
V29 = REPOSITORY / "local/runs/optimization-v0.29-oob-leaf-responses-r1"
TARGETS = {"B": {"target": TARGET, "unit": UNIT, "source": PARENT_CANDIDATE_ID}}
SOURCE_RUNS = {"v26": str(V26), "v29": str(V29)}
SLOTS = range(6, 13)
MAX_APPEND_FITS = 7


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V26_ADAPTER = _load_module("qrf_v026_adapter_for_v030", V26_WORKER / "worker.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
        "append_forest": HERE / "append_forest.py",
        "oob_append": HERE / "oob_append.py",
        "v029_oob_core": V29_WORKER / "oob_response.py",
        "v026_adapter": V26_WORKER / "worker.py",
        "v026_partition_forest": V26_WORKER / "partition_forest.py",
        "parent_qrf_model": FROZEN / "qrf_model.py",
        "preprocessing": FROZEN / "preprocessing.py",
        "pyproject": FROZEN / "pyproject.toml",
        "lock": FROZEN / "uv.lock",
    }
    return {key: {"path": str(path), "sha256": sha(path)} for key, path in files.items()}


def registration():
    if V26_PARAMETERS["A"] != PARENT_PARAMETERS:
        raise ValueError("imported v0.26 A parameters differ from the frozen append registration")
    return {
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "parent_candidate_id": PARENT_CANDIDATE_ID,
        "parent_trees": PARENT_TREES,
        "total_trees": TOTAL_TREES,
        "new_trees": NEW_TREES,
        "parent_parameters": PARENT_PARAMETERS,
        "parameters": APPENDED_PARAMETERS,
        "attachment_protocol": ATTACHMENT_PROTOCOL,
        "oob_core_protocol": CORE_PROTOCOL,
        "target": TARGET,
        "unit": UNIT,
        "fit_calls_per_slot": 1,
        "fit_calls_total": MAX_APPEND_FITS,
        "new_trees_per_slot": NEW_TREES,
        "reused_parent_trees_per_model": PARENT_TREES,
        "warm_start": True,
    }


def _registered(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered v0.30 worker environment/source differs")
    if manifest["append_registration"] != registration() or manifest["candidate_targets"] != TARGETS:
        raise ValueError("registered v0.30 append protocol/targets differ")
    if manifest["source_runs"] != SOURCE_RUNS:
        raise ValueError("registered v0.30 source runs differ")
    return manifest


def _completion(slot):
    name = "final_models_complete.json" if slot == 12 else "development_models_complete.json"
    return json.loads((V26 / name).read_text(encoding="utf-8"))["A"][str(slot)]


def _restore_parent(slot):
    folder = V26 / "models" / "A" / str(slot)
    model, preprocessor, metadata = V26_ADAPTER.restore(folder, _completion(slot), "A")
    if model.candidate != "A" or len(model.forest.estimators_) != PARENT_TREES:
        raise ValueError("restored V26A parent identity/tree count differs")
    if model.ids != preprocessor.training_ids:
        raise ValueError("restored V26A parent training identity differs")
    if metadata["candidate_id"] != PARENT_CANDIDATE_ID or metadata["protocol"] != PARENT_PROTOCOL:
        raise ValueError("restored V26A parent registration differs")
    return folder, model, preprocessor, metadata


def _features(root, slot, kind):
    path = root / "features" / str(slot) / f"{kind}.npz"
    info = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if sha(path) != info["sha256"]:
        raise ValueError("v0.30 worker input identity changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if arrays["ids"].tolist() != info["ids"]:
        raise ValueError("v0.30 worker input IDs differ")
    return arrays, info


def _transform(preprocessor, arrays, info, training):
    if training and preprocessor.training_ids != arrays["ids"].tolist():
        raise ValueError("frozen preprocessor/model training IDs differ")
    transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    if transformed.dtype != np.float32 or not np.isfinite(transformed).all():
        raise ValueError("frozen transform is not finite float32")
    return transformed, diagnostic


def _training_boundary(arrays, info, expected_fields):
    if set(arrays) != set(expected_fields):
        raise ValueError("v0.30 training payload fields differ")
    if (arrays["reference_ns"] >= info["cutoff_ns"]).any() or (arrays["available_ns"] > info["cutoff_ns"]).any():
        raise ValueError("v0.30 training payload violates the registered cutoff")
    response = np.asarray(arrays["y"], dtype=np.float64)
    if not np.isfinite(response).all() or (response < 0).any():
        raise ValueError("raw tap_time_len must remain finite nonnegative")
    return response


@contextmanager
def append_fit_counting():
    counter = {"random_forest_fit_attempts": 0}
    original = RandomForestRegressor.fit

    def counted(self, features, targets, *args, **kwargs):
        counter["random_forest_fit_attempts"] += 1
        if counter["random_forest_fit_attempts"] > 1:
            raise ValueError("exactly one warm-start append fit per slot is registered")
        return original(self, features, targets, *args, **kwargs)

    with patch.object(RandomForestRegressor, "fit", counted):
        yield counter


@contextmanager
def zero_fit():
    counter = {
        "append_fit_attempts": 0,
        "random_forest_fit_attempts": 0,
        "extra_trees_fit_attempts": 0,
        "partition_forest_fit_attempts": 0,
        "parent_qrf_fit_attempts": 0,
        "preprocessor_fit_attempts": 0,
    }

    def reject(name):
        def inner(*args, **kwargs):
            counter[name] += 1
            raise ValueError("cold inference forbids fit")
        return inner

    with (
        patch.object(append_module, "append_forest", reject("append_fit_attempts")),
        patch.object(RandomForestRegressor, "fit", reject("random_forest_fit_attempts")),
        patch.object(ExtraTreesRegressor, "fit", reject("extra_trees_fit_attempts")),
        patch.object(PartitionForest, "fit", reject("partition_forest_fit_attempts")),
        patch.object(QRF, "fit", reject("parent_qrf_fit_attempts")),
        patch.object(Preprocessor, "fit", reject("preprocessor_fit_attempts")),
    ):
        yield counter


def fit_ledger(root: Path):
    intents = sorted((root / "models" / "B").glob("*/append_intent.json"))
    records = sorted((root / "models" / "B").glob("*/fit_record.json"))
    attempted_trees = sum(json.loads(path.read_text(encoding="utf-8"))["new_trees"] for path in intents)
    completed_trees = sum(json.loads(path.read_text(encoding="utf-8"))["new_trees"] for path in records)
    return {
        "fit_attempted": len(intents),
        "fit_completed": len(records),
        "new_trees_attempted": attempted_trees,
        "new_trees_completed": completed_trees,
        "reused_parent_trees": PARENT_TREES * len(records),
        "trees_held": TOTAL_TREES * len(records),
        "slots": sorted(int(path.parent.name) for path in records),
        "budget": {"development": 6, "final": 1, "total": MAX_APPEND_FITS},
    }


def append_fit(root: Path, slot: int):
    _registered(root)
    if slot not in SLOTS:
        raise ValueError("unregistered v0.30 append slot")
    folder = root / "models" / "B" / str(slot)
    if folder.exists():
        raise ValueError("never overwrite a v0.30 append slot")
    if len(list((root / "models" / "B").glob("*/append_intent.json"))) >= MAX_APPEND_FITS:
        raise ValueError("seven-fit v0.30 warm-start append budget exhausted")
    folder.mkdir(parents=True, exist_ok=False)
    arrays, info = _features(root, slot, "train")
    response = _training_boundary(arrays, info, ("ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months"))
    parent_folder, parent, preprocessor, _ = _restore_parent(slot)
    if parent.ids != arrays["ids"].tolist() or not np.array_equal(parent.y, response):
        raise ValueError("certified V26A parent training identity differs from the frozen payload")
    train_x, diagnostic = _transform(preprocessor, arrays, info, True)
    identity = ordered_training_identity(arrays["ids"], train_x, response)
    intent = {
        "slot": slot,
        "candidate": "B",
        "protocol": PROTOCOL,
        "parent_candidate_id": PARENT_CANDIDATE_ID,
        "parent_folder": str(parent_folder.relative_to(REPOSITORY)),
        "parent_bundle_sha256": _completion(slot),
        "parent_forest_sha256": sha(parent_folder / "forest.joblib"),
        "parameters": APPENDED_PARAMETERS,
        "training_identity": identity,
        "new_trees": NEW_TREES,
        "total_trees": TOTAL_TREES,
        "per_slot_fit_limit": 1,
    }
    write(folder / "append_intent.json", intent)
    start = time.perf_counter()
    with append_fit_counting() as counter:
        forest, certificate = append_module.append_forest(
            parent, arrays["ids"], train_x, response,
            fit_hook=lambda estimator, features, targets: estimator.fit(features, targets),
        )
    append_seconds = time.perf_counter() - start
    if counter["random_forest_fit_attempts"] != 1:
        raise ValueError("warm-start append must consume exactly one fit call")
    partition = partition_sha256(forest, train_x)
    if partition[:PARENT_TREES] != certificate["partition_sha256"]:
        raise ValueError("appended forest changed the certified parent full-leaf mapping")
    leaves = full_leaf_mapping(parent, train_x) + full_leaf_mapping(forest, train_x)[PARENT_TREES:]
    if len(leaves) != TOTAL_TREES:
        raise ValueError("appended full-leaf mapping count differs")
    model = AppendedTimeForest(forest, arrays["ids"], response, arrays["training_months"], leaves, certificate)
    joblib.dump(model, folder / "forest.joblib", compress=3)
    bundle = {
        **intent,
        "protocol": PROTOCOL,
        "append_registration": registration(),
        "worker_sources": source_identity(),
        "environment": environment(),
        "manifest_sha256": sha(root / "manifest.json"),
        "input": info,
        "input_sha256": info["sha256"],
        "training_transform_diagnostic": diagnostic,
        "forest_sha256": sha(folder / "forest.joblib"),
        "certificate": certificate,
        "full_leaf_partition_sha256": partition,
        "support": model.prediction_support(),
        "new_tree_random_states": certificate["new_tree_random_states"],
        "append_seconds": append_seconds,
        "fit_calls": counter["random_forest_fit_attempts"],
        "model_bytes": (folder / "forest.joblib").stat().st_size,
        "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write(folder / "bundle.json", bundle)
    write(folder / "fit_record.json", {
        "status": "COMPLETED",
        "candidate": "B",
        "slot": slot,
        "protocol": PROTOCOL,
        "parent_bundle_sha256": _completion(slot),
        "bundle_sha256": sha(folder / "bundle.json"),
        "forest_sha256": bundle["forest_sha256"],
        "fit_calls": 1,
        "new_trees": NEW_TREES,
        "reused_parent_trees": PARENT_TREES,
        "total_trees": TOTAL_TREES,
        "append_seconds": append_seconds,
        "peak_memory_kib": bundle["peak_memory_kib"],
    })


def _restore_appended(root: Path, slot: int):
    folder = root / "models" / "B" / str(slot)
    bundle = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if bundle["protocol"] != PROTOCOL or bundle["append_registration"] != registration():
        raise ValueError("persisted v0.30 append protocol differs")
    if bundle["environment"] != environment() or bundle["worker_sources"] != source_identity():
        raise ValueError("persisted v0.30 append environment/source differs")
    if bundle["parameters"] != APPENDED_PARAMETERS or bundle["new_trees"] != NEW_TREES:
        raise ValueError("persisted v0.30 append parameters differ")
    if sha(folder / "forest.joblib") != bundle["forest_sha256"]:
        raise ValueError("persisted v0.30 appended forest identity differs")
    model = joblib.load(folder / "forest.joblib")
    require_1024_identity(model)
    if model.ids != bundle["input"]["ids"]:
        raise ValueError("persisted v0.30 appended training IDs differ")
    certificate = bundle["certificate"]
    states = forest_state_sha256(model.forest)
    if states != certificate["tree_state_sha256"] or states[:PARENT_TREES] != certificate["prefix"]["prefix_tree_state_sha256"]:
        raise ValueError("persisted v0.30 tree states differ from the append certificate")
    if bootstrap_sha256(bootstrap_draws(model.forest)) != certificate["bootstrap_sha256"]:
        raise ValueError("persisted v0.30 bootstrap draws differ from the append certificate")
    if certificate["parent_candidate_id"] != PARENT_CANDIDATE_ID or certificate["prefix"]["prefix_exact"] is not True:
        raise ValueError("persisted v0.30 append prefix was not verified")
    return model, bundle, folder


def _v29_evidence(slot: int):
    attachment = V29 / "oob_attachments" / "B" / f"{slot}.npz"
    metadata = json.loads(Path(str(attachment) + ".json").read_text(encoding="utf-8"))
    if metadata["protocol"] != CORE_PROTOCOL or sha(attachment) != metadata["sha256"]:
        raise ValueError("v0.29 OOB attachment identity/protocol differs")
    arrays = load_npz(attachment)
    prediction = V29 / "worker_predictions" / "B" / f"{slot}.npz"
    with np.load(prediction, allow_pickle=False) as saved:
        saved_prediction = {name: saved[name] for name in saved.files}
    return attachment, arrays, metadata, prediction, saved_prediction


def _save_attachment(path: Path, arrays, certificate):
    if path.exists() or Path(str(path) + ".json").exists():
        raise ValueError("never overwrite a v0.30 OOB attachment")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    metadata = {**certificate, "sha256": sha(path), "fit": False}
    write(Path(str(path) + ".json"), metadata)
    return metadata


def _load_attachment(path: Path, model, transformed):
    metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if sha(path) != metadata["sha256"] or metadata["protocol"] != ATTACHMENT_PROTOCOL:
        raise ValueError("persisted v0.30 OOB attachment identity/protocol differs")
    if metadata["core_protocol"] != CORE_PROTOCOL or metadata["trees"] != TOTAL_TREES:
        raise ValueError("persisted v0.30 OOB attachment registration differs")
    arrays = load_npz(path)
    validate_attachment(model, transformed, arrays, rederive=True)
    for name, identity in metadata["arrays"].items():
        if core_array_identity(arrays[name]) != identity:
            raise ValueError("persisted v0.30 OOB attachment array digest differs")
    return arrays, metadata


def _compare_saved_prediction(saved, ids, median, mean, label):
    if saved["ids"].tolist() != list(ids):
        raise ValueError(f"{label}: prediction IDs differ")
    if not np.array_equal(saved["median"], median) or not np.array_equal(saved["mean"], mean):
        raise ValueError(f"{label}: prediction values differ")
    return True


def _check_invariance(model, transformed, arrays, median, mean):
    index_sets = (
        np.arange(len(transformed))[::-1],
        np.asarray([0, len(transformed) // 2, len(transformed) - 1]),
        np.asarray([len(transformed) // 2]),
    )
    for indices in index_sets:
        current = predict(model, transformed[indices], arrays, model.training_months)
        if not np.array_equal(current[0], median[indices]) or not np.array_equal(current[1], mean[indices]):
            raise ValueError("1024-tree OOB reverse/subset/single invariance failed")
    chunks = [predict(model, transformed[index:index + 127], arrays, model.training_months)[:2]
              for index in range(0, len(transformed), 127)]
    if not np.array_equal(np.concatenate([part[0] for part in chunks]), median):
        raise ValueError("1024-tree OOB chunk median invariance failed")
    if not np.array_equal(np.concatenate([part[1] for part in chunks]), mean):
        raise ValueError("1024-tree OOB chunk mean invariance failed")


def _regression_prefix(model, parent, train_x):
    if not np.array_equal(bootstrap_draws(parent), bootstrap_draws(model.forest)[:REGRESSION_TREES]):
        raise ValueError("regression parent bootstrap prefix differs")
    if partition_sha256(model.forest, train_x)[:REGRESSION_TREES] != partition_sha256(parent, train_x):
        raise ValueError("regression parent full-leaf prefix differs")
    return RestrictedView(model, REGRESSION_TREES)


def _fallback_counts(diagnostics):
    return np.asarray([item["fallback_tree_count"] for item in diagnostics], dtype=np.int16)


def derive_predict(root: Path, slot: int, output: Path, cold: bool):
    _registered(root)
    if slot not in SLOTS:
        raise ValueError("unregistered v0.30 slot")
    with zero_fit() as counter:
        start = time.perf_counter()
        model, bundle, folder = _restore_appended(root, slot)
        load_seconds = time.perf_counter() - start
        train, train_info = _features(root, slot, "train")
        _training_boundary(train, train_info, ("ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months"))
        preprocessor, _ = V26_ADAPTER.original_preprocessor(train_info, train, require_training_ids=True)
        train_x, train_diagnostic = _transform(preprocessor, train, train_info, True)
        response = np.asarray(train["y"], dtype=np.float64)
        identity = ordered_training_identity(train["ids"], train_x, response)
        if identity != bundle["training_identity"]:
            raise ValueError("v0.30 append training identity differs from the frozen append run")
        if model.ids != train["ids"].tolist() or not np.array_equal(model.y, response):
            raise ValueError("v0.30 appended model training identity differs")
        attachment = root / "oob_attachments" / "B" / f"{slot}.npz"
        start = time.perf_counter()
        if cold:
            arrays, attachment_metadata = _load_attachment(attachment, model, train_x)
        else:
            arrays, certificate = derive_attachment(model, train_x)
            certificate.update({
                "protocol": ATTACHMENT_PROTOCOL,
                "core_protocol": CORE_PROTOCOL,
                "slot": slot,
                "candidate": "B",
                "target": TARGET,
                "unit": UNIT,
                "trees": TOTAL_TREES,
                "prefix_trees": PARENT_TREES,
                "cutoff": train_info["cutoff"],
                "cutoff_ns": train_info["cutoff_ns"],
                "source_bundle_sha256": sha(folder / "bundle.json"),
                "source_forest_sha256": sha(folder / "forest.joblib"),
                "source_training_ids": core_array_identity(train["ids"]),
                "source_response": core_array_identity(model.y),
                "source_transformed_float32": core_array_identity(train_x),
            })
            attachment_metadata = _save_attachment(attachment, arrays, certificate)
        attachment_seconds = time.perf_counter() - start
        _, v29_arrays, v29_metadata, v29_prediction_path, v29_saved = _v29_evidence(slot)
        prefix = assert_prefix_equal(arrays, v29_arrays, PARENT_TREES, label="v0.30/v0.29 time attachment")
        evaluation, evaluation_info = _features(root, slot, "evaluation")
        if (evaluation["reference_ns"] < evaluation_info["cutoff_ns"]).any():
            raise ValueError("v0.30 evaluation payload precedes the registered cutoff")
        eval_x, eval_diagnostic = _transform(preprocessor, evaluation, evaluation_info, False)
        start = time.perf_counter()
        median, mean, diagnostics = predict(model, eval_x, arrays, model.training_months)
        prediction_seconds = time.perf_counter() - start

        # Restricted-256 engineering regression, never a third candidate.
        parent_folder, parent, _, _ = _restore_parent(slot)
        view = _regression_prefix(model, parent, train_x)
        restricted_median, restricted_mean, _ = predict(view, eval_x, v29_arrays, model.training_months)
        _compare_saved_prediction(v29_saved, evaluation["ids"], restricted_median, restricted_mean, "restricted-256 v0.29 OOB")
        full_median, full_mean, _ = predict(view, eval_x, full_leaf_arrays(view), model.training_months)
        v26_prediction = V26 / "worker_predictions" / "A" / f"{slot}.npz"
        with np.load(v26_prediction, allow_pickle=False) as saved_v26:
            if saved_v26["ids"].tolist() != evaluation["ids"].tolist():
                raise ValueError("restricted-256 V26A prediction IDs differ")
            if not np.array_equal(saved_v26["median"], full_median) or not np.array_equal(saved_v26["mean"], full_mean):
                raise ValueError("restricted-256 full-leaf QRF does not reproduce the certified V26A output")

        saved_output = root / "worker_predictions" / "B" / f"{slot}.npz"
        regression_output = root / "worker_predictions" / "regression_256" / f"{slot}.npz"
        if cold:
            with np.load(saved_output, allow_pickle=False) as saved:
                _compare_saved_prediction(saved, evaluation["ids"], median, mean, "cold v0.30 1024-tree OOB")
                if not np.array_equal(saved["fallback_tree_counts"], _fallback_counts(diagnostics)):
                    raise ValueError("cold v0.30 fallback diagnostics differ")
            with np.load(regression_output, allow_pickle=False) as saved:
                _compare_saved_prediction(saved, evaluation["ids"], restricted_median, restricted_mean, "cold restricted-256")
            _check_invariance(model, eval_x, arrays, median, mean)
        else:
            for path in (output, saved_output, regression_output):
                if path.exists() or Path(str(path) + ".json").exists():
                    raise ValueError("never overwrite a v0.30 worker prediction")
            saved_output.parent.mkdir(parents=True, exist_ok=True)
            regression_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(saved_output, ids=evaluation["ids"], median=median, mean=mean,
                     fallback_tree_counts=_fallback_counts(diagnostics))
            np.savez(regression_output, ids=evaluation["ids"], median=restricted_median, mean=restricted_mean)

        fallback = _fallback_counts(diagnostics)
        receipt = {
            "slot": slot,
            "candidate": "B",
            "candidate_id": "V30B_OOB_TIME_1024",
            "protocol": PROTOCOL,
            "attachment_protocol": ATTACHMENT_PROTOCOL,
            "target": TARGET,
            "unit": UNIT,
            "rows": len(evaluation["ids"]),
            "trees": TOTAL_TREES,
            "new_trees": NEW_TREES,
            "source_bundle_sha256": sha(folder / "bundle.json"),
            "source_forest_sha256": sha(folder / "forest.joblib"),
            "source_response": core_array_identity(model.y),
            "attachment_sha256": attachment_metadata["sha256"],
            "attachment_metadata_sha256": sha(Path(str(attachment) + ".json")),
            "v29_attachment_sha256": v29_metadata["sha256"],
            "v29_prediction_sha256": sha(v29_prediction_path),
            "v26_prediction_sha256": sha(v26_prediction),
            "v29_prefix_exact": prefix,
            "restricted_256_oob_exact": True,
            "restricted_256_full_leaf_qrf_exact": True,
            "parent_bundle_sha256": _completion(slot),
            "parent_forest_sha256": sha(parent_folder / "forest.joblib"),
            "bootstrap_recovered_from_estimators_samples": True,
            "lower_boundary": int(np.count_nonzero(median == np.min(model.y))),
            "upper_boundary": int(np.count_nonzero(median == np.max(model.y))),
            "fallback_query_tree_count_quantiles": {
                str(q): float(np.quantile(fallback, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
            "fallback_query_tree_mean": float(fallback.mean()),
            "fallback_query_tree_max": int(fallback.max()),
            "load_seconds": load_seconds,
            "attachment_seconds": attachment_seconds,
            "prediction_seconds": prediction_seconds,
            "model_bytes": (folder / "forest.joblib").stat().st_size,
            "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "training_input": train_info,
            "evaluation_input": evaluation_info,
            "training_transform_diagnostic": train_diagnostic,
            "evaluation_transform_diagnostic": eval_diagnostic,
            "zero_fit": counter,
            "exact_cold_checks": cold,
        }
        write(Path(str(output) + ".json"), receipt)


def audit(root: Path, slot: int):
    """P0: zero-fit replay of the certified parent and the existing v0.29 evidence."""
    _registered(root)
    if slot not in SLOTS:
        raise ValueError("unregistered v0.30 audit slot")
    with zero_fit() as counter:
        parent_folder, parent, preprocessor, parent_metadata = _restore_parent(slot)
        train, train_info = _features(root, slot, "train")
        response = _training_boundary(train, train_info, ("ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months"))
        train_x, _ = _transform(preprocessor, train, train_info, True)
        identity = ordered_training_identity(train["ids"], train_x, response)
        if parent.ids != train["ids"].tolist() or not np.array_equal(parent.y, response):
            raise ValueError("P0 parent training identity differs")
        arrays, certificate = derive_attachment(parent, train_x)
        _, v29_arrays, v29_metadata, v29_prediction_path, v29_saved = _v29_evidence(slot)
        prefix = assert_prefix_equal(arrays, v29_arrays, PARENT_TREES, label="P0 v0.26A/v0.29 time attachment")
        evaluation, evaluation_info = _features(root, slot, "evaluation")
        eval_x, _ = _transform(preprocessor, evaluation, evaluation_info, False)
        full_median, full_mean, _ = predict(parent, eval_x, full_leaf_arrays(parent), parent.training_months)
        v26_prediction = V26 / "worker_predictions" / "A" / f"{slot}.npz"
        with np.load(v26_prediction, allow_pickle=False) as saved_v26:
            if saved_v26["ids"].tolist() != evaluation["ids"].tolist():
                raise ValueError("P0 certified V26A prediction IDs differ")
            if not np.array_equal(saved_v26["median"], full_median) or not np.array_equal(saved_v26["mean"], full_mean):
                raise ValueError("P0 full-leaf QRF does not reproduce the certified V26A output")
        restored_median, restored_mean, _ = predict(parent, eval_x, v29_arrays, parent.training_months)
        _compare_saved_prediction(v29_saved, evaluation["ids"], restored_median, restored_mean, "P0 parent/v0.29 OOB")
    return {
        "slot": slot,
        "parent_folder": str(parent_folder.relative_to(REPOSITORY)),
        "parent_bundle_sha256": _completion(slot),
        "parent_candidate_id": parent_metadata["candidate_id"],
        "parent_protocol": parent_metadata["protocol"],
        "training_identity": identity,
        "training_transform": core_array_identity(train_x),
        "certificate_preview": certificate,
        "v29_attachment_protocol": v29_metadata["protocol"],
        "v29_attachment_sha256": v29_metadata["sha256"],
        "v29_prediction_sha256": sha(v29_prediction_path),
        "v26_prediction_sha256": sha(v26_prediction),
        "v29_prefix_exact": prefix,
        "full_leaf_source_exact": True,
        "v29_oob_source_exact": True,
        "zero_fit": counter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "environment", "identity", "registration", "audit", "append-fit",
        "derive-predict", "cold", "fit-ledger",
    ))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "environment":
        print(json.dumps(environment(), sort_keys=True))
    elif arguments.command == "identity":
        print(json.dumps(source_identity(), sort_keys=True))
    elif arguments.command == "registration":
        print(json.dumps(registration(), sort_keys=True))
    elif arguments.command == "fit-ledger":
        print(json.dumps(fit_ledger(arguments.root.resolve()), sort_keys=True))
    elif arguments.command == "audit":
        print(json.dumps(audit(arguments.root.resolve(), arguments.slot), sort_keys=True))
    elif arguments.command == "append-fit":
        append_fit(arguments.root.resolve(), arguments.slot)
    else:
        derive_predict(arguments.root.resolve(), arguments.slot, arguments.output.resolve(), arguments.command == "cold")


if __name__ == "__main__":
    main()
