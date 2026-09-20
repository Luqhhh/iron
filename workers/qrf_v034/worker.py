#!/usr/bin/env python3
"""Locked zero-fit worker for v0.34 OOB leaf-point bagging."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
FROZEN = HERE.parent / "qrf_v015"
V26_WORKER = HERE.parent / "qrf_v026"
V27_WORKER = HERE.parent / "qrf_v027"
V29_WORKER = HERE.parent / "qrf_v029"
V31_WORKER = HERE.parent / "qrf_v031"
sys.path[:0] = [str(HERE), str(V31_WORKER), str(V29_WORKER), str(V27_WORKER), str(V26_WORKER), str(FROZEN)]

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl

from leaf_point_bagging import PROTOCOL, derive_point_table, predict as predict_points
from oob_response import array_identity, predict as predict_legacy, tree_structure_identity


V29 = REPOSITORY / "local/runs/optimization-v0.29-oob-leaf-responses-r1"
TARGETS = {
    "A": {"target": "tap_iron", "unit": "tonne", "source": "V27I_ABS_QRF_DIRECT_IRON"},
    "B": {"target": "tap_time_len", "unit": "minutes", "source": "V26A_QRF_ABSOLUTE_SPLIT_TIME"},
}
ATTACHMENT_PROTOCOL = "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"
CANDIDATES = {
    "A": "V34I_OOB_LEAF_MEDIAN_BAGGING_BLEND",
    "B": "V34T_OOB_LEAF_MEDIAN_BAGGING_TIME",
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V31_ADAPTER = _load_module("qrf_v031_adapter_for_v034", V31_WORKER / "worker.py")
V29_ADAPTER = V31_ADAPTER.V29_ADAPTER


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__}


def source_identity():
    files = {
        "adapter": HERE / "worker.py", "leaf_point_bagging": HERE / "leaf_point_bagging.py",
        "v31_adapter": V31_WORKER / "worker.py", "v29_adapter": V29_WORKER / "worker.py",
        "v29_oob_response": V29_WORKER / "oob_response.py", "v27_adapter": V27_WORKER / "worker.py",
        "iron_target_forest": V27_WORKER / "target_forest.py", "v26_adapter": V26_WORKER / "worker.py",
        "partition_forest": V26_WORKER / "partition_forest.py", "qrf_model": FROZEN / "qrf_model.py",
        "preprocessing": FROZEN / "preprocessing.py", "lock": FROZEN / "uv.lock",
    }
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in files.items()}


def registration():
    return {"protocol": PROTOCOL, "attachment_protocol": ATTACHMENT_PROTOCOL, "targets": TARGETS,
            "candidates": CANDIDATES, "cross_tree_aggregation": "exact_binary64_arithmetic_mean_256_leaf_lower_medians"}


def _registered(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered v0.34 worker environment/source differs")
    if manifest["worker_registration"] != registration():
        raise ValueError("registered v0.34 worker protocol differs")
    return manifest


def _restore(candidate, slot):
    restored = V31_ADAPTER._train_transformed(candidate, slot)
    folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic = restored
    attachment, arrays, attachment_metadata = V31_ADAPTER._load_certified_attachment(candidate, slot, model, train_x)
    if metadata.get("candidate_id") != TARGETS[candidate]["source"] or len(model.forest.estimators_) != 256:
        raise ValueError("v0.34 frozen source identity differs")
    return (*restored, attachment, arrays, attachment_metadata)


def _save_npz(path, arrays, metadata):
    path = Path(path)
    if path.exists() or Path(str(path) + ".json").exists():
        raise ValueError("never overwrite v0.34 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    write(Path(str(path) + ".json"), {**metadata, "sha256": sha(path)})


def _table_certificate(candidate, slot, restored, table):
    folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic, attachment, arrays, attachment_metadata = restored
    return {
        "protocol": PROTOCOL, "slot": slot, "candidate": candidate, "candidate_id": CANDIDATES[candidate],
        "target": TARGETS[candidate]["target"], "unit": TARGETS[candidate]["unit"],
        "cutoff": train_info["cutoff"], "cutoff_ns": train_info["cutoff_ns"],
        "training_ids": array_identity(train["ids"]), "raw_numeric": array_identity(train["numeric"]),
        "raw_spout": array_identity(train["spout"]), "raw_response": array_identity(model.y),
        "transformed_float32": array_identity(train_x), "preprocessor_training_ids": array_identity(np.asarray(preprocessor.training_ids)),
        "source_bundle_sha256": sha(folder / "bundle.json"), "source_forest_sha256": sha(folder / "forest.joblib"),
        "source_tree_structure_sha256": tree_structure_identity(model.forest),
        "actual_bootstrap_draws": attachment_metadata["arrays"]["draws"],
        "attachment_path": str(attachment), "attachment_sha256": sha(attachment),
        "attachment_metadata_sha256": sha(Path(str(attachment) + ".json")),
        "attachment_arrays": {name: array_identity(arrays[name]) for name in sorted(arrays)},
        "attachment_rederived_from_bootstrap_exact": True,
        "implementation_sha256": sha(HERE / "leaf_point_bagging.py"),
        "table_arrays": {name: array_identity(table[name]) for name in sorted(table)},
        "leaf_count": len(table["leaf_nodes"]),
        "selected_count_minimum": int(table["selected_counts"].min()),
        "selected_count_maximum": int(table["selected_counts"].max()),
        "leaf_point_minimum": float(table["lower_median_raw"].min()),
        "leaf_point_maximum": float(table["lower_median_raw"].max()),
        "lower_median_even_rule": "smaller_middle_value", "fit": False,
        "supervised_response_statistic_created": True, "train_transform_diagnostic": train_diagnostic,
    }


def _derive_table(root, candidate, slot, restored, *, persist):
    model, arrays = restored[1], restored[10]
    before_response = array_identity(model.y); before_attachment = {name: array_identity(arrays[name]) for name in sorted(arrays)}
    table = derive_point_table(model.y, arrays)
    if before_response != array_identity(model.y) or before_attachment != {name: array_identity(arrays[name]) for name in sorted(arrays)}:
        raise ValueError("v0.34 table derivation mutated frozen response or attachment")
    certificate = _table_certificate(candidate, slot, restored, table)
    path = root / "leaf_point_tables" / candidate / f"{slot}.npz"
    if persist:
        _save_npz(path, table, certificate)
    else:
        with np.load(path, allow_pickle=False) as saved:
            expected = {name: saved[name] for name in saved.files}
        if set(expected) != set(table) or any(not np.array_equal(expected[name], table[name]) for name in table):
            raise ValueError("cold v0.34 leaf point table differs")
        metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
        recorded_sha = metadata.pop("sha256", None)
        if sha(path) != recorded_sha or metadata != certificate:
            raise ValueError("cold v0.34 leaf point table certificate differs")
    return path, table, certificate


def _diagnostic_arrays(rows):
    return {
        "leaf_point_minimum": np.asarray([row["leaf_point_minimum"] for row in rows], dtype=np.float64),
        "leaf_point_maximum": np.asarray([row["leaf_point_maximum"] for row in rows], dtype=np.float64),
        "leaf_point_range": np.asarray([row["leaf_point_range"] for row in rows], dtype=np.float64),
        "leaf_point_standard_deviation": np.asarray([row["leaf_point_standard_deviation"] for row in rows], dtype=np.float64),
        "distinct_leaf_point_count": np.asarray([row["distinct_leaf_point_count"] for row in rows], dtype=np.int16),
    }


def _summary(rows):
    return {
        "query_count": len(rows),
        "leaf_point_range": {"minimum": float(min(r["leaf_point_range"] for r in rows)),
                             "median": float(np.median([r["leaf_point_range"] for r in rows])),
                             "maximum": float(max(r["leaf_point_range"] for r in rows))},
        "leaf_point_standard_deviation": {"minimum": float(min(r["leaf_point_standard_deviation"] for r in rows)),
                                           "median": float(np.median([r["leaf_point_standard_deviation"] for r in rows])),
                                           "maximum": float(max(r["leaf_point_standard_deviation"] for r in rows))},
        "distinct_leaf_point_count": {"minimum": int(min(r["distinct_leaf_point_count"] for r in rows)),
                                      "median": float(np.median([r["distinct_leaf_point_count"] for r in rows])),
                                      "maximum": int(max(r["distinct_leaf_point_count"] for r in rows))},
    }


def audit(root, slot, candidate):
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("v0.34 slot/candidate differs")
    _registered(root)
    with V29_ADAPTER.zero_fit() as counter:
        restored = _restore(candidate, slot)
        folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic, attachment, arrays, attachment_metadata = restored
        table = derive_point_table(model.y, arrays)
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
    return {"slot": slot, "candidate": candidate, "target": TARGETS[candidate],
            "source_folder": str(folder), "source_bundle_sha256": sha(folder / "bundle.json"),
            "source_forest_sha256": sha(folder / "forest.joblib"), "source_tree_structure_sha256": tree_structure_identity(model.forest),
            "training_ids": array_identity(train["ids"]), "training_response": array_identity(model.y),
            "training_transform": array_identity(train_x), "evaluation_ids": array_identity(evaluation["ids"]),
            "evaluation_transform": array_identity(evaluation_x), "attachment_sha256": sha(attachment),
            "attachment_metadata_sha256": sha(Path(str(attachment) + ".json")),
            "attachment_arrays": {name: array_identity(arrays[name]) for name in sorted(arrays)},
            "attachment_rederived_from_bootstrap_exact": True,
            "attachment_trees_checked": int(len(np.unique(arrays["leaf_trees"]))),
            "point_table_preview": {name: array_identity(table[name]) for name in sorted(table)},
            "zero_fit": counter, "train_transform_diagnostic": train_diagnostic, "evaluation_transform_diagnostic": evaluation_diagnostic}


def derive_predict(root, slot, candidate, output):
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("v0.34 slot/candidate differs")
    _registered(root)
    with V29_ADAPTER.zero_fit() as counter:
        restored = _restore(candidate, slot)
        folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic, attachment, arrays, attachment_metadata = restored
        table_path, table, certificate = _derive_table(root, candidate, slot, restored, persist=True)
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        start = time.perf_counter(); point_mean, diagnostics = predict_points(model.forest, evaluation_x, table); seconds = time.perf_counter() - start
        legacy, _, _ = predict_legacy(model, evaluation_x, arrays, model.training_months)
        legacy_sha = V31_ADAPTER._legacy_endpoint_check(candidate, slot, evaluation["ids"], legacy)
        support = [float(np.min(model.y)), float(np.max(model.y))]
        if not np.isfinite(point_mean).all() or (point_mean < support[0]).any() or (point_mean > support[1]).any():
            raise ValueError("v0.34 prediction left raw response support")
        arrays_out = {"ids": evaluation["ids"], "point_mean": point_mean, "legacy_median": legacy, **_diagnostic_arrays(diagnostics)}
        receipt = {"slot": slot, "candidate": candidate, "candidate_id": CANDIDATES[candidate], "target": TARGETS[candidate],
                   "rows": len(evaluation), "source_folder": str(folder), "source_bundle_sha256": sha(folder / "bundle.json"),
                   "source_forest_sha256": sha(folder / "forest.joblib"), "source_tree_structure_sha256": tree_structure_identity(model.forest),
                   "source_response": array_identity(model.y), "training_ids": array_identity(train["ids"]),
                   "attachment_path": str(attachment), "attachment_sha256": sha(attachment),
                   "attachment_metadata_sha256": sha(Path(str(attachment) + ".json")),
                   "leaf_point_table_path": str(table_path), "leaf_point_table_sha256": sha(table_path),
                   "leaf_point_table_metadata_sha256": sha(Path(str(table_path) + ".json")),
                   "legacy_v29_endpoint_sha256": legacy_sha, "legacy_switch_back_exact": True,
                   "support": support, "changed_from_legacy_count": int(np.count_nonzero(point_mean != legacy)),
                   "nonfinite_or_negative": int(np.count_nonzero(~np.isfinite(point_mean) | (point_mean < 0))),
                   "diagnostics": _summary(diagnostics), "zero_fit": counter, "prediction_seconds": seconds,
                   "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "training_transform_diagnostic": train_diagnostic, "evaluation_transform_diagnostic": evaluation_diagnostic,
                   "train_input": train_info, "evaluation_input": evaluation_info, "supervised_leaf_point_table_created": True,
                   "exact_cold_checks": False}
        _save_npz(output, arrays_out, receipt)
    return receipt


def cold(root, slot, candidate, output):
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("v0.34 slot/candidate differs")
    _registered(root)
    with V29_ADAPTER.zero_fit() as counter:
        restored = _restore(candidate, slot)
        folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic, attachment, arrays, attachment_metadata = restored
        table_path, table, certificate = _derive_table(root, candidate, slot, restored, persist=False)
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        point_mean, diagnostics = predict_points(model.forest, evaluation_x, table)
        legacy, _, _ = predict_legacy(model, evaluation_x, arrays, model.training_months)
        saved_path = root / "worker_predictions" / candidate / f"{slot}.npz"
        with np.load(saved_path, allow_pickle=False) as saved: expected = {name: saved[name] for name in saved.files}
        actual = {"ids": evaluation["ids"], "point_mean": point_mean, "legacy_median": legacy, **_diagnostic_arrays(diagnostics)}
        if set(expected) != set(actual) or any(not np.array_equal(expected[name], actual[name]) for name in actual):
            raise ValueError("cold v0.34 prediction arrays differ")
        reverse = predict_points(model.forest, evaluation_x[::-1], table)[0]
        middle = len(evaluation_x) // 2; indices = np.asarray([0, middle, len(evaluation_x) - 1])
        subset = predict_points(model.forest, evaluation_x[indices], table)[0]
        single = predict_points(model.forest, evaluation_x[middle:middle + 1], table)[0]
        chunks = [predict_points(model.forest, evaluation_x[i:i + 127], table)[0] for i in range(0, len(evaluation_x), 127)]
        if not np.array_equal(reverse, point_mean[::-1]) or not np.array_equal(subset, point_mean[indices]) or not np.array_equal(single, point_mean[middle:middle + 1]) or not np.array_equal(np.concatenate(chunks), point_mean):
            raise ValueError("cold v0.34 reverse/chunk/subset/single invariance failed")
        V31_ADAPTER._legacy_endpoint_check(candidate, slot, evaluation["ids"], legacy)
        receipt = {"slot": slot, "candidate": candidate, "rows": len(evaluation), "match": True,
                   "point_table_rederived_exact": True, "attachment_rederived_from_bootstrap_exact": True,
                   "reverse_chunk_subset_single_exact": True, "legacy_switch_back_exact": True,
                   "source_tree_structure_sha256": tree_structure_identity(model.forest),
                   "attachment_sha256": sha(attachment), "leaf_point_table_sha256": sha(table_path),
                   "zero_fit": counter, "nonfinite_or_negative": int(np.count_nonzero(~np.isfinite(point_mean) | (point_mean < 0)))}
        _save_npz(output, {"ids": evaluation["ids"], "point_mean": point_mean, "legacy_median": legacy}, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("environment", "identity", "registration", "audit", "derive-predict", "cold"))
    parser.add_argument("--root", type=Path); parser.add_argument("--slot", type=int); parser.add_argument("--candidate", choices=("A", "B")); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "environment": print(json.dumps(environment(), sort_keys=True))
    elif args.command == "identity": print(json.dumps(source_identity(), sort_keys=True))
    elif args.command == "registration": print(json.dumps(registration(), sort_keys=True))
    elif args.command == "audit": print(json.dumps(audit(args.root.resolve(), args.slot, args.candidate), sort_keys=True))
    elif args.command == "derive-predict": print(json.dumps(derive_predict(args.root.resolve(), args.slot, args.candidate, args.output.resolve()), sort_keys=True))
    else: print(json.dumps(cold(args.root.resolve(), args.slot, args.candidate, args.output.resolve()), sort_keys=True))


if __name__ == "__main__": main()
