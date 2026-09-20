#!/usr/bin/env python3
"""Locked zero-fit worker for v0.35 time-leaf aggregation."""
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
V34_WORKER = HERE.parent / "qrf_v034"
V31_WORKER = HERE.parent / "qrf_v031"
V29_WORKER = HERE.parent / "qrf_v029"
V27_WORKER = HERE.parent / "qrf_v027"
V26_WORKER = HERE.parent / "qrf_v026"
FROZEN = HERE.parent / "qrf_v015"
sys.path[:0] = [str(HERE), str(V34_WORKER), str(V31_WORKER), str(V29_WORKER), str(V27_WORKER), str(V26_WORKER), str(FROZEN)]

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl

from leaf_point_bagging import derive_point_table as derive_parent_table
from oob_response import array_identity, tree_structure_identity
from time_leaf_location_vote import PROTOCOL, derive_midpoint_table, predict_lower_vote, predict_midpoint_mean


PARENT_RUN = REPOSITORY / "local/runs/optimization-v0.34-oob-leaf-median-bagging-r2"
ATTACHMENT_PROTOCOL = "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"
PARENT_TABLE_PROTOCOL = "FROZEN_OOB_LEAF_LOWER_MEDIAN_BAGGING_v034"
CANDIDATES = {"A": "V35A_OOB_MIDPOINT_LEAF_MEAN_TIME", "B": "V35B_OOB_LOWER_MEDIAN_OF_LEAF_POINTS_TIME"}
TARGET = {"target": "tap_time_len", "unit": "minutes", "source": "V26A_QRF_ABSOLUTE_SPLIT_TIME"}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    return module


V34_ADAPTER = _load_module("qrf_v034_adapter_for_v035", V34_WORKER / "worker.py")
V29_ADAPTER = V34_ADAPTER.V29_ADAPTER


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle: json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__}


def source_identity():
    files = {
        "adapter": HERE / "worker.py", "numeric_core": HERE / "time_leaf_location_vote.py",
        "v34_adapter": V34_WORKER / "worker.py", "v34_core": V34_WORKER / "leaf_point_bagging.py",
        "v31_adapter": V31_WORKER / "worker.py", "v29_adapter": V29_WORKER / "worker.py",
        "v29_oob_response": V29_WORKER / "oob_response.py", "v26_adapter": V26_WORKER / "worker.py",
        "partition_forest": V26_WORKER / "partition_forest.py", "qrf_model": FROZEN / "qrf_model.py",
        "preprocessing": FROZEN / "preprocessing.py", "lock": FROZEN / "uv.lock",
    }
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in files.items()}


def registration():
    return {"protocol": PROTOCOL, "attachment_protocol": ATTACHMENT_PROTOCOL,
            "parent_table_protocol": PARENT_TABLE_PROTOCOL, "candidates": CANDIDATES, "target": TARGET,
            "A": "median_interval_endpoints_exact_sum_divided_by_512",
            "B": "index_127_lower_median_of_256_parent_lower_points"}


def _registered(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered v0.35 worker environment/source differs")
    if manifest["worker_registration"] != registration(): raise ValueError("registered v0.35 worker protocol differs")


def _restore(slot): return V34_ADAPTER._restore("B", slot)


def _parent_table(slot, restored):
    path = PARENT_RUN / "leaf_point_tables/B" / f"{slot}.npz"
    metadata_path = Path(str(path) + ".json")
    with np.load(path, allow_pickle=False) as source: table = {name: source[name] for name in source.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha(path) != metadata["sha256"] or metadata["protocol"] != PARENT_TABLE_PROTOCOL:
        raise ValueError("v0.35 parent lower table certificate differs")
    rederived = derive_parent_table(restored[1].y, restored[10])
    if set(table) != set(rederived) or any(not np.array_equal(table[name], rederived[name]) for name in table):
        raise ValueError("v0.35 parent lower table differs from rederived v0.29 members")
    return path, table, metadata


def _parent_raw(slot, ids):
    path = PARENT_RUN / "worker_predictions/B" / f"{slot}.npz"
    with np.load(path, allow_pickle=False) as source:
        if source["ids"].tolist() != np.asarray(ids).tolist(): raise ValueError("v0.35 parent raw IDs differ")
        result = source["point_mean"].copy()
    return path, result


def _save_npz(path, arrays, metadata):
    path = Path(path)
    if path.exists() or Path(str(path) + ".json").exists(): raise ValueError("never overwrite v0.35 output")
    path.parent.mkdir(parents=True, exist_ok=True); np.savez(path, **arrays)
    write(Path(str(path) + ".json"), {**metadata, "sha256": sha(path)})


def _midpoint_certificate(slot, restored, table, parent_path, parent_metadata):
    folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic, attachment, arrays, attachment_metadata = restored
    counts, lower, upper = table["selected_counts"], table["lower_raw"], table["upper_raw"]
    return {
        "protocol": PROTOCOL, "slot": slot, "candidate": "A", "candidate_id": CANDIDATES["A"], "target": TARGET,
        "cutoff": train_info["cutoff"], "cutoff_ns": train_info["cutoff_ns"],
        "training_ids": array_identity(train["ids"]), "raw_response": array_identity(model.y),
        "transformed_float32": array_identity(train_x), "source_bundle_sha256": sha(folder / "bundle.json"),
        "source_forest_sha256": sha(folder / "forest.joblib"), "source_tree_structure_sha256": tree_structure_identity(model.forest),
        "attachment_path": str(attachment), "attachment_sha256": sha(attachment),
        "attachment_metadata_sha256": sha(Path(str(attachment) + ".json")),
        "attachment_arrays": {name: array_identity(arrays[name]) for name in sorted(arrays)},
        "parent_lower_table_path": str(parent_path), "parent_lower_table_sha256": sha(parent_path),
        "parent_lower_table_metadata_sha256": sha(Path(str(parent_path) + ".json")),
        "parent_lower_table_protocol": parent_metadata["protocol"],
        "table_arrays": {name: array_identity(table[name]) for name in sorted(table)}, "leaf_count": len(counts),
        "selected_count_minimum": int(counts.min()), "selected_count_maximum": int(counts.max()),
        "odd_leaf_count": int(np.count_nonzero(counts % 2)), "even_leaf_count": int(np.count_nonzero(counts % 2 == 0)),
        "nonzero_interval_count": int(np.count_nonzero(upper > lower)),
        "lower_equals_parent_table_exact": True, "upper_gte_lower": True, "odd_lower_equals_upper": True,
        "implementation_sha256": sha(HERE / "time_leaf_location_vote.py"), "fit": False,
        "supervised_response_statistic_created": True, "train_transform_diagnostic": train_diagnostic,
    }


def _midpoint_table(root, slot, restored, parent_path, parent_table, parent_metadata, *, persist):
    model, arrays = restored[1], restored[10]
    table = derive_midpoint_table(model.y, arrays)
    if not np.array_equal(table["lower_raw"], parent_table["lower_median_raw"]):
        raise ValueError("v0.35 lower endpoint differs from parent lower table")
    for field in ("leaf_trees", "leaf_nodes", "selected_counts"):
        if not np.array_equal(table[field], parent_table[field]): raise ValueError("v0.35 midpoint table identity differs")
    certificate = _midpoint_certificate(slot, restored, table, parent_path, parent_metadata)
    path = root / "midpoint_tables/A" / f"{slot}.npz"
    if persist: _save_npz(path, table, certificate)
    else:
        with np.load(path, allow_pickle=False) as source: saved = {name: source[name] for name in source.files}
        if set(saved) != set(table) or any(not np.array_equal(saved[name], table[name]) for name in table):
            raise ValueError("cold v0.35 midpoint table differs")
        metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8")); recorded = metadata.pop("sha256", None)
        if recorded != sha(path) or metadata != certificate: raise ValueError("cold v0.35 midpoint certificate differs")
    return path, table, certificate


def _predict(candidate, model, transformed, table):
    return predict_midpoint_mean(model.forest, transformed, table) if candidate == "A" else predict_lower_vote(model.forest, transformed, table)


def _diag_arrays(candidate, rows):
    common = {"raw_delta": np.asarray([row["raw_delta"] for row in rows], dtype=np.float64)}
    if candidate == "A":
        return {**common, "nonzero_interval_tree_count": np.asarray([row["nonzero_interval_tree_count"] for row in rows], dtype=np.int16),
                "mean_half_interval_width": np.asarray([row["mean_half_interval_width"] for row in rows], dtype=np.float64)}
    return {**common, "leaf_point_minimum": np.asarray([row["leaf_point_minimum"] for row in rows]),
            "leaf_point_maximum": np.asarray([row["leaf_point_maximum"] for row in rows]),
            "distinct_leaf_point_count": np.asarray([row["distinct_leaf_point_count"] for row in rows], dtype=np.int16)}


def _summary(candidate, rows):
    result = {"query_count": len(rows), "raw_delta": {"minimum": float(min(r["raw_delta"] for r in rows)),
              "median": float(np.median([r["raw_delta"] for r in rows])), "maximum": float(max(r["raw_delta"] for r in rows))}}
    if candidate == "A":
        result["nonzero_interval_tree_count"] = {"minimum": int(min(r["nonzero_interval_tree_count"] for r in rows)),
              "median": float(np.median([r["nonzero_interval_tree_count"] for r in rows])), "maximum": int(max(r["nonzero_interval_tree_count"] for r in rows))}
        result["mean_half_interval_width"] = {"minimum": float(min(r["mean_half_interval_width"] for r in rows)),
              "median": float(np.median([r["mean_half_interval_width"] for r in rows])), "maximum": float(max(r["mean_half_interval_width"] for r in rows))}
    else:
        result["distinct_leaf_point_count"] = {"minimum": int(min(r["distinct_leaf_point_count"] for r in rows)),
              "median": float(np.median([r["distinct_leaf_point_count"] for r in rows])), "maximum": int(max(r["distinct_leaf_point_count"] for r in rows))}
    return result


def audit(root, slot, candidate):
    if slot not in range(6, 13) or candidate not in CANDIDATES: raise ValueError("v0.35 slot/candidate differs")
    _registered(root)
    with V29_ADAPTER.zero_fit() as counter:
        restored = _restore(slot); model, preprocessor = restored[1], restored[2]
        parent_path, parent_table, parent_metadata = _parent_table(slot, restored)
        table = derive_midpoint_table(model.y, restored[10]) if candidate == "A" else parent_table
        if candidate == "A" and not np.array_equal(table["lower_raw"], parent_table["lower_median_raw"]): raise ValueError("v0.35 P0 lower mismatch")
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        raw, parent, diagnostics = _predict(candidate, model, evaluation_x, table)
        parent_prediction_path, frozen_parent = _parent_raw(slot, evaluation["ids"])
        if not np.array_equal(parent, frozen_parent): raise ValueError("v0.35 P0 parent raw regression differs")
    return {"slot": slot, "candidate": candidate, "candidate_id": CANDIDATES[candidate], "rows": len(evaluation["ids"]),
            "source_forest_sha256": sha(restored[0] / "forest.joblib"), "source_tree_structure_sha256": tree_structure_identity(model.forest),
            "attachment_sha256": sha(restored[9]), "attachment_rederived_from_bootstrap_exact": True,
            "parent_lower_table_path": str(parent_path), "parent_lower_table_sha256": sha(parent_path),
            "parent_worker_prediction_path": str(parent_prediction_path), "parent_raw_regression_exact": True,
            "table_preview": {name: array_identity(table[name]) for name in sorted(table)},
            "diagnostics": _summary(candidate, diagnostics), "zero_fit": counter,
            "evaluation_transform": array_identity(evaluation_x), "evaluation_transform_diagnostic": evaluation_diagnostic}


def derive_predict(root, slot, candidate, output):
    if slot not in range(6, 13) or candidate not in CANDIDATES: raise ValueError("v0.35 slot/candidate differs")
    _registered(root)
    with V29_ADAPTER.zero_fit() as counter:
        restored = _restore(slot); model, preprocessor = restored[1], restored[2]
        parent_path, parent_table, parent_metadata = _parent_table(slot, restored)
        midpoint_path = None
        if candidate == "A": midpoint_path, table, certificate = _midpoint_table(root, slot, restored, parent_path, parent_table, parent_metadata, persist=True)
        else: table = parent_table
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        start = time.perf_counter(); raw, parent, diagnostics = _predict(candidate, model, evaluation_x, table); seconds = time.perf_counter() - start
        parent_prediction_path, frozen_parent = _parent_raw(slot, evaluation["ids"])
        if not np.array_equal(parent, frozen_parent): raise ValueError("v0.35 parent raw regression differs")
        if candidate == "A" and (raw < parent).any(): raise ValueError("v0.35 A raw monotonicity failed")
        arrays_out = {"ids": evaluation["ids"], "raw": raw, "parent_raw": parent, **_diag_arrays(candidate, diagnostics)}
        receipt = {"slot": slot, "candidate": candidate, "candidate_id": CANDIDATES[candidate], "target": TARGET,
                   "rows": len(evaluation["ids"]), "source_forest_sha256": sha(restored[0] / "forest.joblib"),
                   "source_tree_structure_sha256": tree_structure_identity(model.forest), "attachment_sha256": sha(restored[9]),
                   "parent_lower_table_path": str(parent_path), "parent_lower_table_sha256": sha(parent_path),
                   "parent_worker_prediction_path": str(parent_prediction_path), "parent_worker_prediction_sha256": sha(parent_prediction_path),
                   "parent_raw_regression_exact": True, "new_midpoint_table_path": str(midpoint_path) if midpoint_path else None,
                   "new_midpoint_table_sha256": sha(midpoint_path) if midpoint_path else None,
                   "new_supervised_table_created": candidate == "A", "diagnostics": _summary(candidate, diagnostics),
                   "changed_from_parent_raw_count": int(np.count_nonzero(raw != parent)), "nonfinite_or_negative": int(np.count_nonzero(~np.isfinite(raw) | (raw < 0))),
                   "zero_fit": counter, "prediction_seconds": seconds, "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "evaluation_input": evaluation_info, "evaluation_transform_diagnostic": evaluation_diagnostic, "exact_cold_checks": False}
        _save_npz(output, arrays_out, receipt)
    return receipt


def cold(root, slot, candidate, output):
    if slot not in range(6, 13) or candidate not in CANDIDATES: raise ValueError("v0.35 slot/candidate differs")
    _registered(root)
    with V29_ADAPTER.zero_fit() as counter:
        restored = _restore(slot); model, preprocessor = restored[1], restored[2]
        parent_path, parent_table, parent_metadata = _parent_table(slot, restored)
        if candidate == "A": _, table, _ = _midpoint_table(root, slot, restored, parent_path, parent_table, parent_metadata, persist=False)
        else: table = parent_table
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        raw, parent, diagnostics = _predict(candidate, model, evaluation_x, table)
        _, frozen_parent = _parent_raw(slot, evaluation["ids"])
        if not np.array_equal(parent, frozen_parent): raise ValueError("cold v0.35 parent raw differs")
        saved_path = root / "worker_predictions" / candidate / f"{slot}.npz"
        with np.load(saved_path, allow_pickle=False) as source: expected = {name: source[name] for name in source.files}
        actual = {"ids": evaluation["ids"], "raw": raw, "parent_raw": parent, **_diag_arrays(candidate, diagnostics)}
        if set(actual) != set(expected) or any(not np.array_equal(actual[name], expected[name]) for name in actual): raise ValueError("cold v0.35 arrays differ")
        reverse = _predict(candidate, model, evaluation_x[::-1], table)[0]
        middle = len(evaluation_x) // 2; indices = np.asarray([0, middle, len(evaluation_x) - 1])
        subset = _predict(candidate, model, evaluation_x[indices], table)[0]
        single = _predict(candidate, model, evaluation_x[middle:middle + 1], table)[0]
        chunks = [_predict(candidate, model, evaluation_x[i:i + 127], table)[0] for i in range(0, len(evaluation_x), 127)]
        if not np.array_equal(reverse, raw[::-1]) or not np.array_equal(subset, raw[indices]) or not np.array_equal(single, raw[middle:middle + 1]) or not np.array_equal(np.concatenate(chunks), raw):
            raise ValueError("cold v0.35 reverse/chunk/subset/single invariance failed")
        receipt = {"slot": slot, "candidate": candidate, "rows": len(evaluation["ids"]), "match": True,
                   "parent_lower_table_rederived_exact": True, "midpoint_table_rederived_exact": candidate == "A",
                   "parent_raw_regression_exact": True, "reverse_chunk_subset_single_exact": True,
                   "zero_fit": counter, "nonfinite_or_negative": int(np.count_nonzero(~np.isfinite(raw) | (raw < 0)))}
        _save_npz(output, {"ids": evaluation["ids"], "raw": raw, "parent_raw": parent}, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("environment", "identity", "registration", "audit", "derive-predict", "cold"))
    parser.add_argument("--root", type=Path); parser.add_argument("--slot", type=int); parser.add_argument("--candidate", choices=("A", "B")); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "environment": print(json.dumps(environment(), sort_keys=True))
    elif args.command == "identity": print(json.dumps(source_identity(), sort_keys=True))
    elif args.command == "registration": print(json.dumps(registration(), sort_keys=True))
    elif args.command == "audit": print(json.dumps(audit(args.root.resolve(), args.slot, args.candidate), sort_keys=True))
    elif args.command == "derive-predict": print(json.dumps(derive_predict(args.root.resolve(), args.slot, args.candidate, args.output.resolve()), sort_keys=True))
    else: print(json.dumps(cold(args.root.resolve(), args.slot, args.candidate, args.output.resolve()), sort_keys=True))


if __name__ == "__main__": main()
