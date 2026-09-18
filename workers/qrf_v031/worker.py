"""Locked v0.31 adapter for occurrence-pooled OOB response aggregation.

No model, tree, preprocessor, or calibration fit is performed.  The source
forests and the certified v0.29 OOB attachments are restored and revalidated;
only the response mass assigned to each selected member occurrence changes.
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

from oob_response import MODE_FULL_SAME_LEAF_FALLBACK, MODE_OOB, array_identity, load_npz, tree_structure_identity
from pooled_response import PROTOCOL, selected_member_diagnostics
from qrf_model import distribution_weights, lower_median


V26 = REPOSITORY / "local/runs/optimization-v0.26-qrf-partition-tests-r1"
V27 = REPOSITORY / "local/runs/optimization-v0.27-qrf-iron-and-leaf-recency-r1"
V29 = REPOSITORY / "local/runs/optimization-v0.29-oob-leaf-responses-r1"
TARGETS = {
    "A": {"target": "tap_iron", "unit": "tonne", "source": "V27I_ABS_QRF_DIRECT_IRON"},
    "B": {"target": "tap_time_len", "unit": "minutes", "source": "V26A_QRF_ABSOLUTE_SPLIT_TIME"},
}
ATTACHMENT_PROTOCOL = "QRF_FROZEN_FOREST_OOB_LEAF_RESPONSE_v029"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V27_ADAPTER = _load_module("qrf_v027_adapter_for_v031", V27_WORKER / "worker.py")
V29_ADAPTER = _load_module("qrf_v029_adapter_for_v031", V29_WORKER / "worker.py")


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
        "pooled_response": HERE / "pooled_response.py",
        "pooled_oob_reference": HERE / "pooled_oob_reference.py",
        "v29_adapter": V29_WORKER / "worker.py",
        "v29_oob_response": V29_WORKER / "oob_response.py",
        "v27_adapter": V27_WORKER / "worker.py",
        "iron_target_forest": V27_WORKER / "target_forest.py",
        "v26_adapter": V26_WORKER / "worker.py",
        "partition_forest": V26_WORKER / "partition_forest.py",
        "parent_qrf_model": FROZEN / "qrf_model.py",
        "preprocessing": FROZEN / "preprocessing.py",
        "pyproject": FROZEN / "pyproject.toml",
        "lock": FROZEN / "uv.lock",
    }
    return {key: {"path": str(path), "sha256": sha(path)} for key, path in files.items()}


def registration():
    return {
        "pooled_protocol": PROTOCOL,
        "attachment_protocol": ATTACHMENT_PROTOCOL,
        "targets": TARGETS,
        "legacy_tree_mass_switch": "resident_reference_only_not_used_for_new_rule_certification",
    }


def _registered(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered v0.31 worker environment/source differs")
    if manifest["worker_registration"] != registration():
        raise ValueError("registered v0.31 worker protocol/targets differ")
    return manifest


def _attachment_path(candidate: str, slot: int) -> Path:
    if candidate not in TARGETS:
        raise ValueError("registered v0.31 candidate A or B required")
    path = V29 / "oob_attachments" / candidate / f"{slot}.npz"
    if not path.is_file():
        raise ValueError(f"missing certified v0.29 OOB attachment: {path}")
    return path


def _v29_prediction_path(candidate: str, slot: int) -> Path:
    return V29 / "worker_predictions" / candidate / f"{slot}.npz"


def _restore_source(candidate: str, slot: int):
    folder, model, preprocessor, metadata, source_prediction = V29_ADAPTER._restore_source(slot, candidate)
    if len(model.forest.estimators_) != 256 or len(model.leaves) != 256:
        raise ValueError("v0.31 source forest must retain exactly 256 frozen trees")
    if metadata.get("candidate_id") not in {
        "V27I_ABS_QRF_DIRECT_IRON", "V26A_QRF_ABSOLUTE_SPLIT_TIME",
    }:
        raise ValueError("v0.31 source metadata candidate identity differs")
    return folder, model, preprocessor, metadata, source_prediction


def _train_transformed(candidate: str, slot: int):
    folder, model, preprocessor, metadata, source_prediction = _restore_source(candidate, slot)
    train, train_info = V29_ADAPTER._features(slot, "train")
    train_x, train_diagnostic = V29_ADAPTER._transform(preprocessor, train, train_info, True)
    return folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic


def _load_certified_attachment(candidate: str, slot: int, model, train_x):
    attachment = _attachment_path(candidate, slot)
    arrays, attachment_metadata = V29_ADAPTER._load_attachment(attachment, model, train_x)
    if attachment_metadata.get("protocol") != ATTACHMENT_PROTOCOL:
        raise ValueError("certified v0.29 attachment protocol differs")
    return attachment, arrays, attachment_metadata


def _lookup(arrays: dict):
    offsets = arrays["member_offsets"]
    lookup, fallback = {}, set()
    for index, (tree, node, mode) in enumerate(zip(
        arrays["leaf_trees"], arrays["leaf_nodes"], arrays["modes"], strict=True,
    )):
        key = (int(tree), int(node))
        if key in lookup:
            raise ValueError("duplicate tree/leaf in certified OOB attachment")
        lookup[key] = arrays["members"][offsets[index]:offsets[index + 1]].astype(np.int64, copy=False)
        if int(mode) == int(MODE_FULL_SAME_LEAF_FALLBACK):
            fallback.add(key)
        elif int(mode) != int(MODE_OOB):
            raise ValueError("unknown selected-member mode in certified OOB attachment")
    return lookup, fallback


def _predict_pooled(model, transformed: np.ndarray, arrays: dict):
    lookup, fallback_keys = _lookup(arrays)
    routed = np.column_stack([tree.apply(transformed) for tree in model.forest.estimators_])
    medians, legacy_medians, diagnostics = [], [], []
    response = np.asarray(model.y, dtype=np.float64)
    for row in routed:
        selected = [lookup[(tree, int(node))] for tree, node in enumerate(row)]
        fallback = [((tree, int(node)) in fallback_keys) for tree, node in enumerate(row)]
        value, diagnostic = selected_member_diagnostics(response, selected, fallback_modes=fallback)
        old_weights = distribution_weights(selected, len(response))
        legacy = lower_median(response, old_weights, selected)
        medians.append(value)
        legacy_medians.append(legacy)
        diagnostics.append(diagnostic)
    return (
        np.asarray(medians, dtype=np.float64),
        np.asarray(legacy_medians, dtype=np.float64),
        diagnostics,
    )


def _diagnostic_arrays(diagnostics: list[dict]):
    keys = (
        "tree_count", "minimum_selected_count", "maximum_selected_count",
        "S_occurrence_total", "distinct_selected_rows", "fallback_tree_count",
        "fallback_mass_numerator",
    )
    arrays = {
        key: np.asarray([item[key] for item in diagnostics], dtype=np.int64)
        for key in keys
    }
    for key in (
        "fallback_tree_fraction", "fallback_mass_fraction",
        "weight_l1_change_from_equal_tree", "effective_neighbors_new",
        "effective_neighbors_old_equal_tree", "maximum_weight_new",
    ):
        arrays[key] = np.asarray([item[key] for item in diagnostics], dtype=np.float64)
    return arrays


def _summary(diagnostics: list[dict]):
    if not diagnostics:
        raise ValueError("at least one query diagnostic is required")
    rows = {
        "query_count": len(diagnostics),
        "tree_count_exact": int({item["tree_count"] for item in diagnostics} == {256}),
        "minimum_selected_count_min": int(min(item["minimum_selected_count"] for item in diagnostics)),
        "maximum_selected_count_max": int(max(item["maximum_selected_count"] for item in diagnostics)),
        "S_occurrence_total_min": int(min(item["S_occurrence_total"] for item in diagnostics)),
        "S_occurrence_total_max": int(max(item["S_occurrence_total"] for item in diagnostics)),
        "distinct_selected_rows_min": int(min(item["distinct_selected_rows"] for item in diagnostics)),
        "distinct_selected_rows_max": int(max(item["distinct_selected_rows"] for item in diagnostics)),
        "fallback_tree_count_mean": float(np.mean([item["fallback_tree_count"] for item in diagnostics])),
        "fallback_tree_count_max": int(max(item["fallback_tree_count"] for item in diagnostics)),
        "fallback_mass_fraction_mean": float(np.mean([item["fallback_mass_fraction"] for item in diagnostics])),
        "fallback_mass_fraction_max": float(max(item["fallback_mass_fraction"] for item in diagnostics)),
        "weight_l1_change_mean": float(np.mean([item["weight_l1_change_from_equal_tree"] for item in diagnostics])),
        "weight_l1_change_max": float(max(item["weight_l1_change_from_equal_tree"] for item in diagnostics)),
        "effective_neighbors_new_min": float(min(item["effective_neighbors_new"] for item in diagnostics)),
        "effective_neighbors_new_max": float(max(item["effective_neighbors_new"] for item in diagnostics)),
        "effective_neighbors_old_min": float(min(item["effective_neighbors_old_equal_tree"] for item in diagnostics)),
        "effective_neighbors_old_max": float(max(item["effective_neighbors_old_equal_tree"] for item in diagnostics)),
        "maximum_weight_new_max": float(max(item["maximum_weight_new"] for item in diagnostics)),
    }
    return rows


def _legacy_endpoint_check(candidate: str, slot: int, ids, legacy_medians):
    path = _v29_prediction_path(candidate, slot)
    if not path.is_file():
        raise ValueError(f"missing v0.29 legacy endpoint for switch-back check: {path}")
    with np.load(path, allow_pickle=False) as saved:
        if saved["ids"].tolist() != list(ids):
            raise ValueError("v0.29 legacy endpoint IDs differ")
        if not np.array_equal(saved["median"], np.asarray(legacy_medians, dtype=np.float64)):
            raise ValueError("v0.31 switch-back does not reproduce the certified v0.29 endpoint")
    return sha(path)


def _write_npz(path: Path, arrays: dict, receipt: dict):
    if path.exists() or Path(str(path) + ".json").exists():
        raise ValueError("never overwrite v0.31 worker output")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    write(Path(str(path) + ".json"), {**receipt, "sha256": sha(path)})
    return sha(path)


def audit(root: Path, slot: int, candidate: str):
    _registered(root)
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("unregistered v0.31 slot/candidate")
    with V29_ADAPTER.zero_fit() as counter:
        (
            folder, model, preprocessor, metadata, source_prediction,
            train, train_info, train_x, train_diagnostic,
        ) = _train_transformed(candidate, slot)
        attachment, arrays, attachment_metadata = _load_certified_attachment(candidate, slot, model, train_x)
        before = tree_structure_identity(model.forest)
        if before != attachment_metadata.get("tree_structure_sha256"):
            raise ValueError("certified attachment tree structure differs from restored forest")
        if before != tree_structure_identity(model.forest):
            raise ValueError("source forest structure changed during v0.31 audit")
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
    return {
        "slot": slot,
        "candidate": candidate,
        "target": TARGETS[candidate],
        "source_folder": str(folder),
        "source_bundle_sha256": sha(folder / "bundle.json"),
        "source_forest_sha256": sha(folder / "forest.joblib"),
        "source_prediction_sha256": sha(source_prediction),
        "source_metadata": {
            "candidate_id": metadata.get("candidate_id"),
            "support": metadata.get("support"),
        },
        "training_ids": array_identity(train["ids"]),
        "training_response": array_identity(model.y),
        "training_transform": array_identity(train_x),
        "evaluation_ids": array_identity(evaluation["ids"]),
        "evaluation_transform": array_identity(evaluation_x),
        "attachment_path": str(attachment),
        "attachment_sha256": sha(attachment),
        "attachment_metadata_sha256": sha(Path(str(attachment) + ".json")),
        "attachment_arrays": {name: array_identity(arrays[name]) for name in sorted(arrays)},
        "attachment_tree_structure_sha256": attachment_metadata["tree_structure_sha256"],
        "source_tree_structure_sha256": before,
        "attachment_rederived_from_bootstrap_exact": True,
        "attachment_trees_checked": int(len(model.forest.estimators_)),
        "zero_fit": counter,
        "train_transform_diagnostic": train_diagnostic,
        "evaluation_transform_diagnostic": evaluation_diagnostic,
    }


def derive_predict(root: Path, slot: int, candidate: str, output: Path):
    _registered(root)
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("unregistered v0.31 slot/candidate")
    with V29_ADAPTER.zero_fit() as counter:
        (
            folder, model, preprocessor, metadata, source_prediction,
            train, train_info, train_x, train_diagnostic,
        ) = _train_transformed(candidate, slot)
        attachment, arrays, attachment_metadata = _load_certified_attachment(candidate, slot, model, train_x)
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        start = time.perf_counter()
        median, legacy_median, diagnostics = _predict_pooled(model, evaluation_x, arrays)
        seconds = time.perf_counter() - start
        legacy_sha = _legacy_endpoint_check(candidate, slot, evaluation["ids"], legacy_median)
        response_min, response_max = float(np.min(model.y)), float(np.max(model.y))
        if (median < response_min).any() or (median > response_max).any():
            raise ValueError("v0.31 pooled median left original response support")
        arrays_out = {
            "ids": evaluation["ids"],
            "median": median,
            "legacy_median": legacy_median,
            **_diagnostic_arrays(diagnostics),
        }
        receipt = {
            "slot": slot,
            "candidate": candidate,
            "candidate_id": "V31I_OOB_OCCURRENCE_POOL_BLEND" if candidate == "A" else "V31T_OOB_OCCURRENCE_POOL_TIME",
            "target": TARGETS[candidate],
            "rows": len(evaluation),
            "source_folder": str(folder),
            "source_bundle_sha256": digest_file(folder / "bundle.json"),
            "source_forest_sha256": digest_file(folder / "forest.joblib"),
            "source_prediction_sha256": digest_file(source_prediction),
            "source_tree_structure_sha256": tree_structure_identity(model.forest),
            "source_response": array_identity(model.y),
            "training_ids": array_identity(train["ids"]),
            "attachment_path": str(attachment),
            "attachment_sha256": attachment_metadata["sha256"],
            "attachment_metadata_sha256": digest_file(Path(str(attachment) + ".json")),
            "attachment_arrays": {name: array_identity(arrays[name]) for name in sorted(arrays)},
            "legacy_v29_endpoint_sha256": legacy_sha,
            "legacy_equal_tree_switch_back_exact": True,
            "pooled_occurrence_rule_isolated_from_legacy_median": True,
            "support": [response_min, response_max],
            "lower_boundary": int(np.count_nonzero(median == response_min)),
            "upper_boundary": int(np.count_nonzero(median == response_max)),
            "changed_from_legacy_count": int(np.count_nonzero(median != legacy_median)),
            "nonfinite_or_negative_pooled": int(np.count_nonzero(~np.isfinite(median) | (median < 0))),
            "diagnostics": _summary(diagnostics),
            "zero_fit": counter,
            "prediction_seconds": seconds,
            "training_transform_diagnostic": train_diagnostic,
            "evaluation_transform_diagnostic": evaluation_diagnostic,
            "train_input": train_info,
            "evaluation_input": evaluation_info,
            "exact_cold_checks": False,
        }
        _write_npz(output, arrays_out, receipt)
    return receipt


def cold(root: Path, slot: int, candidate: str, output: Path):
    _registered(root)
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("unregistered v0.31 slot/candidate")
    with V29_ADAPTER.zero_fit() as counter:
        (
            folder, model, preprocessor, metadata, source_prediction,
            train, train_info, train_x, train_diagnostic,
        ) = _train_transformed(candidate, slot)
        attachment, arrays, attachment_metadata = _load_certified_attachment(candidate, slot, model, train_x)
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        median, legacy_median, diagnostics = _predict_pooled(model, evaluation_x, arrays)
        saved = root / "worker_predictions" / candidate / f"{slot}.npz"
        if not saved.is_file():
            raise ValueError("missing frozen v0.31 pooled worker output for cold comparison")
        with np.load(saved, allow_pickle=False) as source:
            expected = {name: source[name] for name in source.files}
        actual = {
            "ids": evaluation["ids"],
            "median": median,
            "legacy_median": legacy_median,
            **_diagnostic_arrays(diagnostics),
        }
        if set(expected) != set(actual):
            raise ValueError("cold v0.31 prediction array fields differ")
        for name in sorted(actual):
            if not np.array_equal(expected[name], actual[name]):
                raise ValueError(f"cold v0.31 prediction differs: {name}")
        # Reverse, subset, single-row, and chunk independence against the exact
        # pooled integer-count rule.
        reverse = _predict_pooled(model, evaluation_x[::-1], arrays)[0]
        if not np.array_equal(reverse, median[::-1]):
            raise ValueError("cold v0.31 reverse-order invariance failed")
        middle = len(evaluation_x) // 2
        subset_index = np.asarray([0, middle, len(evaluation_x) - 1])
        subset = _predict_pooled(model, evaluation_x[subset_index], arrays)[0]
        if not np.array_equal(subset, median[subset_index]):
            raise ValueError("cold v0.31 subset invariance failed")
        single = _predict_pooled(model, evaluation_x[middle:middle + 1], arrays)[0]
        if not np.array_equal(single, median[middle:middle + 1]):
            raise ValueError("cold v0.31 single-row invariance failed")
        pieces = [
            _predict_pooled(model, evaluation_x[index:index + 127], arrays)[0]
            for index in range(0, len(evaluation_x), 127)
        ]
        if not np.array_equal(np.concatenate(pieces), median):
            raise ValueError("cold v0.31 chunk invariance failed")
        receipt = {
            "slot": slot,
            "candidate": candidate,
            "rows": len(evaluation),
            "match": True,
            "reverse_chunk_subset_single_exact": True,
            "attachment_rederived_from_bootstrap_exact": True,
            "source_tree_structure_sha256": tree_structure_identity(model.forest),
            "attachment_sha256": attachment_metadata["sha256"],
            "legacy_switch_back_exact": bool(np.array_equal(
                legacy_median,
                np.load(_v29_prediction_path(candidate, slot), allow_pickle=False)["median"],
            )),
            "zero_fit": counter,
            "nonfinite_or_negative_pooled": int(np.count_nonzero(~np.isfinite(median) | (median < 0))),
        }
        _write_npz(output, {"ids": evaluation["ids"], "median": median, "legacy_median": legacy_median}, receipt)
    return receipt


def digest_file(path: Path):
    return sha(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("environment", "identity", "registration", "audit", "derive-predict", "cold"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--candidate", choices=("A", "B"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "environment":
        print(json.dumps(environment(), sort_keys=True))
    elif arguments.command == "identity":
        print(json.dumps(source_identity(), sort_keys=True))
    elif arguments.command == "registration":
        print(json.dumps(registration(), sort_keys=True))
    elif arguments.command == "audit":
        print(json.dumps(audit(arguments.root.resolve(), arguments.slot, arguments.candidate), sort_keys=True))
    elif arguments.command == "derive-predict":
        print(json.dumps(derive_predict(arguments.root.resolve(), arguments.slot, arguments.candidate, arguments.output.resolve()), sort_keys=True))
    else:
        print(json.dumps(cold(arguments.root.resolve(), arguments.slot, arguments.candidate, arguments.output.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
