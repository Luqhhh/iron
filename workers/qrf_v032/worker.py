"""Locked zero-fit v0.32 adapter for same-spout OOB response conditioning."""
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
sys.path[:0] = [str(HERE), str(V29_WORKER), str(V27_WORKER), str(V26_WORKER), str(FROZEN)]

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl

from oob_response import MODE_FULL_SAME_LEAF_FALLBACK, MODE_OOB, array_identity, tree_structure_identity
from same_spout import (
    PROTOCOL,
    QUERY_KNOWN,
    QUERY_MISSING,
    QUERY_UNKNOWN,
    conditioned_lower_median,
    exact_tokens,
    validate_training_spout,
)


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


V29_ADAPTER = _load_module("qrf_v029_adapter_for_v032", V29_WORKER / "worker.py")


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
        "same_spout": HERE / "same_spout.py",
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
        "same_spout_protocol": PROTOCOL,
        "attachment_protocol": ATTACHMENT_PROTOCOL,
        "targets": TARGETS,
        "member_rule": "known_query_intersection_if_nonempty_else_original_selected_set",
        "missing_or_unknown_query": "bypass",
        "aggregation": "original_equal_tree_equal_within_tree_lower_median",
    }


def _registered(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered v0.32 worker environment/source differs")
    if manifest["worker_registration"] != registration():
        raise ValueError("registered v0.32 worker protocol/targets differ")
    return manifest


def _attachment_path(candidate: str, slot: int) -> Path:
    if candidate not in TARGETS:
        raise ValueError("registered v0.32 candidate A or B required")
    path = V29 / "oob_attachments" / candidate / f"{slot}.npz"
    if not path.is_file():
        raise ValueError(f"missing certified v0.29 OOB attachment: {path}")
    return path


def _legacy_prediction_path(candidate: str, slot: int) -> Path:
    return V29 / "worker_predictions" / candidate / f"{slot}.npz"


def _restore(candidate: str, slot: int):
    folder, model, preprocessor, metadata, source_prediction = V29_ADAPTER._restore_source(slot, candidate)
    if len(model.forest.estimators_) != 256 or len(model.leaves) != 256:
        raise ValueError("v0.32 source forest must retain exactly 256 frozen trees")
    expected = TARGETS[candidate]["source"]
    if metadata.get("candidate_id") != expected:
        raise ValueError("v0.32 source metadata candidate identity differs")
    train, train_info = V29_ADAPTER._features(slot, "train")
    train_x, train_diagnostic = V29_ADAPTER._transform(preprocessor, train, train_info, True)
    training_spout = validate_training_spout(train["spout"], train["ids"], model.ids, preprocessor)
    attachment = _attachment_path(candidate, slot)
    arrays, attachment_metadata = V29_ADAPTER._load_attachment(attachment, model, train_x)
    if attachment_metadata.get("protocol") != ATTACHMENT_PROTOCOL:
        raise ValueError("certified v0.29 attachment protocol differs")
    return (
        folder, model, preprocessor, metadata, source_prediction, train, train_info,
        train_x, train_diagnostic, training_spout, attachment, arrays, attachment_metadata,
    )


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


def _predict(model, transformed, query_spout, training_spout, vocabulary, arrays):
    matrix = np.asarray(transformed)
    queries = exact_tokens(query_spout)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or len(matrix) != len(queries) or not np.isfinite(matrix).all():
        raise ValueError("v0.32 query transform/spout identity differs")
    lookup, fallback_keys = _lookup(arrays)
    routed = np.column_stack([tree.apply(matrix) for tree in model.forest.estimators_])
    medians, legacy_medians, diagnostics = [], [], []
    for row, spout in zip(routed, queries, strict=True):
        selected = [lookup[(tree, int(node))] for tree, node in enumerate(row)]
        fallback = [((tree, int(node)) in fallback_keys) for tree, node in enumerate(row)]
        value, legacy, diagnostic = conditioned_lower_median(
            model.y, selected, training_spout, spout, vocabulary,
            original_fallback_modes=fallback,
        )
        medians.append(value)
        legacy_medians.append(legacy)
        diagnostics.append(diagnostic)
    return np.asarray(medians), np.asarray(legacy_medians), diagnostics


def _diagnostic_arrays(diagnostics):
    integer_keys = (
        "query_state", "tree_count", "same_spout_nonempty_tree_count", "condition_fallback_tree_count",
        "original_oob_fallback_tree_count", "original_minimum_selected_count",
        "original_maximum_selected_count", "new_minimum_selected_count", "new_maximum_selected_count",
        "original_distinct_selected_rows", "new_distinct_selected_rows",
    )
    float_keys = (
        "original_cross_spout_mass", "new_cross_spout_mass", "original_effective_neighbors",
        "new_effective_neighbors", "original_maximum_weight", "new_maximum_weight", "prediction_delta",
    )
    result = {key: np.asarray([row[key] for row in diagnostics], dtype=np.int64) for key in integer_keys}
    result.update({key: np.asarray([row[key] for row in diagnostics], dtype=np.float64) for key in float_keys})
    result["prediction_changed"] = np.asarray([row["prediction_changed"] for row in diagnostics], dtype=np.uint8)
    result["query_token"] = np.asarray([row["query_token"] for row in diagnostics], dtype=str)
    return result


def _arrays_equal(left, right):
    """Exact array comparison, with NaN equality only for floating arrays."""
    first, second = np.asarray(left), np.asarray(right)
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    if first.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    return {str(q): float(np.quantile(array, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)}


def _summary(diagnostics):
    if not diagnostics:
        raise ValueError("at least one v0.32 query diagnostic is required")

    def group(rows):
        known = [item for item in rows if item["query_state"] == int(QUERY_KNOWN)]
        return {
            "query_count": len(rows),
            "known_query_count": len(known),
            "missing_query_count": sum(item["query_state"] == int(QUERY_MISSING) for item in rows),
            "unknown_query_count": sum(item["query_state"] == int(QUERY_UNKNOWN) for item in rows),
            "same_spout_nonempty_tree_fraction_mean": float(np.mean([
                item["same_spout_nonempty_tree_count"] / item["tree_count"] for item in known
            ])) if known else None,
            "condition_fallback_tree_count_quantiles": _quantiles([
                item["condition_fallback_tree_count"] for item in known
            ]),
            "original_oob_fallback_tree_count_quantiles": _quantiles([
                item["original_oob_fallback_tree_count"] for item in rows
            ]),
            "original_cross_spout_mass_mean": float(np.mean([
                item["original_cross_spout_mass"] for item in known
            ])) if known else None,
            "new_cross_spout_mass_mean": float(np.mean([
                item["new_cross_spout_mass"] for item in known
            ])) if known else None,
            "new_selected_count_min_quantiles": _quantiles([
                item["new_minimum_selected_count"] for item in rows
            ]),
            "new_selected_count_max_quantiles": _quantiles([
                item["new_maximum_selected_count"] for item in rows
            ]),
            "new_effective_neighbors_quantiles": _quantiles([
                item["new_effective_neighbors"] for item in rows
            ]),
            "prediction_delta_quantiles": _quantiles([item["prediction_delta"] for item in rows]),
            "prediction_changed_count": sum(item["prediction_changed"] for item in rows),
        }

    return {
        "overall": group(diagnostics),
        "by_query_token": {
            token: group([item for item in diagnostics if item["query_token"] == token])
            for token in sorted({item["query_token"] for item in diagnostics})
        },
    }


def _legacy_check(candidate, slot, ids, legacy):
    path = _legacy_prediction_path(candidate, slot)
    with np.load(path, allow_pickle=False) as saved:
        if saved["ids"].tolist() != list(ids) or not np.array_equal(saved["median"], legacy):
            raise ValueError("v0.32 disabled conditioning does not reproduce certified v0.29 endpoint")
    return sha(path)


def _write_npz(path, arrays, receipt):
    path = Path(path)
    if path.exists() or Path(str(path) + ".json").exists():
        raise ValueError("never overwrite v0.32 worker output")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    write(Path(str(path) + ".json"), {**receipt, "sha256": sha(path)})


def audit(root, slot, candidate):
    _registered(root)
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("unregistered v0.32 slot/candidate")
    with V29_ADAPTER.zero_fit() as counter:
        restored = _restore(candidate, slot)
        folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic, training_spout, attachment, arrays, attachment_metadata = restored
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        structure = tree_structure_identity(model.forest)
        if structure != attachment_metadata["tree_structure_sha256"]:
            raise ValueError("certified attachment tree structure differs from restored forest")
        vocabulary = [str(item) for item in preprocessor.vocabulary]
        states = [int(QUERY_MISSING if not str(value) else QUERY_KNOWN if str(value) in vocabulary else QUERY_UNKNOWN) for value in evaluation["spout"]]
    return {
        "slot": slot, "candidate": candidate, "target": TARGETS[candidate],
        "source_folder": str(folder), "source_bundle_sha256": sha(folder / "bundle.json"),
        "source_forest_sha256": sha(folder / "forest.joblib"), "source_prediction_sha256": sha(source_prediction),
        "source_candidate_id": metadata.get("candidate_id"), "source_tree_structure_sha256": structure,
        "training_ids": array_identity(train["ids"]), "training_spout": array_identity(training_spout),
        "training_response": array_identity(model.y), "training_transform": array_identity(train_x),
        "evaluation_ids": array_identity(evaluation["ids"]), "evaluation_spout": array_identity(exact_tokens(evaluation["spout"])),
        "evaluation_transform": array_identity(evaluation_x), "preprocessor_vocabulary": vocabulary,
        "query_state_counts": {"known": states.count(int(QUERY_KNOWN)), "missing": states.count(int(QUERY_MISSING)), "unknown": states.count(int(QUERY_UNKNOWN))},
        "attachment_path": str(attachment), "attachment_sha256": sha(attachment),
        "attachment_metadata_sha256": sha(Path(str(attachment) + ".json")),
        "attachment_arrays": {name: array_identity(arrays[name]) for name in sorted(arrays)},
        "attachment_rederived_from_bootstrap_exact": True, "attachment_trees_checked": len(model.forest.estimators_),
        "train_transform_diagnostic": train_diagnostic, "evaluation_transform_diagnostic": evaluation_diagnostic,
        "zero_fit": counter,
    }


def derive_predict(root, slot, candidate, output):
    _registered(root)
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("unregistered v0.32 slot/candidate")
    with V29_ADAPTER.zero_fit() as counter:
        restored = _restore(candidate, slot)
        folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic, training_spout, attachment, arrays, attachment_metadata = restored
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        before = {
            "tree": tree_structure_identity(model.forest), "response": array_identity(model.y),
            "attachment": {name: array_identity(arrays[name]) for name in sorted(arrays)},
        }
        start = time.perf_counter()
        median, legacy, diagnostics = _predict(
            model, evaluation_x, evaluation["spout"], training_spout, preprocessor.vocabulary, arrays,
        )
        seconds = time.perf_counter() - start
        after = {
            "tree": tree_structure_identity(model.forest), "response": array_identity(model.y),
            "attachment": {name: array_identity(arrays[name]) for name in sorted(arrays)},
        }
        if before != after:
            raise ValueError("v0.32 prediction changed frozen tree/response/selected-member source")
        legacy_sha = _legacy_check(candidate, slot, evaluation["ids"], legacy)
        output_arrays = {
            "ids": evaluation["ids"], "query_spout": exact_tokens(evaluation["spout"]),
            "median": median, "legacy_median": legacy, **_diagnostic_arrays(diagnostics),
        }
        receipt = {
            "slot": slot, "candidate": candidate,
            "candidate_id": "V32I_SAME_SPOUT_OOB_BLEND" if candidate == "A" else "V32T_SAME_SPOUT_OOB_TIME",
            "target": TARGETS[candidate], "rows": len(evaluation),
            "source_folder": str(folder), "source_bundle_sha256": sha(folder / "bundle.json"),
            "source_forest_sha256": sha(folder / "forest.joblib"), "source_prediction_sha256": sha(source_prediction),
            "source_tree_structure_sha256": before["tree"], "source_response": before["response"],
            "training_ids": array_identity(train["ids"]), "training_spout": array_identity(training_spout),
            "preprocessor_vocabulary": [str(value) for value in preprocessor.vocabulary],
            "attachment_path": str(attachment), "attachment_sha256": attachment_metadata["sha256"],
            "attachment_metadata_sha256": sha(Path(str(attachment) + ".json")),
            "attachment_arrays": before["attachment"], "legacy_v29_endpoint_sha256": legacy_sha,
            "legacy_equal_tree_switch_back_exact": True,
            "only_selected_member_subsetting_changed": True,
            "diagnostics": _summary(diagnostics), "changed_from_legacy_count": int(np.count_nonzero(median != legacy)),
            "support": [float(np.min(model.y)), float(np.max(model.y))],
            "nonfinite_or_negative": int(np.count_nonzero(~np.isfinite(median) | (median < 0))),
            "zero_fit": counter, "prediction_seconds": seconds,
            "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "training_transform_diagnostic": train_diagnostic, "evaluation_transform_diagnostic": evaluation_diagnostic,
            "train_input": train_info, "evaluation_input": evaluation_info, "exact_cold_checks": False,
        }
        _write_npz(output, output_arrays, receipt)
    return receipt


def cold(root, slot, candidate, output):
    _registered(root)
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("unregistered v0.32 slot/candidate")
    with V29_ADAPTER.zero_fit() as counter:
        restored = _restore(candidate, slot)
        folder, model, preprocessor, metadata, source_prediction, train, train_info, train_x, train_diagnostic, training_spout, attachment, arrays, attachment_metadata = restored
        evaluation, evaluation_info = V29_ADAPTER._features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = V29_ADAPTER._transform(preprocessor, evaluation, evaluation_info, False)
        median, legacy, diagnostics = _predict(model, evaluation_x, evaluation["spout"], training_spout, preprocessor.vocabulary, arrays)
        saved_path = root / "worker_predictions" / candidate / f"{slot}.npz"
        with np.load(saved_path, allow_pickle=False) as saved:
            expected = {name: saved[name] for name in saved.files}
        actual = {
            "ids": evaluation["ids"], "query_spout": exact_tokens(evaluation["spout"]),
            "median": median, "legacy_median": legacy, **_diagnostic_arrays(diagnostics),
        }
        if set(expected) != set(actual) or any(not _arrays_equal(expected[name], actual[name]) for name in actual):
            raise ValueError("cold v0.32 prediction arrays differ")
        indices = np.asarray([0, len(evaluation_x) // 2, len(evaluation_x) - 1])
        checks = {
            "reverse": (_predict(model, evaluation_x[::-1], evaluation["spout"][::-1], training_spout, preprocessor.vocabulary, arrays)[0], median[::-1]),
            "subset": (_predict(model, evaluation_x[indices], evaluation["spout"][indices], training_spout, preprocessor.vocabulary, arrays)[0], median[indices]),
            "single": (_predict(model, evaluation_x[indices[1]:indices[1] + 1], evaluation["spout"][indices[1]:indices[1] + 1], training_spout, preprocessor.vocabulary, arrays)[0], median[indices[1]:indices[1] + 1]),
        }
        for name, (current, expected_values) in checks.items():
            if not np.array_equal(current, expected_values):
                raise ValueError(f"cold v0.32 {name} invariance failed")
        pieces = [
            _predict(model, evaluation_x[index:index + 127], evaluation["spout"][index:index + 127], training_spout, preprocessor.vocabulary, arrays)[0]
            for index in range(0, len(evaluation_x), 127)
        ]
        if not np.array_equal(np.concatenate(pieces), median):
            raise ValueError("cold v0.32 chunk invariance failed")
        _legacy_check(candidate, slot, evaluation["ids"], legacy)
        receipt = {
            "slot": slot, "candidate": candidate, "rows": len(evaluation), "match": True,
            "reverse_chunk_subset_single_exact": True, "attachment_rederived_from_bootstrap_exact": True,
            "source_tree_structure_sha256": tree_structure_identity(model.forest),
            "attachment_sha256": attachment_metadata["sha256"], "legacy_switch_back_exact": True,
            "conditioned_selected_members_nonempty_subsets": True,
            "unknown_and_missing_query_bypass": True, "separate_oob_and_same_spout_fallback_diagnostics": True,
            "zero_fit": counter, "nonfinite_or_negative": int(np.count_nonzero(~np.isfinite(median) | (median < 0))),
        }
        _write_npz(output, {"ids": evaluation["ids"], "median": median, "legacy_median": legacy}, receipt)
    return receipt


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
