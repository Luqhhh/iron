"""Locked v0.29 adapter deriving OOB leaf responses from frozen forests."""
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
sys.path[:0] = [str(HERE), str(V27_WORKER), str(V26_WORKER), str(FROZEN)]

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl

from oob_response import PROTOCOL, array_identity, derive_attachment, load_npz, predict, tree_structure_identity, validate_attachment


V26 = REPOSITORY / "local/runs/optimization-v0.26-qrf-partition-tests-r1"
V27 = REPOSITORY / "local/runs/optimization-v0.27-qrf-iron-and-leaf-recency-r1"
TARGETS = {
    "A": {"target": "tap_iron", "unit": "tonne", "source": "V27I_ABS_QRF_DIRECT_IRON"},
    "B": {"target": "tap_time_len", "unit": "minutes", "source": "V26A_QRF_ABSOLUTE_SPLIT_TIME"},
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V27_ADAPTER = _load_module("qrf_v027_adapter_for_v029", V27_WORKER / "worker.py")


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
        "oob_response": HERE / "oob_response.py",
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


def _registered(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered v0.29 worker environment/source differs")
    if manifest["oob_protocol"] != PROTOCOL or manifest["candidate_targets"] != TARGETS:
        raise ValueError("registered v0.29 OOB protocol/targets differ")
    return manifest


def _completion(root, slot, candidate):
    name = "final_models_complete.json" if slot == 12 else "development_models_complete.json"
    return json.loads((root / name).read_text(encoding="utf-8"))[candidate][str(slot)]


def _restore_source(slot, candidate):
    if candidate == "A":
        folder = V27 / "models" / "A" / str(slot)
        model, preprocessor, metadata = V27_ADAPTER.restore_a(folder, _completion(V27, slot, "A"))
        source_prediction = V27 / "worker_predictions" / "A" / f"{slot}.npz"
    elif candidate == "B":
        model, preprocessor, metadata = V27_ADAPTER.restore_parent(slot)
        folder = V26 / "models" / "A" / str(slot)
        source_prediction = V26 / "worker_predictions" / "A" / f"{slot}.npz"
    else:
        raise ValueError("registered v0.29 candidate A or B required")
    if len(model.forest.estimators_) != 256 or len(model.leaves) != 256:
        raise ValueError("registered source must retain exactly 256 frozen trees")
    return folder, model, preprocessor, metadata, source_prediction


def _features(slot, kind):
    return V27_ADAPTER.payload(V27 / "features" / str(slot) / f"{kind}.npz")


def _transform(preprocessor, arrays, info, training):
    if training and preprocessor.training_ids != arrays["ids"].tolist():
        raise ValueError("frozen preprocessor/model training IDs differ")
    transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    if transformed.dtype != np.float32 or not np.isfinite(transformed).all():
        raise ValueError("frozen transform is not finite float32")
    return transformed, diagnostic


def _save_attachment(path, arrays, certificate):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or Path(str(path) + ".json").exists():
        raise ValueError("never overwrite v0.29 OOB attachment")
    np.savez(path, **arrays)
    metadata = {**certificate, "sha256": sha(path), "fit": False}
    write(Path(str(path) + ".json"), metadata)
    return metadata


def _load_attachment(path, model, transformed):
    metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if sha(path) != metadata["sha256"] or metadata["protocol"] != PROTOCOL:
        raise ValueError("persisted OOB attachment identity/protocol differs")
    arrays = load_npz(path)
    validate_attachment(model, transformed, arrays, rederive=True)
    for name, identity in metadata["arrays"].items():
        if array_identity(arrays[name]) != identity:
            raise ValueError("persisted OOB attachment array digest differs")
    return arrays, metadata


def _check_invariance(model, transformed, arrays, median, mean):
    sets = (
        np.arange(len(transformed))[::-1],
        np.asarray([0, len(transformed) // 2, len(transformed) - 1]),
        np.asarray([len(transformed) // 2]),
    )
    for indices in sets:
        current = predict(model, transformed[indices], arrays, model.training_months)
        if not np.array_equal(current[0], median[indices]) or not np.array_equal(current[1], mean[indices]):
            raise ValueError("OOB response reverse/subset/single invariance failed")
    chunks = [predict(model, transformed[index:index + 127], arrays, model.training_months)[:2]
              for index in range(0, len(transformed), 127)]
    if not np.array_equal(np.concatenate([part[0] for part in chunks]), median):
        raise ValueError("OOB response chunk median invariance failed")
    if not np.array_equal(np.concatenate([part[1] for part in chunks]), mean):
        raise ValueError("OOB response chunk mean invariance failed")


@contextmanager
def zero_fit():
    with V27_ADAPTER.zero_fit() as counter:
        yield counter


def audit(root, slot, candidate):
    _registered(root)
    with zero_fit() as counter:
        folder, model, preprocessor, metadata, source_prediction = _restore_source(slot, candidate)
        train, train_info = _features(slot, "train")
        transformed, diagnostic = _transform(preprocessor, train, train_info, True)
        before = tree_structure_identity(model.forest)
        arrays, certificate = derive_attachment(model, transformed)
        full_replay = model.predict(transformed[: min(17, len(transformed))], model.training_months)
        if before != tree_structure_identity(model.forest):
            raise ValueError("P0 derivation changed frozen forest structure")
    return {
        "slot": slot,
        "candidate": candidate,
        "target": TARGETS[candidate],
        "source_folder": str(folder),
        "source_bundle_sha256": sha(folder / "bundle.json"),
        "source_forest_sha256": sha(folder / "forest.joblib"),
        "source_prediction_sha256": sha(source_prediction),
        "training_ids": array_identity(train["ids"]),
        "training_response": array_identity(model.y),
        "training_transform": array_identity(transformed),
        "transform_diagnostic": diagnostic,
        "certificate_preview": certificate,
        "full_member_smoke_rows": len(full_replay[0]),
        "zero_fit": counter,
    }


def execute(root, slot, candidate, output, cold):
    _registered(root)
    if slot not in range(6, 13) or candidate not in TARGETS:
        raise ValueError("unregistered v0.29 slot/candidate")
    with zero_fit() as counter:
        folder, model, preprocessor, source_metadata, source_prediction = _restore_source(slot, candidate)
        train, train_info = _features(slot, "train")
        train_x, train_diagnostic = _transform(preprocessor, train, train_info, True)
        if (train["reference_ns"] >= train_info["cutoff_ns"]).any() or (train["available_ns"] > train_info["cutoff_ns"]).any():
            raise ValueError("OOB response donor violates training cutoff/availability")
        if model.ids != train["ids"].tolist():
            raise ValueError("OOB response donor IDs differ from frozen model order")
        attachment = root / "oob_attachments" / candidate / f"{slot}.npz"
        if cold:
            arrays, attachment_metadata = _load_attachment(attachment, model, train_x)
        else:
            arrays, certificate = derive_attachment(model, train_x)
            certificate.update({
                "slot": slot,
                "candidate": candidate,
                "target": TARGETS[candidate]["target"],
                "unit": TARGETS[candidate]["unit"],
                "cutoff": train_info["cutoff"],
                "cutoff_ns": train_info["cutoff_ns"],
                "source_bundle_sha256": sha(folder / "bundle.json"),
                "source_forest_sha256": sha(folder / "forest.joblib"),
                "source_training_ids": array_identity(train["ids"]),
                "source_response": array_identity(model.y),
                "source_transformed_float32": array_identity(train_x),
            })
            attachment_metadata = _save_attachment(attachment, arrays, certificate)

        evaluation, evaluation_info = _features(slot, "evaluation")
        evaluation_x, evaluation_diagnostic = _transform(preprocessor, evaluation, evaluation_info, False)
        start = time.perf_counter()
        median, mean, diagnostics = predict(model, evaluation_x, arrays, model.training_months)
        seconds = time.perf_counter() - start

        # Replacing selected members by the model's original full members must
        # exactly recover the certified source endpoint.
        full_median, full_mean, _ = model.predict(evaluation_x, model.training_months)
        with np.load(source_prediction, allow_pickle=False) as saved_source:
            if saved_source["ids"].tolist() != evaluation["ids"].tolist():
                raise ValueError("source endpoint IDs differ")
            if not np.array_equal(full_median, saved_source["median"]) or not np.array_equal(full_mean, saved_source["mean"]):
                raise ValueError("full-member replay differs from certified source QRF")

        saved_output = root / "worker_predictions" / candidate / f"{slot}.npz"
        if cold:
            with np.load(saved_output, allow_pickle=False) as saved:
                if saved["ids"].tolist() != evaluation["ids"].tolist() or not np.array_equal(saved["median"], median) or not np.array_equal(saved["mean"], mean):
                    raise ValueError("cold OOB worker prediction differs")
                if not np.array_equal(saved["fallback_tree_counts"], np.asarray([item["fallback_tree_count"] for item in diagnostics], dtype=np.int16)):
                    raise ValueError("cold OOB fallback diagnostics differ")
            _check_invariance(model, evaluation_x, arrays, median, mean)
        else:
            if saved_output.exists() or output != saved_output:
                raise ValueError("registered OOB output path required and never overwritten")
            saved_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(saved_output, ids=evaluation["ids"], median=median, mean=mean,
                     fallback_tree_counts=np.asarray([item["fallback_tree_count"] for item in diagnostics], dtype=np.int16))

        receipt = {
            "slot": slot,
            "candidate": candidate,
            "candidate_id": "V29I_OOB_LEAF_QRF_BLEND" if candidate == "A" else "V29T_OOB_LEAF_QRF_TIME",
            "target": TARGETS[candidate]["target"],
            "unit": TARGETS[candidate]["unit"],
            "rows": len(evaluation),
            "source_bundle_sha256": sha(folder / "bundle.json"),
            "source_forest_sha256": sha(folder / "forest.joblib"),
            "source_prediction_sha256": sha(source_prediction),
            "source_tree_structure_sha256": tree_structure_identity(model.forest),
            "source_response": array_identity(model.y),
            "attachment_sha256": attachment_metadata["sha256"],
            "attachment_metadata_sha256": sha(Path(str(attachment) + ".json")),
            "full_member_replay_exact": True,
            "bootstrap_recovered_from_estimators_samples": True,
            "prediction_seconds": seconds,
            "lower_boundary": int(np.count_nonzero(median == np.min(model.y))),
            "upper_boundary": int(np.count_nonzero(median == np.max(model.y))),
            "fallback_query_tree_count_quantiles": {
                str(q): float(np.quantile([item["fallback_tree_count"] for item in diagnostics], q))
                for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
            "neighbors": diagnostics,
            "training_input": train_info,
            "evaluation_input": evaluation_info,
            "training_transform_diagnostic": train_diagnostic,
            "evaluation_transform_diagnostic": evaluation_diagnostic,
            "zero_fit": counter,
            "exact_cold_checks": cold,
            "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        write(Path(str(output) + ".json"), receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("environment", "identity", "audit", "derive-predict", "cold"))
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
        print(json.dumps(audit(arguments.root.resolve(), arguments.slot, arguments.candidate), sort_keys=True))
    else:
        execute(arguments.root.resolve(), arguments.slot, arguments.candidate, arguments.output.resolve(), arguments.command == "cold")


if __name__ == "__main__":
    main()
