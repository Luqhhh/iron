"""v0.25 signed-response QRF adapter using the frozen v0.15 environment and preprocessors."""
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
from sklearn.ensemble import RandomForestRegressor
from preprocessing import Preprocessor
from qrf_model import PARAMETERS
from signed_qrf import PARENT_PROTOCOL, PROTOCOL, SignedQRF


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
        "signed_qrf": HERE / "signed_qrf.py",
        "parent_qrf_model": FROZEN / "qrf_model.py",
        "preprocessing": FROZEN / "preprocessing.py",
        "pyproject": FROZEN / "pyproject.toml",
        "lock": FROZEN / "uv.lock",
    }
    return {key: {"path": str(path), "sha256": sha(path)} for key, path in files.items()}


def payload(path):
    info = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if sha(path) != info["sha256"]:
        raise ValueError("worker input identity changed")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if arrays["ids"].tolist() != info["ids"]:
        raise ValueError("worker input IDs differ")
    return arrays, info


def original_preprocessor(info, arrays):
    path = (REPOSITORY / info["original_preprocessor_path"]).resolve()
    if not path.is_relative_to(REPOSITORY.resolve()) or sha(path) != info["original_preprocessor_sha256"]:
        raise ValueError("original preprocessor identity differs")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    preprocessor = Preprocessor.restore(metadata)
    if preprocessor.training_ids != arrays["ids"].tolist() or preprocessor.columns != info["numeric_columns"]:
        raise ValueError("original preprocessor training/schema identity differs")
    return preprocessor, path


@contextmanager
def zero_fit():
    counter = {"forest_fit_attempts": 0, "preprocessor_fit_attempts": 0}

    def reject_forest(*args, **kwargs):
        counter["forest_fit_attempts"] += 1
        raise ValueError("cold inference forbids forest fit")

    def reject_preprocessor(*args, **kwargs):
        counter["preprocessor_fit_attempts"] += 1
        raise ValueError("cold inference forbids preprocessor fit")

    with patch.object(SignedQRF, "fit", reject_forest), patch.object(RandomForestRegressor, "fit", reject_forest), patch.object(Preprocessor, "fit", reject_preprocessor):
        yield counter


def fit(root: Path, slot: int):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered worker environment/source differs")
    if manifest["qrf_parameters"] != PARAMETERS or slot not in range(6, 13):
        raise ValueError("unregistered signed QRF fit slot")
    attempted = len(list((root / "models/B").glob("*/forest_intent.json")))
    if attempted >= 7:
        raise ValueError("seven-fit signed QRF budget exhausted")
    folder = root / "models/B" / str(slot)
    folder.mkdir(parents=True, exist_ok=False)
    arrays, info = payload(root / "features" / str(slot) / "train.npz")
    expected = {"ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months",
                "baseline_iron", "baseline_time", "baseline_iron_source", "baseline_time_source"}
    if set(arrays) != expected or (arrays["reference_ns"] >= info["cutoff_ns"]).any() or (arrays["available_ns"] > info["cutoff_ns"]).any():
        raise ValueError("signed QRF training boundary/payload differs")
    preprocessor, preprocessor_path = original_preprocessor(info, arrays)
    transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    signed = np.asarray(arrays["y"], dtype=float) - np.asarray(arrays["baseline_time"], dtype=float)
    common = {
        "input": info,
        "parameters": PARAMETERS,
        "protocol": PROTOCOL,
        "parent_protocol": PARENT_PROTOCOL,
        "response_transform": "tap_time_len_minus_legal_last100_baseline",
        "worker_sources": source_identity(),
        "environment": environment(),
        "manifest_sha256": sha(root / "manifest.json"),
        "original_preprocessor_path": str(preprocessor_path.relative_to(REPOSITORY)),
        "original_preprocessor_sha256": sha(preprocessor_path),
        "preprocessor_fit": False,
    }
    write(folder / "forest_intent.json", common)
    start = time.perf_counter()
    model = SignedQRF().fit(transformed, signed, info["ids"])
    model.training_months = arrays["training_months"]
    joblib.dump(model, folder / "forest.joblib", compress=3)
    write(folder / "bundle.json", {
        **common,
        "forest_sha256": sha(folder / "forest.joblib"),
        "signed_support": [float(model.y.min()), float(model.y.max())],
        "training_diagnostic": diagnostic,
        "fit_seconds": time.perf_counter() - start,
        "model_bytes": (folder / "forest.joblib").stat().st_size,
        "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })


def restore(folder: Path, trusted_sha: str):
    if sha(folder / "bundle.json") != trusted_sha:
        raise ValueError("untrusted signed QRF bundle")
    metadata = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if (metadata["environment"] != environment() or metadata["worker_sources"] != source_identity()
            or metadata["parameters"] != PARAMETERS or metadata["protocol"] != PROTOCOL
            or metadata["parent_protocol"] != PARENT_PROTOCOL or metadata["preprocessor_fit"] is not False):
        raise ValueError("signed QRF bundle source/protocol differs")
    if sha(folder / "forest.joblib") != metadata["forest_sha256"]:
        raise ValueError("signed QRF forest identity differs")
    preprocessor_path = REPOSITORY / metadata["original_preprocessor_path"]
    if sha(preprocessor_path) != metadata["original_preprocessor_sha256"]:
        raise ValueError("restored original preprocessor identity differs")
    model = joblib.load(folder / "forest.joblib")
    preprocessor = Preprocessor.restore(json.loads(preprocessor_path.read_text(encoding="utf-8")))
    if model.ids != preprocessor.training_ids or len(model.forest.estimators_) != 256:
        raise ValueError("signed QRF restored training/tree identity differs")
    return model, preprocessor, metadata


def predict(root: Path, slot: int, output: Path, cold: bool):
    completion = root / ("final_models_complete.json" if slot == 12 else "development_models_complete.json")
    completed = json.loads(completion.read_text(encoding="utf-8"))
    with zero_fit() as counter:
        model, preprocessor, metadata = restore(root / "models/B" / str(slot), completed[str(slot)])
        arrays, info = payload(root / "features" / str(slot) / "evaluation.npz")
        expected = {"ids", "numeric", "spout", "reference_ns", "baseline_iron", "baseline_time",
                    "baseline_iron_source", "baseline_time_source"}
        if set(arrays) != expected or set(info["ids"]) & set(model.ids) or (arrays["reference_ns"] < info["cutoff_ns"]).any():
            raise ValueError("signed QRF evaluation payload/boundary differs")
        transformed, diagnostic = preprocessor.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
        signed_median, signed_mean, neighbors = model.predict(transformed, model.training_months)
        absolute = np.maximum(0.0, np.asarray(arrays["baseline_time"], dtype=float) + signed_median)
        absolute_mean = np.maximum(0.0, np.asarray(arrays["baseline_time"], dtype=float) + signed_mean)
        if cold:
            with np.load(root / "worker_predictions" / f"{slot}.npz", allow_pickle=False) as saved:
                for key, value in (("absolute", absolute), ("absolute_mean", absolute_mean),
                                   ("signed_median", signed_median), ("signed_mean", signed_mean)):
                    if not np.array_equal(saved[key], value):
                        raise ValueError("cold signed QRF output differs")
            index_sets = (np.arange(len(transformed))[::-1], np.asarray([0, len(transformed)//2, len(transformed)-1]), np.asarray([len(transformed)//2]))
            for indices in index_sets:
                qrf, mean, _ = model.predict(transformed[indices])
                if not np.array_equal(qrf, signed_median[indices]) or not np.array_equal(mean, signed_mean[indices]):
                    raise ValueError("signed QRF subset/order invariance failed")
                if not np.array_equal(np.maximum(0.0, arrays["baseline_time"][indices] + qrf), absolute[indices]):
                    raise ValueError("signed QRF reconstruction subset invariance failed")
            chunks = [model.predict(transformed[index:index+127])[:2] for index in range(0, len(transformed), 127)]
            if not np.array_equal(np.concatenate([chunk[0] for chunk in chunks]), signed_median):
                raise ValueError("signed QRF chunk median invariance failed")
            if not np.array_equal(np.concatenate([chunk[1] for chunk in chunks]), signed_mean):
                raise ValueError("signed QRF chunk mean invariance failed")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise ValueError("never overwrite signed QRF prediction")
            np.savez(output, ids=arrays["ids"], absolute=absolute, absolute_mean=absolute_mean,
                     signed_median=signed_median, signed_mean=signed_mean)
        write(Path(str(output) + ".json"), {
            "slot": slot,
            "rows": len(transformed),
            "input": info,
            "signed_support": metadata["signed_support"],
            "neighbors": neighbors,
            "preprocessor_diagnostic": diagnostic,
            "preprocessor_fit": False,
            "zero_fit": counter,
            "exact_cold_checks": cold,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("environment", "identity", "fit", "predict", "cold"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "environment":
        print(json.dumps(environment(), sort_keys=True))
    elif arguments.command == "identity":
        print(json.dumps(source_identity(), sort_keys=True))
    elif arguments.command == "fit":
        fit(arguments.root.resolve(), arguments.slot)
    else:
        predict(arguments.root.resolve(), arguments.slot, arguments.output.resolve(), arguments.command == "cold")


if __name__ == "__main__":
    main()

