"""v0.24 adapter around the frozen v0.15 QRF numeric core and lock."""
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
FROZEN = HERE.parent / "qrf_v015"
sys.path.insert(0, str(FROZEN))

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.ensemble import RandomForestRegressor
from preprocessing import Preprocessor
from qrf_model import PARAMETERS, PROTOCOL, QRF


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__, threadpoolctl=threadpoolctl.__version__)


def source_identity():
    files = {"adapter": HERE / "worker.py", "qrf_model": FROZEN / "qrf_model.py",
             "preprocessing": FROZEN / "preprocessing.py", "pyproject": FROZEN / "pyproject.toml",
             "lock": FROZEN / "uv.lock"}
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


@contextmanager
def zero_fit():
    counter = {"forest_fit_attempts": 0, "preprocessor_fit_attempts": 0}
    def reject_forest(*args, **kwargs):
        counter["forest_fit_attempts"] += 1; raise ValueError("cold inference forbids forest fit")
    def reject_pre(*args, **kwargs):
        counter["preprocessor_fit_attempts"] += 1; raise ValueError("cold inference forbids preprocessor fit")
    with patch.object(QRF, "fit", reject_forest), patch.object(RandomForestRegressor, "fit", reject_forest), patch.object(Preprocessor, "fit", reject_pre):
        yield counter


def fit(root: Path, slot: int):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["worker_environment"] != environment() or manifest["worker_sources"] != source_identity():
        raise ValueError("registered worker environment/source differs")
    if manifest["qrf_parameters"] != PARAMETERS or slot not in range(6, 13):
        raise ValueError("unregistered QRF fit slot")
    attempted = len(list((root / "models/B").glob("*/forest_intent.json")))
    if attempted >= 7:
        raise ValueError("seven-fit QRF budget exhausted")
    folder = root / "models/B" / str(slot); folder.mkdir(parents=True, exist_ok=False)
    arrays, info = payload(root / "features" / str(slot) / "train.npz")
    expected = {"ids", "numeric", "spout", "reference_ns", "available_ns", "y", "training_months"}
    if set(arrays) != expected or (arrays["reference_ns"] >= info["cutoff_ns"]).any() or (arrays["available_ns"] > info["cutoff_ns"]).any():
        raise ValueError("QRF training boundary/payload differs")
    common = dict(input=info, parameters=PARAMETERS, protocol=PROTOCOL,
                  worker_sources=source_identity(), environment=environment(), manifest_sha256=sha(root / "manifest.json"))
    write(folder / "preprocessor_intent.json", common)
    pre = Preprocessor().fit(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    write(folder / "preprocessor.json", pre.metadata())
    x, diagnostic = pre.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
    write(folder / "forest_intent.json", common)
    start = time.perf_counter(); model = QRF().fit(x, arrays["y"], info["ids"]); model.training_months = arrays["training_months"]
    joblib.dump(model, folder / "forest.joblib", compress=3)
    write(folder / "bundle.json", dict(**common, forest_sha256=sha(folder / "forest.joblib"),
          preprocessor_sha256=sha(folder / "preprocessor.json"), support=[float(model.y.min()), float(model.y.max())],
          training_diagnostic=diagnostic, fit_seconds=time.perf_counter() - start,
          model_bytes=(folder / "forest.joblib").stat().st_size,
          peak_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def restore(folder: Path, trusted_sha: str):
    if sha(folder / "bundle.json") != trusted_sha:
        raise ValueError("untrusted QRF bundle")
    md = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if md["environment"] != environment() or md["worker_sources"] != source_identity() or md["parameters"] != PARAMETERS or md["protocol"] != PROTOCOL:
        raise ValueError("QRF bundle source/protocol differs")
    if sha(folder / "forest.joblib") != md["forest_sha256"] or sha(folder / "preprocessor.json") != md["preprocessor_sha256"]:
        raise ValueError("QRF serialized component identity differs")
    model = joblib.load(folder / "forest.joblib")
    pre = Preprocessor.restore(json.loads((folder / "preprocessor.json").read_text(encoding="utf-8")))
    if model.ids != pre.training_ids or len(model.forest.estimators_) != 256:
        raise ValueError("QRF restored training/tree identity differs")
    return model, pre, md


def predict(root: Path, slot: int, output: Path, cold: bool):
    completion = root / ("final_models_complete.json" if slot == 12 else "development_models_complete.json")
    completed = json.loads(completion.read_text(encoding="utf-8"))
    with zero_fit() as counter:
        model, pre, md = restore(root / "models/B" / str(slot), completed[str(slot)])
        arrays, info = payload(root / "features" / str(slot) / "evaluation.npz")
        if set(arrays) != {"ids", "numeric", "spout", "reference_ns"} or set(info["ids"]) & set(model.ids) or (arrays["reference_ns"] < info["cutoff_ns"]).any():
            raise ValueError("QRF evaluation payload/boundary differs")
        x, diagnostic = pre.transform(arrays["numeric"], arrays["spout"], info["ids"], info["numeric_columns"])
        median, mean, neighbors = model.predict(x, model.training_months)
        if cold:
            with np.load(root / "worker_predictions" / f"{slot}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["median"], median) or not np.array_equal(saved["mean"], mean):
                    raise ValueError("cold QRF output differs")
            for indices in (np.arange(len(x))[::-1], np.array([0, len(x)//2, len(x)-1]), np.array([len(x)//2])):
                q, m, _ = model.predict(x[indices])
                if not np.array_equal(q, median[indices]) or not np.array_equal(m, mean[indices]):
                    raise ValueError("QRF subset/order invariance failed")
            chunks = [model.predict(x[i:i+127])[:2] for i in range(0, len(x), 127)]
            if not np.array_equal(np.concatenate([x[0] for x in chunks]), median) or not np.array_equal(np.concatenate([x[1] for x in chunks]), mean):
                raise ValueError("QRF chunk invariance failed")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists(): raise ValueError("never overwrite QRF prediction")
            np.savez(output, ids=arrays["ids"], median=median, mean=mean)
        write(Path(str(output) + ".json"), dict(slot=slot, rows=len(x), input=info, support=md["support"],
              neighbors=neighbors, preprocessor_diagnostic=diagnostic, zero_fit=counter, exact_cold_checks=cold))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("environment", "identity", "fit", "predict", "cold"))
    parser.add_argument("--root", type=Path); parser.add_argument("--slot", type=int); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "environment": print(json.dumps(environment(), sort_keys=True))
    elif args.command == "identity": print(json.dumps(source_identity(), sort_keys=True))
    elif args.command == "fit": fit(args.root.resolve(), args.slot)
    else: predict(args.root.resolve(), args.slot, args.output.resolve(), args.command == "cold")


if __name__ == "__main__": main()
