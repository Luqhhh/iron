"""Source-bound synthetic engineering experiment; no official data readers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import sklearn

from .joint_support_audit import local_output, sha, write_new
from .laplace_noise import BOUNDS, LEARNING_RATE, STEPS, LaplaceTreeRegressor

SPEC = "configs/laplace_noise_research/SPEC.json"
SOURCES = (SPEC, "src/bf_tap_r2/laplace_noise.py", "src/bf_tap_r2/laplace_noise_g0.py",
           "src/bf_tap_r2/joint_support_audit.py", "src/bf_tap_r2/data.py")


def inventory(root):
    return {p: sha(root/p) for p in SOURCES}


def validate_spec(spec):
    expected = {
        "identity": "XJN_LAPLACE_NOISE_RESEARCH_G0_V1", "stage": "synthetic_engineering_only",
        "recipes": ["FIXED", "ADAPTIVE"], "seed": 961201, "rows": 2754, "train_rows": 2204,
        "numeric_features": 21, "spout_values": [1, 2], "epochs": 500,
        "max_depth": 3, "min_samples_leaf": 20, "tree_seed": 42,
        "learning_rate": LEARNING_RATE, "log_scale_bounds": list(BOUNDS),
        "line_search_steps": list(STEPS),
        "gate": "each_arm_beats_training_median_query_mae_and_cold_predictions_match",
        "require_adaptive_beats_fixed": False, "maximum_runtime_seconds": None,
        "automatic_retry": False, "official_data_reads": 0, "quality_claim": False, "packages": 0,
    }
    if spec != expected:
        raise ValueError("Frozen G0 specification differs")


def synthetic_data():
    rng = np.random.default_rng(961201)
    x = rng.standard_normal((2754, 21))
    spout = rng.integers(1, 3, 2754)
    noise = rng.laplace(size=2754)
    y = 100.+6*x[:, 0]+4*np.sin(x[:, 1])+2*x[:, 2]*x[:, 3]+.4*np.exp(.5*x[:, 4])*noise
    features = np.column_stack((x, spout == 1, spout == 2))
    return features[:2204], y[:2204], features[2204:], y[2204:]


def verify(root, out):
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    result = json.loads((out/"result.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != inventory(root):
        raise ValueError("Source identity changed")
    for name, digest in result["output_sha256"].items():
        if sha(out/name) != digest:
            raise ValueError("Saved artifact identity differs")
    with np.load(out/"query.npz", allow_pickle=False) as data:
        query, actual = data["x"], data["y"]
    tx, ty, expected_x, expected_y = synthetic_data()
    if not np.array_equal(query, expected_x) or not np.array_equal(actual, expected_y):
        raise ValueError("Frozen synthetic query differs")
    checked = []
    for recipe in ("FIXED", "ADAPTIVE"):
        # Never load untrusted external pickles; source and model hashes were
        # checked above and these are our own local experiment artifacts.
        with (out/f"{recipe}.pkl").open("rb") as handle:
            model = pickle.load(handle)
        saved = np.load(out/f"{recipe}.npy", allow_pickle=False)
        pred = model.predict(query)
        chunk = np.concatenate([model.predict(query[i:i+37]) for i in range(0, len(query), 37)])
        reverse = model.predict(query[::-1])[::-1]
        scale = model.predict_scale(query)
        if (not np.array_equal(pred, saved) or not np.array_equal(pred, chunk)
                or not np.array_equal(pred, reverse) or not np.isfinite(scale).all()
                or not np.all(scale > 0) or not np.isfinite(pred).all() or not np.all(pred >= 0)):
            raise ValueError("Cold/order/chunk/finite/nonnegative engineering gate failed")
        expected_count = 500 if recipe == "FIXED" else 1000
        history = np.array([model.initial_nll_, *[entry["train_nll"] for entry in model.history_]])
        trace = json.loads((out/f"{recipe}-trace.json").read_text(encoding="utf-8"))
        if (model.metadata() != result["arms"][recipe]["metadata"]
                or trace != {"initial_nll": model.initial_nll_, "history": model.history_}
                or model.fitted_tree_count_ != expected_count or model.selected_epoch_ != 500
                or not np.isfinite(history).all() or not np.all(np.diff(history) <= 0)
                or not np.array_equal(model.x_mean_, tx.mean(axis=0))
                or not np.array_equal(model.x_scale_, np.where(tx.std(axis=0) > 0, tx.std(axis=0), 1.))
                or model.y_median_ != float(np.median(ty)) or model.y_scale_ != float(ty.std())):
            raise ValueError("Fit preprocessing, history or ledger differs")
        mae = float(np.abs(pred-actual).mean())
        if mae != result["arms"][recipe]["query_mae"] or not mae < result["training_median_query_mae"]:
            raise ValueError("Engineering learnability gate failed")
        checked.append(recipe)
    return {"status": "passed_synthetic_engineering_cold_readback", "arms": checked,
            "model_fits": 0, "official_data_reads": 0, "quality_claim": False,
            "manifest_sha256": sha(out/"manifest.json"), "result_sha256": sha(out/"result.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    out = local_output(root, args.output)
    if args.action == "verify":
        receipt = verify(root, out)
        write_new(out/"cold-readback.json", receipt)
        print(json.dumps(receipt))
        return
    spec = json.loads((root/SPEC).read_text(encoding="utf-8"))
    validate_spec(spec)
    frozen = inventory(root)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"manifest.json", {"identity": spec["identity"], "source_sha256": frozen,
              "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
              "official_data_reads": 0})
    try:
        x, y, qx, qy = synthetic_data()
        with (out/"query.npz").open("xb") as handle:
            np.savez(handle, x=qx, y=qy)
        constant_mae = float(np.abs(qy-np.median(y)).mean())
        arms = {}
        for recipe in spec["recipes"]:
            model = LaplaceTreeRegressor(recipe).fit(x, y, epochs=spec["epochs"])
            prediction = model.predict(qx)
            mae = float(np.abs(prediction-qy).mean())
            if not mae < constant_mae:
                raise ValueError(f"{recipe} failed the frozen learnability gate")
            with (out/f"{recipe}.pkl").open("xb") as handle:
                pickle.dump(model, handle, protocol=5)
            with (out/f"{recipe}.npy").open("xb") as handle:
                np.save(handle, prediction, allow_pickle=False)
            write_new(out/f"{recipe}-trace.json", {"initial_nll": model.initial_nll_, "history": model.history_})
            arms[recipe] = {"metadata": model.metadata(), "query_mae": mae}
        if inventory(root) != frozen:
            raise ValueError("Source changed during experiment")
        names = ["query.npz", *[f"{r}{s}" for r in spec["recipes"] for s in (".pkl", ".npy", "-trace.json")]]
        result = {"status": "complete_synthetic_not_official_quality", "arms": arms,
                  "training_median_query_mae": constant_mae, "model_fits": 2, "tree_fits": 1500,
                  "official_data_reads": 0, "packages": 0, "quality_claim": False,
                  "output_sha256": {name: sha(out/name) for name in names}}
        write_new(out/"result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "output_sha256"}))
    except Exception as exc:
        write_new(out/"FAILED.json", {"error_type": type(exc).__name__, "error": str(exc), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
