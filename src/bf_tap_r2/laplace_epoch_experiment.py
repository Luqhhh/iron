"""Frozen epoch experiment: engineering, public-V2 candidate OOF, cold audit.

No submission release or automatic Q75 promotion is implemented here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import psutil
import sklearn

from .data import FEATURES
from .joint_support_audit import local_output, sha, write_new
from .laplace_epoch import LaplaceEpochRegressor, selected_epoch
from .laplace_noise import LaplaceTreeRegressor
from .metrics import wmape
from .splits import make_folds
from .submission import deny_training_reads
from .v2_release import load_v2

SPEC = "configs/laplace_epoch_research/SPEC.json"
SOURCES = (SPEC, "docs/laplace_epoch_research/PREREGISTRATION.md", "uv.lock", "pyproject.toml",
           "scripts/run_laplace_epoch_sequence.py",
           "src/bf_tap_r2/laplace_epoch.py", "src/bf_tap_r2/laplace_epoch_experiment.py",
           "src/bf_tap_r2/laplace_noise.py", "src/bf_tap_r2/joint_support_audit.py",
           "src/bf_tap_r2/data.py", "src/bf_tap_r2/splits.py", "src/bf_tap_r2/metrics.py",
           "src/bf_tap_r2/v2_release.py", "src/bf_tap_r2/submission.py", "src/bf_tap_r2/audit.py")
INPUTS = ("复赛_train/train_samples.csv", "复赛_train/train_features.csv")
EXPECTED = {
    "identity": "XJN_LAPLACE_EPOCH_V2_V1", "stage": "public_v2_candidate_only_not_Q75_promotion",
    "target": "tap_iron", "recipes": ["FIXED", "ADAPTIVE"], "horizons": [500, 3000],
    "max_epochs": 3000, "early_stopping": False, "split_seeds": [42, 3407], "folds": 5,
    "calibration_seed": 27001, "calibration_fold": 0, "tree_seed": 42, "max_depth": 3,
    "min_samples_leaf": 20, "learning_rate": .05, "log_scale_bounds": [-6., 3.],
    "line_search_steps": [1., .5, .25, .125, .0625, .03125, 0.],
    "selection": "calibration_raw_MAE_smallest_epoch_on_exact_ties",
    "fresh_refit": "all_outer_training_at_long_selected_epoch_then_read_short_prefix",
    "development_outer_units": 20, "development_model_fits": 40, "development_max_tree_fits": 180000,
    "workers": 1, "numerical_threads": 1, "max_worker_rss_mib": 1536, "monitor_seconds": 600,
    "maximum_runtime_seconds": None, "automatic_retry": False,
    "engineering_seed": 611002, "engineering_rows": 180, "engineering_fit_rows": 108,
    "engineering_calibration_rows": 36, "engineering_features": 4, "engineering_models": 4,
    "engineering_tree_fits": 10500, "confirmation_seeds_consumed": 0, "full_data_fits": 0,
    "packages": 0, "agent_uploads": 0,
}


def validate_spec(spec):
    if spec != EXPECTED:
        raise ValueError("Frozen epoch specification differs")


def inventory(root):
    return {name: sha(root/name) for name in SOURCES}


def synthetic_arrays():
    rng = np.random.default_rng(611002)
    x = rng.standard_normal((180, 4))
    noise = rng.laplace(size=len(x))
    y = 100.+6*x[:, 0]+4*np.sin(x[:, 1])+2*x[:, 2]*x[:, 3]+.4*np.exp(.5*x[:, 3])*noise
    return x, y


def save_arrays(path, **arrays):
    with path.open("xb") as handle:
        np.savez(handle, **arrays)


def save_model(out, key, model):
    with (out/f"{key}.pkl").open("xb") as handle:
        pickle.dump(model, handle, protocol=5)
    write_new(out/f"{key}-trace.json", {"metadata": model.metadata(), "history": model.history_})
    return {"key": key, "metadata": model.metadata()}


def record(out, event):
    with (out/"ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc_ns": time.time_ns(), **event}, allow_nan=False)+"\n")


def resource_check():
    rss = psutil.Process().memory_info().rss/1024**2
    if rss > EXPECTED["max_worker_rss_mib"]:
        raise MemoryError("Frozen worker RSS gate exceeded")
    return rss


def folds_for(frame, seed):
    return make_folds(frame, seed).set_index("sample_id").loc[frame.sample_id, "fold"].to_numpy(dtype=int)


def common_result(stage, models, peak_rss):
    return {"stage": stage, "model_fits": len(models),
            "tree_fits": sum(m["metadata"]["fitted_tree_count"] for m in models),
            "peak_observed_rss_mib": peak_rss, "models": models,
            "Q75_gain": None, "formal_promoted": False, "confirmation_seeds_consumed": 0,
            "full_data_fits": 0, "packages": 0, "platform_score": None, "agent_uploads": 0}


def engineering(out):
    x, y = synthetic_arrays()
    save_arrays(out/"data.npz", x=x, y=y)
    models, arms, peak = [], {}, resource_check()
    for recipe in EXPECTED["recipes"]:
        key = f"{recipe}-complete"
        record(out, {"event": "model_started", "key": key})
        model = LaplaceEpochRegressor(recipe).fit(x[:108], y[:108], epochs=3000,
                                                validation=(x[108:144], y[108:144]))
        models.append(save_model(out, key, model))
        record(out, {"event": "model_completed", **models[-1]})
        record(out, {"event": "model_started", "key": f"{recipe}-legacy500"})
        old = LaplaceTreeRegressor(recipe).fit(x[:108], y[:108], epochs=500)
        for k in range(500):
            if any(model.history_[k][name] != old.history_[k][name] for name in ("epoch", "train_nll", "step")):
                raise ValueError("Legacy trace prefix differs")
        if not np.array_equal(model.predict(x[144:], epoch=500), old.predict(x[144:])):
            raise ValueError("Legacy prediction prefix differs")
        models.append(save_model(out, f"{recipe}-legacy500", old))
        record(out, {"event": "model_completed", **models[-1]})
        short, long = model.best_epoch(500), model.best_epoch(3000)
        save_arrays(out/f"{recipe}-query.npz", prefix500=model.predict(x[144:], epoch=500),
                    short=model.predict(x[144:], epoch=short), long=model.predict(x[144:], epoch=long))
        arms[recipe] = {"short_selected_epoch": short, "long_selected_epoch": long,
                        "short_query_mae": float(np.abs(y[144:]-model.predict(x[144:], epoch=short)).mean()),
                        "long_query_mae": float(np.abs(y[144:]-model.predict(x[144:], epoch=long)).mean())}
        peak = max(peak, resource_check())
    result = common_result("synthetic_complete_epoch_engineering", models, peak)
    result.update(arms=arms, official_sample_reads=0)
    if result["model_fits"] != 4 or result["tree_fits"] != 10500:
        raise ValueError("Engineering ledger count differs")
    return result


def development(root, out):
    frame = load_v2(root/"复赛_train", "train", 2754)
    if not frame.spout_no.isin([1, 2]).all():
        raise ValueError("Frozen spout1/2 encoding does not match this snapshot")
    x = np.column_stack((frame[list(FEATURES)].to_numpy(dtype=float), frame.spout_no == 1, frame.spout_no == 2))
    y, ids = frame.tap_iron.to_numpy(dtype=float), frame.sample_id.to_numpy(dtype=str)
    save_arrays(out/"data.npz", x=x, y=y, ids=ids)
    models, units, metrics, peak = [], [], {}, resource_check()
    for seed in EXPECTED["split_seeds"]:
        folds = folds_for(frame, seed)
        save_arrays(out/f"folds-{seed}.npz", folds=folds)
        predictions = {(r, h): np.full(len(y), np.nan) for r in EXPECTED["recipes"] for h in (500, 3000)}
        constant = np.full(len(y), np.nan)
        for fold in range(5):
            outer, query = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            inner_folds = folds_for(frame.iloc[outer], 27001)
            fit, calibration = outer[inner_folds != 0], outer[inner_folds == 0]
            constant[query] = np.median(y[outer])
            for recipe in EXPECTED["recipes"]:
                key = f"s{seed}-f{fold}-{recipe}"
                record(out, {"event": "model_started", "key": key+"-selector"})
                selector = LaplaceEpochRegressor(recipe).fit(x[fit], y[fit], epochs=3000,
                                                             validation=(x[calibration], y[calibration]))
                models.append(save_model(out, key+"-selector", selector))
                record(out, {"event": "model_completed", **models[-1]})
                short, long = selector.best_epoch(500), selector.best_epoch(3000)
                if not short <= long:
                    raise ValueError("Prefix selection exceeds long selection")
                record(out, {"event": "model_started", "key": key+"-refit"})
                refit = LaplaceEpochRegressor(recipe).fit(x[outer], y[outer], epochs=long)
                models.append(save_model(out, key+"-refit", refit))
                record(out, {"event": "model_completed", **models[-1]})
                short_pred, long_pred = refit.predict(x[query], epoch=short), refit.predict(x[query], epoch=long)
                predictions[recipe, 500][query] = short_pred
                predictions[recipe, 3000][query] = long_pred
                save_arrays(out/f"{key}-query.npz", short=short_pred, long=long_pred,
                            fit=fit, calibration=calibration, outer=outer, query=query, inner_folds=inner_folds)
                unit = {"key": key, "seed": seed, "fold": fold, "recipe": recipe,
                        "short_selected_epoch": short, "long_selected_epoch": long,
                        "selector_cap_hit_selected": long == 3000,
                        "selector_zero_steps": selector.steps_.count(0.)}
                write_new(out/f"{key}-unit.json", unit)
                units.append(unit)
                peak = max(peak, resource_check())
                del selector, refit
        seed_metrics = {"M0_train_median_wmape": wmape(y, constant), "arms": {}}
        save_arrays(out/f"constant-{seed}.npz", prediction=constant)
        for recipe in EXPECTED["recipes"]:
            seed_metrics["arms"][recipe] = {}
            for horizon in (500, 3000):
                pred = predictions[recipe, horizon]
                if not np.isfinite(pred).all():
                    raise ValueError("Incomplete OOF coverage")
                save_arrays(out/f"oof-{seed}-{recipe}-{horizon}.npz", prediction=pred)
                seed_metrics["arms"][recipe][str(horizon)] = {"iron_wmape": wmape(y, pred),
                    "iron_mae": float(np.abs(y-pred).mean()), "rows": len(y)}
        metrics[str(seed)] = seed_metrics
    result = common_result("public_V2_complete_two_seed_candidate_only", models, peak)
    result.update(units=units, metrics=metrics, official_sample_reads=1,
                  G1="candidate_only_no_matched_Q75_comparison_not_promotion")
    if result["model_fits"] != 40 or result["tree_fits"] > 180000 or len(units) != 20:
        raise ValueError("Development ledger count differs")
    return result


def independent_params(model, x, epoch, calibration_y=None):
    z = (x-model.x_mean_)/model.x_scale_
    p = np.tile((0., model.initial_log_scale_), (len(x), 1))
    maes = []
    for trees, step in zip(model.trees_[:epoch], model.steps_[:epoch], strict=True):
        d = np.zeros_like(p)
        for dim, tree in enumerate(trees):
            d[:, dim] = tree.predict(z)
        p -= .05*step*d
        p[:, 1] = np.minimum(3., np.maximum(-6., p[:, 1]))
        if calibration_y is not None:
            maes.append(float(np.abs(calibration_y-(p[:, 0]*model.y_scale_+model.y_median_)).mean()))
    return p, maes


def checked_model(root, out, key, x, y):
    # Only our hash-bound local run is accepted; never unpickle user archives.
    with (out/f"{key}.pkl").open("rb") as handle:
        model = pickle.load(handle)
    trace = json.loads((out/f"{key}-trace.json").read_text(encoding="utf-8"))
    scale = x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    ys = float(y.std()) or 1.
    if (not np.array_equal(model.x_mean_, x.mean(axis=0)) or not np.array_equal(model.x_scale_, scale)
            or model.y_median_ != float(np.median(y)) or model.y_scale_ != ys
            or model.fit_rows_ != len(y) or model.history_ != trace["history"]
            or model.metadata() != trace["metadata"]):
        raise ValueError("Training-only preprocessing or trace differs")
    values = np.r_[model.initial_nll_, [h["train_nll"] for h in model.history_]]
    dimensions = 1 if model.recipe == "FIXED" else 2
    if (not np.isfinite(values).all() or not np.all(np.diff(values) <= 0)
            or model.fitted_tree_count_ != len(model.history_)*dimensions):
        raise ValueError("Training history or actual tree count differs")
    for trees in model.trees_:
        if len(trees) != dimensions:
            raise ValueError("Incorrect tree bank dimension")
        for tree in trees:
            if tree.max_depth != 3 or tree.min_samples_leaf != 20 or tree.random_state != 42:
                raise ValueError("Frozen tree recipe differs")
    return model


def check_prediction(model, query, epoch, saved):
    pred = model.predict(query, epoch=epoch)
    reverse = model.predict(query[::-1], epoch=epoch)[::-1]
    chunk = np.concatenate([model.predict(query[i:i+37], epoch=epoch) for i in range(0, len(query), 37)])
    independent, _ = independent_params(model, query, epoch)
    independent_pred = independent[:, 0]*model.y_scale_+model.y_median_
    if (not np.array_equal(pred, saved) or not np.array_equal(pred, reverse)
            or not np.array_equal(pred, chunk) or not np.isfinite(pred).all()
            or not np.allclose(pred, independent_pred, atol=1e-12, rtol=0)):
        raise ValueError("Cold/order/chunk/independent prediction mismatch")
    return float(np.max(np.abs(pred-independent_pred)))


def verify(root, out):
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    result = json.loads((out/"result.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != inventory(root):
        raise ValueError("Source identity changed")
    for name, digest in result["output_sha256"].items():
        if sha(out/name) != digest:
            raise ValueError("Saved output identity changed")
    data = np.load(out/"data.npz", allow_pickle=False)
    x, y = data["x"], data["y"]
    if manifest["action"] == "development":
        if manifest["input_sha256"] != {p: sha(root/p) for p in INPUTS}:
            raise ValueError("Public V2 input identity changed")
        frame = load_v2(root/"复赛_train", "train", 2754)
        original_x = np.column_stack((frame[list(FEATURES)].to_numpy(dtype=float), frame.spout_no == 1, frame.spout_no == 2))
        if (not np.array_equal(x, original_x) or not np.array_equal(y, frame.tap_iron.to_numpy(dtype=float))
                or not np.array_equal(data["ids"], frame.sample_id.to_numpy(dtype=str))):
            raise ValueError("Saved dataset or ID alignment differs")
    else:
        sx, sy = synthetic_arrays()
        if not np.array_equal(x, sx) or not np.array_equal(y, sy):
            raise ValueError("Engineering input differs")
    # Identity/data audit above is separate from inference. Prediction below
    # cannot read original training CSVs, and never calls model.fit().
    sys.addaudithook(deny_training_reads)
    maximum, checked = 0., []
    if manifest["action"] == "engineering":
        for recipe in EXPECTED["recipes"]:
            model = checked_model(root, out, recipe+"-complete", x[:108], y[:108])
            old = checked_model(root, out, recipe+"-legacy500", x[:108], y[:108])
            with np.load(out/f"{recipe}-query.npz", allow_pickle=False) as q:
                if not np.array_equal(q["prefix500"], old.predict(x[144:])):
                    raise ValueError("Legacy cold prefix differs")
                for name, epoch in (("prefix500", 500), ("short", model.best_epoch(500)), ("long", model.best_epoch(3000))):
                    maximum = max(maximum, check_prediction(model, x[144:], epoch, q[name]))
            _, maes = independent_params(model, x[108:144], 3000, y[108:144])
            if maes != [h["calibration_mae"] for h in model.history_]:
                raise ValueError("Calibration history independently differs")
            checked.extend((recipe+"-complete", recipe+"-legacy500"))
    else:
        for seed in EXPECTED["split_seeds"]:
            saved_folds = np.load(out/f"folds-{seed}.npz", allow_pickle=False)["folds"]
            if not np.array_equal(saved_folds, folds_for(frame, seed)):
                raise ValueError("Outer partition differs")
            reconstructed = {(r, h): np.full(len(y), np.nan) for r in EXPECTED["recipes"] for h in (500, 3000)}
            constant = np.full(len(y), np.nan)
            for unit in (u for u in result["units"] if u["seed"] == seed):
                key, fold, recipe = unit["key"], unit["fold"], unit["recipe"]
                with np.load(out/f"{key}-query.npz", allow_pickle=False) as q:
                    outer, query = np.flatnonzero(saved_folds != fold), np.flatnonzero(saved_folds == fold)
                    inner = folds_for(frame.iloc[outer], 27001)
                    fit, calibration = outer[inner != 0], outer[inner == 0]
                    if any(not np.array_equal(q[name], expected) for name, expected in
                           (("outer", outer), ("query", query), ("inner_folds", inner), ("fit", fit), ("calibration", calibration))):
                        raise ValueError("Fit/calibration/query partition differs")
                    selector = checked_model(root, out, key+"-selector", x[fit], y[fit])
                    refit = checked_model(root, out, key+"-refit", x[outer], y[outer])
                    _, maes = independent_params(selector, x[calibration], 3000, y[calibration])
                    if maes != [h["calibration_mae"] for h in selector.history_]:
                        raise ValueError("Calibration history differs")
                    short, long = selector.best_epoch(500), selector.best_epoch(3000)
                    if (short != unit["short_selected_epoch"] or long != unit["long_selected_epoch"]
                            or refit.selected_epoch_ != long or selector.initial_calibration_mae_ != float(np.abs(y[calibration]-np.median(y[fit])).mean())):
                        raise ValueError("Selected epoch differs")
                    for h, name, epoch in ((500, "short", short), (3000, "long", long)):
                        maximum = max(maximum, check_prediction(refit, x[query], epoch, q[name]))
                        reconstructed[recipe, h][query] = q[name]
                    constant[query] = np.median(y[outer])
                    checked.extend((key+"-selector", key+"-refit"))
            for recipe in EXPECTED["recipes"]:
                for h in (500, 3000):
                    pred = reconstructed[recipe, h]
                    saved = np.load(out/f"oof-{seed}-{recipe}-{h}.npz", allow_pickle=False)["prediction"]
                    expected_metric = {"iron_wmape": wmape(y, pred), "iron_mae": float(np.abs(y-pred).mean()), "rows": len(y)}
                    if not np.array_equal(pred, saved) or expected_metric != result["metrics"][str(seed)]["arms"][recipe][str(h)]:
                        raise ValueError("Complete OOF or metrics differ")
            if wmape(y, constant) != result["metrics"][str(seed)]["M0_train_median_wmape"]:
                raise ValueError("Median control differs")
    if len(checked) != result["model_fits"] or sum(m["metadata"]["fitted_tree_count"] for m in result["models"]) != result["tree_fits"]:
        raise ValueError("Audit fit ledger differs")
    return {"status": "passed_source_data_partitions_cold_checkpoints_and_independent_arithmetic",
            "checked_models": len(checked), "maximum_independent_prediction_difference": maximum,
            "audit_model_fits": 0, "Q75_gain": None, "formal_promoted": False, "packages": 0,
            "manifest_sha256": sha(out/"manifest.json"), "result_sha256": sha(out/"result.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("engineering", "development", "verify"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--engineering")
    args = parser.parse_args()
    root, out = Path.cwd().resolve(), local_output(Path.cwd().resolve(), args.output)
    validate_spec(json.loads((root/SPEC).read_text(encoding="utf-8")))
    if args.action == "verify":
        receipt = verify(root, out)
        write_new(out/"cold-readback.json", receipt)
        print(json.dumps(receipt), flush=True)
        return
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError("Freeze numerical threads to one before import")
    frozen = inventory(root)
    if args.action == "development":
        if not args.engineering:
            raise ValueError("Original engineering receipt required")
        engineering_out = local_output(root, args.engineering)
        em = json.loads((engineering_out/"manifest.json").read_text(encoding="utf-8"))
        er = json.loads((engineering_out/"cold-readback.json").read_text(encoding="utf-8"))
        if (em["action"] != "engineering" or em["source_sha256"] != frozen or er["checked_models"] != 4
                or er["status"] != "passed_source_data_partitions_cold_checkpoints_and_independent_arithmetic"
                or er["manifest_sha256"] != sha(engineering_out/"manifest.json")
                or er["result_sha256"] != sha(engineering_out/"result.json")):
            raise ValueError("Engineering admission identity failed")
    out.mkdir(parents=True, exist_ok=False)
    started = time.time_ns()
    manifest = {"identity": EXPECTED["identity"], "action": args.action, "source_sha256": frozen,
                "input_sha256": {p: sha(root/p) for p in INPUTS} if args.action == "development" else {},
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
                "pid": os.getpid(), "started_ns": started, "maximum_runtime_seconds": None, "automatic_retry": False}
    write_new(out/"manifest.json", manifest)
    print(json.dumps({"status": "started", "action": args.action, "pid": os.getpid(), "output": str(out)}), flush=True)
    try:
        result = engineering(out) if args.action == "engineering" else development(root, out)
        if inventory(root) != frozen or (args.action == "development" and manifest["input_sha256"] != {p: sha(root/p) for p in INPUTS}):
            raise ValueError("Source or input changed while running")
        result["started_ns"], result["completed_ns"] = started, time.time_ns()
        result["output_sha256"] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
        write_new(out/"result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("output_sha256", "models", "units")}), flush=True)
    except Exception as exc:
        write_new(out/"FAILED.json", {"error_type": type(exc).__name__, "error": str(exc),
                                     "automatic_retry": False, "completed_ns": time.time_ns()})
        raise


if __name__ == "__main__":
    main()
