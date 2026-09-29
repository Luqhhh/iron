"""Separate synthetic/resource admissions for the two serial queues."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

from .component_regularization import ComponentRegressor
from .component_regularization_run import RECIPE, outputs
from .data import FEATURES, TARGETS
from .independent_checkpoints import EpochSelector, clean_query, native_parts, fit_background, fresh_refit
from .independent_checkpoints_audit import cold_check, verify_epoch_unit
from .independent_checkpoints_run import SPEC, RUN_ROOT, load_spec, sources, validate_native, verify_original_reference
from .v12_joint import JointRegressor
from .v5_library import load_v5_training_frame, fold_vector
from .v5_spec import load_v5_spec
from .v7_periodic import file_hash, write_new
from .v49_run import append_event, check_runtime, verify_hashes, verify_reference_cache


def synthetic():
    rng = np.random.default_rng(53001); x = rng.normal(size=(2754, len(FEATURES)))
    z = 3*x[:, 0]+2*np.sin(x[:, 1])+x[:, 2]*x[:, 3]+.2*rng.normal(size=len(x))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["sample_id"] = [f"synthetic-independent-{i}" for i in range(len(x))]
    frame["spout_no"] = rng.integers(1, 3, len(x))
    frame["tap_iron"] = 500+10*z; frame["tap_time_len"] = 100+2*z
    return frame.iloc[:2204].reset_index(drop=True), frame.iloc[2204:].reset_index(drop=True)


def fit_synthetic(out, spec, target, train_seed, queue):
    directory = out/f"{target}-training-seed-{train_seed}"; directory.mkdir(exist_ok=False)
    training, query = synthetic(); settings = dict(spec["training"][target], random_seed=train_seed)
    y = training[outputs(target)].to_numpy(); started = time.monotonic()
    if queue == "E-COMPOSE":
        with np.load(out/"inner-background/predictions.npz", allow_pickle=False) as background:
            selector = EpochSelector(settings, target, directory)
            selector.select(training, background[target], clean_query(query))
        native = JointRegressor(RECIPE, settings).fit(training, y)
        if native.metadata_["selected_epoch"] != selector.selection_["epochs"]["E-NATIVE"]:
            raise ValueError("Synthetic native epoch replay")
        # Full refit identity is checked separately below at the exact chosen epoch.
        replay_dir = directory/"native-replay"; replay_dir.mkdir()
        replay = fresh_refit(training, target, settings, native.metadata_["selected_epoch"], replay_dir)
        np.testing.assert_array_equal(replay.predict(clean_query(query)), native.predict(clean_query(query)))
        refit_dir = directory/"full-refit"; refit_dir.mkdir()
        started_refit = time.monotonic()
        model = fresh_refit(training, target, settings, 240, refit_dir)
        selector_fits, refit_fits, native_control_fits = 1, 2, 2
    else:
        model = ComponentRegressor(RECIPE, settings, "BASE", {}, directory)
        model._initialize(training, y)
        initial_hash = __import__("hashlib").sha256(b"".join(v.detach().numpy().tobytes() for v in model.model_.state_dict().values())).hexdigest()
        started_refit = time.monotonic()
        model._train(training, y, 240)
        selector_fits, refit_fits, native_control_fits = 0, 1, 0
    refit_seconds = time.monotonic()-started_refit
    prediction = model.predict(clean_query(query))
    with (directory/"prediction.npy").open("xb") as stream: np.save(stream, prediction, allow_pickle=False)
    mae = np.abs(query[outputs(target)].to_numpy()-prediction).mean(axis=0)
    constant = np.abs(query[outputs(target)].to_numpy()-np.median(y, axis=0)).mean(axis=0)
    row = {"target": target, "training_seed": train_seed, "seconds": time.monotonic()-started,
           "refit_seconds": refit_seconds, "mae": mae.tolist(), "constant_mae": constant.tolist(),
           "peak_rss_mib": model.traces["refit"]["peak_rss_mib"],
           "selector_fits": selector_fits, "refit_fits": refit_fits, "native_control_fits": native_control_fits}
    if queue == "DE3": row["initial_state_sha256"] = initial_hash
    write_new(directory/"result.json", row)
    return row


def cold(root, out, queue):
    spec = load_spec(root); training, query = synthetic(); rows = []
    for directory in sorted(out.glob("tap_*-training-seed-*")):
        row = json.loads((directory/"result.json").read_text()); target = row["target"]
        path = directory/("full-refit/refit.pt" if queue == "E-COMPOSE" else "refit.pt")
        model = ComponentRegressor.load(path)
        from .component_regularization_audit import verify_saved
        settings = dict(spec["training"][target], random_seed=row["training_seed"])
        verify_saved(path, training, training[outputs(target)].to_numpy(), "BASE", settings, {}, expected_epoch=240)
        difference = cold_check(model, clean_query(query), np.load(directory/"prediction.npy", allow_pickle=False), spec["preflight"]["cold_predict_atol"])
        if queue == "E-COMPOSE":
            _, _, calibration = native_parts(training)
            with np.load(out/"inner-background/predictions.npz", allow_pickle=False) as bg:
                verify_epoch_unit(directory, training, calibration, bg[target], target, settings)
        rows.append({"target": target, "training_seed": row["training_seed"], "difference": difference})
    print(json.dumps(rows))


def run(root, queue):
    root = Path(root).resolve(); spec = load_spec(root); check_runtime(spec); verify_original_reference(root, spec)
    if queue == "E-COMPOSE" and json.loads((root/RUN_ROOT/"development-DE3/audit.json").read_text())["status"] != "passed":
        raise ValueError("Serial DE3 completion required")
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec["split_seeds"]}
    validate_native(root, spec, frame, folds)
    out = root/RUN_ROOT/f"preflight-{queue}"; out.mkdir(parents=True, exist_ok=False)
    frozen = sources(root)
    write_new(out/"manifest.json", {"spec_sha256": file_hash(root/SPEC), "source_hashes": frozen, "queue": queue})
    reference_seconds = 0.; reference_peak = 0.
    if queue == "E-COMPOSE":
        training, query = synthetic(); _, fitting, calibration = native_parts(training)
        directory = out/"inner-background"; directory.mkdir()
        values, meta = fit_background(root, fitting, calibration, spec, directory)
        reference_seconds = meta["seconds"]
        reference_peak = meta["peak_rss_mib"]
        with (directory/"predictions.npz").open("xb") as stream: np.savez_compressed(stream, **values, query_ids=calibration.sample_id.to_numpy(dtype=str))
        write_new(directory/"metadata.json", meta)
    seeds = spec["training_seeds"][1:] if queue == "DE3" else [42]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(fit_synthetic, out, spec, t, ts, queue): (t, ts) for t in TARGETS for ts in seeds}
        for job in as_completed(jobs):
            try: row = job.result(); rows.append(row); append_event(out/"events.jsonl", {"event": "complete", **row})
            except BaseException as exc:
                append_event(out/"events.jsonl", {"event": "failed", "unit": jobs[job], "error": repr(exc)}); raise
    done = subprocess.run([sys.executable, "-m", "bf_tap_r2.independent_checkpoints_preflight", "--cold", str(out), "--queue", queue],
                          cwd=root, capture_output=True, text=True, check=True)
    cold_rows = json.loads(done.stdout)
    solver_count = 80 if queue == "DE3" else 60
    projection = (2*max(r["refit_seconds"] for r in rows)*solver_count/4+2*reference_seconds*10)/3600
    peak = max(reference_peak, max(r["peak_rss_mib"] for r in rows))
    checks = {"learnability": all(all(m < c for m, c in zip(r["mae"], r["constant_mae"])) for r in rows),
              "projected_hours": projection <= spec["preflight"]["max_projected_hours"],
              "worker_memory": peak <= spec["preflight"]["max_worker_rss_mib"],
              "available_memory": 4*peak+1024 < psutil.virtual_memory().available/1024**2,
              "native_cache_verified": True, "cold_audit": True}
    if queue == "DE3":
        checks["independent_initializations"] = all(len({r["initial_state_sha256"] for r in rows if r["target"] == t}) == 2 for t in TARGETS)
    verify_hashes(root, frozen)
    report = {"status": "passed" if all(checks.values()) else "failed", "queue": queue, "checks": checks,
              "projected_hours": projection, "peak_rss_mib": peak, "results": rows, "cold_results": cold_rows,
              "synthetic_reference_factory_calls": int(queue == "E-COMPOSE"),
              "synthetic_reference_pipeline_fits": 30 if queue == "E-COMPOSE" else 0,
              "reference_seconds": reference_seconds,
              "synthetic_selector_fits": sum(r["selector_fits"] for r in rows),
              "synthetic_refit_fits": sum(r["refit_fits"] for r in rows),
              "native_control_fits": sum(r["native_control_fits"] for r in rows),
              "spec_sha256": file_hash(root/SPEC), "source_hashes": frozen, "official_fits": 0}
    write_new(out/"report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "source_hashes"}), flush=True)
    if report["status"] != "passed": raise ValueError("Synthetic/resource admission refused")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--queue", choices=["DE3", "E-COMPOSE"], required=True)
    parser.add_argument("--cold", type=Path); args = parser.parse_args()
    if args.cold: cold(Path.cwd(), args.cold, args.queue)
    else: run(Path.cwd(), args.queue)
