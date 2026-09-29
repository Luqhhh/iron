"""One-shot small synthetic witness; this never admits official-data training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from .data import FEATURES
from .dcn_cross import ARMS, CrossRegressor, clean, fit_partition
from .dcn_cross_verify import independent_prediction, verify_model
from .v7_periodic import file_hash, write_new

SPEC = "configs/dcn_cross_preparation/SPEC.yaml"


def fixture(spec):
    cfg = spec["synthetic"]
    rng = np.random.default_rng(cfg["seed"])
    x = rng.normal(size=(cfg["rows"], len(FEATURES)))
    y = 20 + 3*x[:, 0] + 4*x[:, 1]*x[:, 2] + 2*x[:, 3]*x[:, 4]*x[:, 5] + .1*rng.normal(size=len(x))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["sample_id"] = [f"dcn-synthetic-{i}" for i in range(len(x))]
    frame["spout_no"] = np.arange(len(x)) % 2 + 1
    frame["tap_iron"] = y
    a, b = cfg["fitting_rows"], cfg["fitting_rows"] + cfg["calibration_rows"]
    return frame.iloc[:a], frame.iloc[a:b], frame.iloc[:b], clean(frame.iloc[b:]), y[b:]


def check_spec(spec):
    if (spec["stage"] != "numerical_and_small_synthetic_preparation_only"
            or spec["arms"] != list(ARMS) or spec["synthetic"]["arm_order"] != list(ARMS)
            or spec["synthetic"]["official_fit_admission"] is not False
            or spec["data"]["official_label_reads"] != 0
            or spec["budget"]["official_data_fits"] != 0
            or spec["budget"]["small_synthetic_procedures"] != 2
            or spec["budget"]["small_synthetic_optimizer_runs"] != 4
            or sum(spec["synthetic"][k] for k in ("fitting_rows", "calibration_rows", "query_rows"))
               != spec["synthetic"]["rows"]):
        raise ValueError("Preparation-only budget/spec mismatch")


def sources(root):
    names = [SPEC, "docs/dcn_cross_preparation/DESIGN.md", "tests/test_dcn_cross.py",
             "uv.lock", "pyproject.toml", "src/bf_tap_r2/data.py", "src/bf_tap_r2/v7_periodic.py"]
    names += [str(p.relative_to(root)) for p in (root/"src/bf_tap_r2").glob("dcn_cross*.py")]
    return {n: file_hash(root/n) for n in sorted(set(names))}


def verify_hashes(root, hashes):
    for name, value in hashes.items():
        if file_hash(root/name) != value:
            raise ValueError(f"Frozen synthetic source changed: {name}")


def audit(root, directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    verify_hashes(root, manifest["source_hashes"])
    spec = yaml.safe_load((root/SPEC).read_text())
    check_spec(spec)
    fitting, calibration, training, query, qy = fixture(spec)
    receipt = json.loads((directory/"completion.json").read_text())
    rows = []
    for arm in ARMS:
        unit = directory/arm
        hashes = receipt["hashes"][arm]
        for name, sha in hashes.items():
            if file_hash(unit/name) != sha:
                raise ValueError("Synthetic artifact hash mismatch")
        traces = json.loads((unit/"traces.json").read_text())
        selector = verify_model(unit/"selector.json", fitting, fitting.tap_iron.to_numpy(), arm,
                                spec["training"], hashes["selector.json"], traces["selector"])
        refit = verify_model(unit/"refit.json", training, training.tap_iron.to_numpy(), arm,
                             spec["training"], hashes["refit.json"], traces["refit"])
        if (selector["selected_epoch"] != refit["selected_epoch"]
                or selector["initial_state_digest"] != refit["initial_state_digest"]):
            raise ValueError("Fresh refit epoch/initialization mismatch")
        with np.load(unit/"predictions.npz", allow_pickle=False) as p:
            pred, cp = p["query"].copy(), p["calibration"].copy()
            if not np.array_equal(p["query_ids"], query.sample_id.to_numpy(dtype=str)):
                raise ValueError("Synthetic query row identity mismatch")
        cal = independent_prediction(unit/"selector.json", clean(calibration), hashes["selector.json"])
        val = float(np.abs(cal-calibration.tap_iron.to_numpy()).mean()/selector["target_std"])
        expected = selector["history"][selector["selected_epoch"]-1]["calibration_standardized_mae"]
        if abs(val-expected) > 1e-10:
            raise ValueError("Selected calibration checkpoint differs from trace")
        cold = CrossRegressor.load(unit/"refit.json", hashes["refit.json"])
        checks = [cold.predict(query), cold.predict(query.iloc[::-1])[::-1],
                  np.concatenate([cold.predict(query.iloc[i:i+7]) for i in range(0, len(query), 7)]),
                  independent_prediction(unit/"refit.json", query, hashes["refit.json"])]
        differences = [float(np.max(np.abs(p-pred))) for p in checks]
        differences.append(float(np.max(np.abs(cal-cp))))
        if max(differences) > 1e-10:
            raise ValueError("Synthetic cold/independent/order/chunk mismatch")
        rows.append({"arm": arm, "selected_epoch": selector["selected_epoch"],
                     "stopped_epoch": selector["stopped_epoch"], "parameter_count": refit["parameter_count"],
                     "query_mae": float(np.abs(qy-pred).mean()),
                     "constant_query_mae": float(np.abs(qy-training.tap_iron.mean()).mean()),
                     "maximum_inference_difference": max(differences)})
    if rows[0]["parameter_count"] != rows[1]["parameter_count"]:
        raise ValueError("Arm parameter counts differ")
    report = {"status": "passed", "scope": "small_synthetic_only", "rows": rows,
              "learnability_both_arms_beat_constant": all(r["query_mae"] < r["constant_query_mae"] for r in rows),
              "optimizer_runs": 4, "official_fit_admission": False, "official_label_reads": 0,
              "official_fits": 0, "full_size_resource_admission": False,
              "manifest_sha256": file_hash(directory/"manifest.json"),
              "completion_sha256": file_hash(directory/"completion.json"), "audit_new_fits": 0}
    write_new(directory/"audit.json", report)
    print(json.dumps(report), flush=True)
    return report


def run(root, output):
    directory = (root/output).resolve()
    if not directory.is_relative_to((root/"local/research/dcn-cross-preparation").resolve()):
        raise ValueError("Private preparation directory required")
    spec = yaml.safe_load((root/SPEC).read_text())
    check_spec(spec)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"source_hashes": sources(root), "spec_sha256": file_hash(root/SPEC),
                "budget_optimizer_runs": 4, "official_label_reads": 0}
    write_new(directory/"manifest.json", manifest)
    fitting, calibration, training, query, _ = fixture(spec)
    hashes = {}
    try:
        for arm in ARMS:
            unit = directory/arm
            unit.mkdir()
            write_new(unit/"start.json", {"arm": arm, "reserved_optimizer_runs": 2, "time_ns": time.time_ns()})
            refit, selector, pred, cp = fit_partition(fitting, calibration, training, query,
                                                     "tap_iron", arm, spec["training"])
            selector.save(unit/"selector.json"); refit.save(unit/"refit.json")
            write_new(unit/"traces.json", {"selector": selector.metadata(), "refit": refit.metadata()})
            with (unit/"predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, query=pred, calibration=cp,
                                    query_ids=query.sample_id.to_numpy(dtype=str))
            hashes[arm] = {n: file_hash(unit/n) for n in ("selector.json", "refit.json", "traces.json", "predictions.npz")}
        verify_hashes(root, manifest["source_hashes"])
        write_new(directory/"completion.json", {"hashes": hashes, "optimizer_runs": 4})
        subprocess.run([sys.executable, "-m", "bf_tap_r2.dcn_cross_synthetic", "--audit", str(directory)],
                       cwd=root, check=True)
    except BaseException as exc:
        write_new(directory/"failure.json", {"error": repr(exc), "no_implicit_retry": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--audit", type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.audit:
        directory = args.audit.resolve()
        if not directory.is_relative_to((root/"local/research/dcn-cross-preparation").resolve()):
            raise ValueError("Private preparation directory required")
        audit(root, directory)
    else:
        run(root, args.output)


if __name__ == "__main__":
    main()
