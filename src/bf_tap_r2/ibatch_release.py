"""Release the two independent-batch averaging candidates, each a single-column change.

Both candidates only need three full-data member fits each; no new CV or confirmation seed.

* ``IBATCH_TIME_A20``: time = incumbent_time + 0.2 * (mean(3 IBATCH time members) - mean(3 EMA time members))
* ``JOINT_IBATCH_IRON_A20``: iron = incumbent_iron + 0.1 * (mean(3 joint IBATCH iron members) - mean(3 DE3 iron members))

The bases are the incumbent's own columns, verified before any column is written:

* the three EMA members are recovered from recorded artifacts and must reproduce the released
  mean3 package through its documented formula;
* the three DE3 iron members are recovered from frozen packages and must reproduce the
  original release's cold-replayed ensemble.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bf_tap_r2.component_regularization_run import RECIPE  # noqa: E402
from bf_tap_r2.de3_independent_batches_model import JointIndependentBatchRegressor  # noqa: E402
from bf_tap_r2.de3_iron_seed6_release import read_test_frame  # noqa: E402
from bf_tap_r2.ema_independent_batches_model import IndependentBatchRegressor  # noqa: E402
from bf_tap_r2.iron_strength_release import payload_bytes  # noqa: E402
from bf_tap_r2.slot_screen import file_sha256, read_json, read_package, template_ids, write_json  # noqa: E402
from bf_tap_r2.submission import ZIP_NAME, package  # noqa: E402
from bf_tap_r2.v5_library import load_v5_training_frame  # noqa: E402

TARGETS = ("tap_iron", "tap_time_len")
IRON, TIME = "pred_tap_iron", "pred_tap_time_len"
BATCH_ORDER = "independent_without_replacement"


def frozen(spec: dict, ids: list[str]) -> dict:
    out = {}
    for name, entry in spec["inputs"].items():
        path = ROOT / entry["path"]
        target = path if path.is_file() else path / ZIP_NAME
        digest = file_sha256(target)
        if digest != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
        out[name] = {"path": target, "sha256": digest, "directory": target.parent}
    for name in ("parent", "q75", "mean3", "v32"):
        key = f"{name}_package"
        out[name] = read_package(out[key]["directory"], ids)
    out["member_1042"] = np.load(out["ema_member_1042"]["path"], allow_pickle=False)["prediction"][:, 0]
    out["member_2042"] = np.load(out["ema_member_2042"]["path"], allow_pickle=False)["prediction"][:, 0]
    out["native_iron"] = np.load(out["v12_full_iron"]["path"], allow_pickle=False).astype(float)
    out["recorded_iron_ensemble"] = np.load(out["de3_release_ensemble"]["path"], allow_pickle=False).astype(float)
    return out


def bases(spec: dict, inputs: dict) -> dict:
    """Recover the incumbent's two columns and the two member means they are built from."""
    tolerance = spec["verification"]["max_absolute_difference"]
    incumbent_iron = inputs["parent"]["values"][IRON]
    incumbent_time = inputs["parent"]["values"][TIME]
    q75_time = inputs["q75"]["values"][TIME]
    mean3_time = inputs["mean3"]["values"][TIME]
    # documented mean3 formula: mean3_package = q75 + 0.75*(mean(members) - member_42)
    member_42 = (inputs["member_1042"] + inputs["member_2042"]) / 2 - 2 * (mean3_time - q75_time)
    time_members = (member_42 + inputs["member_1042"] + inputs["member_2042"]) / 3
    reconstructed_mean3 = q75_time + 0.75 * (time_members - member_42)
    iron_members = inputs["native_iron"] + 2 * (incumbent_iron - inputs["v32"]["values"][IRON])
    checks = {
        "documented_mean3_formula_reproduces_package":
            float(np.abs(reconstructed_mean3 - mean3_time).max()),
        "recovered_iron_members_match_recorded_ensemble":
            float(np.abs(iron_members - inputs["recorded_iron_ensemble"]).max()),
        "zero_new_members_reproduce_time":
            float(np.abs((incumbent_time + 0.2 * (time_members - time_members)) - incumbent_time).max()),
        "zero_new_members_reproduce_iron":
            float(np.abs((incumbent_iron + 0.1 * (iron_members - iron_members)) - incumbent_iron).max()),
    }
    for name, value in checks.items():
        if value > tolerance:
            raise ValueError(f"Base consistency gate failed: {name} = {value}")
    return {"incumbent_iron": incumbent_iron, "incumbent_time": incumbent_time,
            "time_members": time_members, "iron_members": iron_members, "checks": checks}


def fit_time_members(spec: dict, output: Path, ids: list[str]) -> np.ndarray:
    frame = load_v5_training_frame(ROOT)
    query = read_test_frame(spec)
    columns = []
    for seed in spec["time_seeds"]:
        directory = output / f"time-ibatch-seed-{seed}"
        directory.mkdir(parents=True, exist_ok=False)
        settings = dict(spec["time_training"], random_seed=seed, batch_order=BATCH_ORDER)
        model = IndependentBatchRegressor(RECIPE, settings, "EMA", spec["time_mechanisms"], directory)
        model.fit(frame.drop(columns=list(TARGETS)), frame[["tap_time_len"]].to_numpy())
        prediction = model.predict(query)[:, 0]
        np.savez_compressed(directory / "predictions.npz", prediction=prediction,
                            query_ids=query.sample_id.to_numpy(dtype=str))
        write_json(directory / "metadata.json", {"training_seed": seed,
                                                 "selected_epoch": model.metadata_["selected_epoch"],
                                                 "fit_rows": int(len(frame))})
        columns.append(prediction)
    return np.stack(columns)


def fit_iron_members(spec: dict, output: Path, ids: list[str]) -> np.ndarray:
    frame = load_v5_training_frame(ROOT)
    query = read_test_frame(spec)
    columns = []
    for seed in spec["iron_seeds"]:
        directory = output / f"iron-ibatch-seed-{seed}"
        directory.mkdir(parents=True, exist_ok=False)
        settings = dict(spec["iron_training"], random_seed=seed, batch_order=BATCH_ORDER)
        model = JointIndependentBatchRegressor(RECIPE, settings, directory)
        model.fit(frame.drop(columns=list(TARGETS)), frame[list(TARGETS)].to_numpy())
        prediction = model.predict(query)[:, 0]
        np.savez_compressed(directory / "predictions.npz", prediction=prediction,
                            query_ids=query.sample_id.to_numpy(dtype=str))
        write_json(directory / "metadata.json", {"training_seed": seed,
                                                 "selected_epoch": model.metadata_["selected_epoch"],
                                                 "fit_rows": int(len(frame))})
        columns.append(prediction)
    return np.stack(columns)


def screen(name: str, spec: dict, incumbent: np.ndarray, candidate: np.ndarray, target: str) -> dict:
    delta = candidate - incumbent
    scale = float(np.abs(incumbent).mean())
    centred = incumbent - incumbent.mean()
    slope = float((centred * (delta - delta.mean())).sum() / float((centred ** 2).sum()))
    rho = float(np.sqrt(float((delta ** 2).mean())) / scale)
    entry = {"candidate": name, "changed_target": target, "rho": rho, "slope": slope,
             "bias_pct": float(100.0 * delta.mean() / scale),
             "conditions": {"non_expansive_change": slope <= 0.0,
                            "bounded_perturbation_rho_le_0.01": rho <= 0.01}}
    entry["slot_rule_passed"] = all(entry["conditions"].values())
    weights = ROOT / spec["pre_upload_screen"]["fitted_functional"]
    if weights.exists():
        with np.load(weights, allow_pickle=False) as saved:
            key = f"{target}__weights"
            if key in saved.files:
                value = float(saved[key] @ delta)
                entry["fitted_functional"] = {
                    "predicted_platform_delta": value,
                    "predicted_sign": "positive" if value > 0 else "non_positive",
                    "status": "unvalidated_retrospective_fit_recorded_before_upload"}
    return entry


def build(spec: dict, output: Path, which: str) -> dict:
    ids = template_ids()
    inputs = frozen(spec, ids)
    base = bases(spec, inputs)
    parent = inputs["parent"]
    report = {"candidate": which, "base_consistency": base["checks"],
              "budget_actual": {"new_full_fits": 3, "new_optimizers": 6, "new_cv_fits": 0,
                                "packages": 1, "desktop_writes": 0, "agent_uploads": 0}}
    if which == spec["time_candidate"]:
        members = fit_time_members(spec, output, ids)
        candidate = base["incumbent_time"] + spec["time_weight"] * (members.mean(axis=0) - base["time_members"])
        iron_strings = list(parent["fields"][IRON])
        time_strings = [format(value, ".17g") for value in candidate]
        screen_target = TIME
    else:
        members = fit_iron_members(spec, output, ids)
        candidate = base["incumbent_iron"] + spec["iron_weight"] * (members.mean(axis=0) - base["iron_members"])
        iron_strings = [format(value, ".17g") for value in candidate]
        time_strings = list(parent["fields"][TIME])
        screen_target = IRON
    if not np.isfinite(candidate).all() or (candidate < 0).any():
        raise ValueError("Candidate column is not finite and non-negative; no clipping permitted")
    destination = output / which
    destination.mkdir(parents=True, exist_ok=False)
    package(destination, payload_bytes(ids, iron_strings, time_strings), ids)
    incumbent = base["incumbent_iron"] if screen_target == IRON else base["incumbent_time"]
    report.update({
        "output": str(destination.relative_to(ROOT)),
        "zip_sha256": file_sha256(destination / ZIP_NAME),
        "rms_change": float(np.sqrt(((candidate - incumbent) ** 2).mean())),
        "changed_strings": int(sum(1 for a, b in zip(
            [format(v, ".17g") for v in incumbent],
            iron_strings if screen_target == IRON else time_strings) if a != b)),
        "pre_upload_screen": screen(which, spec, incumbent, candidate, screen_target),
    })
    write_json(output / "report.json", report)
    return report


def audit(spec: dict, output: Path, which: str) -> dict:
    import csv
    import io
    import zipfile
    ids = template_ids()
    inputs = frozen(spec, ids)
    archive = output / which / ZIP_NAME
    with zipfile.ZipFile(archive) as handle:
        if handle.namelist() != ["result.csv"] or handle.testzip() is not None:
            raise ValueError("Zip structure or CRC failure")
        payload = handle.read("result.csv")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    values = np.array([[float(row[IRON]), float(row[TIME])] for row in rows])
    changed = IRON if which == spec["iron_candidate"] else TIME
    other = TIME if changed == IRON else IRON
    checks = {
        "zip_sha256": file_sha256(archive),
        "rows": len(rows), "unique_ids": len({row["sample_id"] for row in rows}),
        "official_order": [row["sample_id"] for row in rows] == ids,
        "finite_nonnegative": bool(np.isfinite(values).all() and (values >= 0).all()),
        "unchanged_column_verbatim": [row[other] for row in rows] == inputs["parent"]["fields"][other],
        "changed_column_differs": [row[changed] for row in rows] != inputs["parent"]["fields"][changed],
    }
    passed = all([checks["rows"] == 322, checks["unique_ids"] == 322, checks["official_order"],
                  checks["finite_nonnegative"], checks["unchanged_column_verbatim"],
                  checks["changed_column_differs"]])
    report = {"candidate": which, "audit": "independent_process_readback", "checks": checks, "passed": passed}
    write_json(output / f"independent-audit-{which}.json", report)
    return report


def screen_only(spec: dict, output: Path, which: str) -> dict:
    """Recompute a candidate's report from its packaged column; no fitting."""
    import csv
    import io
    import zipfile
    ids = template_ids()
    inputs = frozen(spec, ids)
    base = bases(spec, inputs)
    archive = output / which / ZIP_NAME
    with zipfile.ZipFile(archive) as handle:
        payload = handle.read("result.csv")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    target = IRON if which == spec["iron_candidate"] else TIME
    incumbent = base["incumbent_iron"] if target == IRON else base["incumbent_time"]
    candidate = np.array([float(row[target]) for row in rows])
    report = {"candidate": which, "recomputed_from_package": True,
              "base_consistency": base["checks"],
              "zip_sha256": file_sha256(archive),
              "rows": len(rows),
              "rms_change": float(np.sqrt(((candidate - incumbent) ** 2).mean())),
              "changed_strings": int(sum(1 for a, b in zip([format(v, ".17g") for v in incumbent],
                                                           [row[target] for row in rows]) if a != b)),
              "pre_upload_screen": screen(which, spec, incumbent, candidate, target)}
    write_json(output / f"report-{which}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/ibatch_release/SPEC.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", choices=["time", "iron", "both"], default="both")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()
    spec = read_json(ROOT / args.spec)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = ([spec["time_candidate"], spec["iron_candidate"]] if args.candidate == "both"
               else [spec["time_candidate"] if args.candidate == "time" else spec["iron_candidate"]])
    results = []
    for which in targets:
        if args.screen:
            results.append(screen_only(spec, output, which))
        elif args.audit:
            results.append(audit(spec, output, which))
        else:
            results.append(build(spec, output, which))
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
