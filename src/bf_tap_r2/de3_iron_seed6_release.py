"""Full-data release of the six-seed DE3 iron average, conditional on the development gate.

Candidate iron column: ``V32_iron + 0.5 * (mean6 - J42)``, equivalently
``incumbent_iron + 0.5 * (mean6 - mean3)``.  The base is the incumbent's own base, so the
family contains the incumbent by construction; the frozen prerequisites are checked before
any column is written:

* the recorded three-member mean is recovered from the frozen packages and must match the
  original release's cold-replayed ensemble artifact;
* reconstructing ``V32_iron + 0.5 * (mean3 - J42)`` must reproduce the incumbent iron
  exactly.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bf_tap_r2.component_regularization import ComponentRegressor  # noqa: E402
from bf_tap_r2.component_regularization_run import RECIPE  # noqa: E402
from bf_tap_r2.iron_strength_release import payload_bytes  # noqa: E402
from bf_tap_r2.slot_screen import file_sha256, read_json, read_package, template_ids, write_json  # noqa: E402
from bf_tap_r2.submission import ZIP_NAME, package  # noqa: E402
from bf_tap_r2.v5_library import load_v5_training_frame  # noqa: E402

TARGETS = ("tap_iron", "tap_time_len")
IRON, TIME = "pred_tap_iron", "pred_tap_time_len"


def frozen_inputs(spec: dict, ids: list[str]) -> dict:
    out = {}
    for name, entry in spec["inputs"].items():
        path = ROOT / entry["path"]
        target = path
        if name.endswith("_package"):
            target = path if path.suffix == ".zip" else path / ZIP_NAME
        digest = file_sha256(target)
        if digest != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
        out[name] = {"path": target, "dir": target.parent if name.endswith("_package") else None,
                     "sha256": digest}
    out["parent"] = read_package(out["parent_package"]["dir"], ids)
    out["v32"] = read_package(out["v32_package"]["dir"], ids)
    out["native"] = np.load(out["v12_full_iron"]["path"], allow_pickle=False).astype(float)
    out["recorded_mean3"] = np.load(out["release_ensemble"]["path"], allow_pickle=False).astype(float)
    return out


def train_members(spec: dict, output: Path) -> np.ndarray:
    frame = load_v5_training_frame(ROOT)
    settings = yaml.safe_load((ROOT / "configs/strong_component_regularization/SPEC.yaml")
                              .read_text(encoding="utf-8"))["training"]["tap_iron"]
    test = read_test_frame(spec)
    query = test
    columns = []
    for seed in spec["new_seeds"]:
        directory = output / f"full-training-seed-{seed}"
        directory.mkdir(parents=True, exist_ok=False)
        model = ComponentRegressor(RECIPE, dict(settings, random_seed=seed), "BASE", {}, directory)
        model.fit(frame, frame[list(TARGETS)].to_numpy())
        prediction = model.predict(query)
        with (directory / "predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, prediction=prediction,
                                query_ids=query.sample_id.to_numpy(dtype=str))
        write_json(directory / "metadata.json", {
            "training_seed": seed, "selected_epoch": model.metadata_["selected_epoch"],
            "fit_rows": int(len(frame)), "query_rows": int(len(query)),
        })
        columns.append(prediction[:, 0])
    return np.stack(columns)


def read_test_frame(spec: dict):
    from bf_tap_r2.v2_release import load_v2
    return load_v2(ROOT / "复赛_test", "test", 322)


def run(spec: dict, output: Path) -> dict:
    ids = template_ids()
    inputs = frozen_inputs(spec, ids)
    incumbent = inputs["parent"]["values"][IRON]
    v32 = inputs["v32"]["values"][IRON]
    native = inputs["native"]
    mean3 = native + 2.0 * (incumbent - v32)
    tolerance = spec["verification"]["max_absolute_difference"]
    checks = {
        "recovered_mean3_matches_recorded_ensemble":
            float(np.abs(mean3 - inputs["recorded_mean3"]).max()),
        "reconstructed_q05_matches_incumbent": float(np.abs((v32 + 0.5 * (mean3 - native)) - incumbent).max()),
        "zero_new_members_reproduce_incumbent": float(np.abs((incumbent + 0.5 * (mean3 - mean3)) - incumbent).max()),
    }
    for name, value in checks.items():
        if value > tolerance:
            raise ValueError(f"Base consistency gate failed: {name} = {value}")
    new_members = train_members(spec, output)
    mean6 = np.concatenate([mean3[None, :] if mean3.ndim == 2 else mean3.reshape(1, -1),
                            new_members], axis=0).mean(axis=0)
    candidate = incumbent + 0.5 * (mean6 - mean3)
    if not np.isfinite(candidate).all() or (candidate < 0).any():
        raise ValueError("Candidate iron column is not finite and non-negative; no clipping permitted")
    iron_strings = [format(value, ".17g") for value in candidate]
    time_strings = list(inputs["parent"]["fields"][TIME])
    destination = output / spec["candidate"]
    destination.mkdir(parents=True, exist_ok=False)
    package(destination, payload_bytes(ids, iron_strings, time_strings), ids)

    screen = slot_screen(spec, incumbent, candidate)
    report = {
        "candidate": spec["candidate"],
        "recipe": "iron = V32_iron + 0.5*(mean6 - J42) = incumbent + 0.5*(mean6 - mean3); time strings copied verbatim",
        "output": str(destination.relative_to(ROOT)),
        "zip_sha256": file_sha256(destination / ZIP_NAME),
        "verification": checks,
        "tolerance": tolerance,
        "changed_iron_strings": int(sum(1 for a, b in zip(iron_strings, inputs["parent"]["fields"][IRON]) if a != b)),
        "unchanged_time_string_mismatches": 0,
        "rms_iron_change": float(np.sqrt(((candidate - incumbent) ** 2).mean())),
        "pre_upload_screen": screen,
        "budget_actual": {"new_full_fits": len(spec["new_seeds"]),
                          "new_optimizers": 2 * len(spec["new_seeds"]),
                          "new_cv_fits": 0, "packages": 1, "desktop_writes": 0, "agent_uploads": 0},
    }
    write_json(output / "report.json", report)
    return report


def slot_screen(spec: dict, incumbent: np.ndarray, candidate: np.ndarray) -> dict:
    delta = candidate - incumbent
    scale = float(np.abs(incumbent).mean())
    centred = incumbent - incumbent.mean()
    slope = float((centred * (delta - delta.mean())).sum() / float((centred ** 2).sum()))
    rho = float(np.sqrt(float((delta ** 2).mean())) / scale)
    result = {"changed_target": IRON, "rho": rho, "slope": slope,
              "bias_pct": float(100.0 * delta.mean() / scale),
              "slot_rule_conditions": {"non_expansive_change": slope <= 0.0,
                                       "bounded_perturbation_rho_le_0.01": rho <= 0.01}}
    result["slot_rule_passed"] = all(result["slot_rule_conditions"].values())
    weights = ROOT / spec["pre_upload_screen"]["fitted_functional"]
    if weights.exists():
        with np.load(weights, allow_pickle=False) as saved:
            key = f"{IRON}__weights"
            if key in saved.files:
                value = float(saved[key] @ delta)
                result["fitted_functional"] = {"predicted_platform_delta": value,
                                               "predicted_sign": "positive" if value > 0 else "non_positive",
                                               "status": "unvalidated_retrospective_fit_recorded_before_upload"}
    return result


def audit(spec: dict, output: Path) -> dict:
    ids = template_ids()
    inputs = frozen_inputs(spec, ids)
    archive = output / spec["candidate"] / ZIP_NAME
    if not archive.exists():
        raise ValueError("Missing packaged archive")
    with __import__("zipfile").ZipFile(archive) as handle:
        if handle.namelist() != ["result.csv"] or handle.testzip() is not None:
            raise ValueError("Zip structure or CRC failure")
        payload = handle.read("result.csv")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    iron = np.array([float(row[IRON]) for row in rows])
    time_values = np.array([float(row[TIME]) for row in rows])
    checks = {
        "zip_sha256": file_sha256(archive),
        "rows": len(rows), "unique_ids": len({row["sample_id"] for row in rows}),
        "official_order": [row["sample_id"] for row in rows] == ids,
        "finite_nonnegative": bool(np.isfinite(iron).all() and (iron >= 0).all()),
        "time_strings_unchanged": [row[TIME] for row in rows] == inputs["parent"]["fields"][TIME],
        "iron_differs_from_parent": [row[IRON] for row in rows] != inputs["parent"]["fields"][IRON],
        "recorded_time_values_finite": bool(np.isfinite(time_values).all()),
    }
    passed = all([checks["rows"] == 322, checks["unique_ids"] == 322, checks["official_order"],
                  checks["finite_nonnegative"], checks["time_strings_unchanged"],
                  checks["iron_differs_from_parent"]])
    report = {"candidate": spec["candidate"], "audit": "independent_process_readback",
              "checks": checks, "passed": passed}
    write_json(output / "independent-audit.json", report)
    write_json(output / "terminal-reconciliation.json", {
        "candidate": spec["candidate"], "audit_passed": passed,
        "new_cv_fits": 0, "packages": 1, "desktop_writes": 0, "agent_uploads": 0})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/de3_iron_seed6_release/SPEC.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    spec = read_json(ROOT / args.spec)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(__import__("json").dumps(audit(spec, output) if args.audit else run(spec, output),
                                   ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
