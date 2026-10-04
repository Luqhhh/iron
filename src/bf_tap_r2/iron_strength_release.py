"""Zero-fit release of the DE3 iron strength endpoint q=1.0.

The incumbent iron column is ``V32_iron + 0.5 * (mean3 - J42)`` where ``mean3`` is the
arithmetic mean of the three full-data DE3 members and ``J42`` is the native V12 member.
Inverting that replacement gives ``mean3 = J42 + 2 * (incumbent_iron - V32_iron)``, which
the recorded release ensemble artifact reproduces to 1.1e-13.  The q=1.0 candidate is
exactly that ``mean3`` column, so no model has to be loaded or trained.

See ``docs/iron_strength_release/PREREGISTRATION.md``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bf_tap_r2.slot_screen import file_sha256, read_json, read_package, template_ids, write_json  # noqa: E402
from bf_tap_r2.slot_screen import geometry  # noqa: E402
from bf_tap_r2.submission import ZIP_NAME, package, validate_result  # noqa: E402

CANDIDATE = "DE3_IRON_STRENGTH_100"
IRON_KEY = "pred_tap_iron"
TIME_KEY = "pred_tap_time_len"


def load_inputs(spec: dict, expected_ids: list[str]) -> dict:
    inputs = {}
    for name in ("parent_package", "v32_package", "de3_release_package", "v12_full_iron", "release_ensemble"):
        entry = spec["inputs"][name]
        raw = ROOT / entry["path"]
        is_package = name.endswith("_package")
        if is_package:
            target = raw if raw.suffix == ".zip" else raw / ZIP_NAME
            package_dir = target.parent
        else:
            target, package_dir = raw, None
        digest = file_sha256(target)
        if digest != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
        inputs[name] = {"path": target, "sha256": digest, "package_dir": package_dir}
    inputs["parent"] = read_package(inputs["parent_package"]["package_dir"], expected_ids)
    inputs["v32"] = read_package(inputs["v32_package"]["package_dir"], expected_ids)
    inputs["de3_release"] = read_package(inputs["de3_release_package"]["package_dir"], expected_ids)
    inputs["v12_full"] = np.load(inputs["v12_full_iron"]["path"], allow_pickle=False)
    inputs["ensemble"] = np.load(inputs["release_ensemble"]["path"], allow_pickle=False)
    return inputs


def derive(inputs: dict) -> dict:
    parent_iron = inputs["parent"]["values"][IRON_KEY]
    v32_iron = inputs["v32"]["values"][IRON_KEY]
    native = np.asarray(inputs["v12_full"], dtype=float)
    recorded_ensemble = np.asarray(inputs["ensemble"], dtype=float)
    mean3 = native + 2.0 * (parent_iron - v32_iron)
    recon_q05 = v32_iron + 0.5 * (mean3 - native)
    return {
        "mean3": mean3,
        "recon_q05": recon_q05,
        "checks": {
            "incumbent_iron_equals_de3_release_iron_strings":
                inputs["parent"]["fields"][IRON_KEY] == inputs["de3_release"]["fields"][IRON_KEY],
            "reconstructed_q05_matches_incumbent": float(np.abs(recon_q05 - parent_iron).max()),
            "derived_mean3_matches_recorded_release_ensemble":
                float(np.abs(mean3 - recorded_ensemble).max()),
            "v12_full_rows": int(native.shape[0]),
        },
    }


def run(spec: dict, output: Path) -> dict:
    expected_ids = template_ids()
    inputs = load_inputs(spec, expected_ids)
    derived = derive(inputs)
    checks = derived["checks"]
    tolerance = spec["verification"]["max_absolute_difference"]
    if not checks["incumbent_iron_equals_de3_release_iron_strings"]:
        raise ValueError("The incumbent no longer carries the DE3 release iron strings")
    if checks["reconstructed_q05_matches_incumbent"] > tolerance:
        raise ValueError("The q=0.5 reconstruction does not reproduce the incumbent column")
    if checks["derived_mean3_matches_recorded_release_ensemble"] > tolerance:
        raise ValueError("The derived three-member mean does not match the recorded ensemble")
    mean3 = derived["mean3"]
    if not np.isfinite(mean3).all() or (mean3 < 0).any():
        raise ValueError("The q=1.0 column is not finite and non-negative; no clipping is permitted")

    iron = [format(value, ".17g") for value in mean3]
    time_strings = list(inputs["parent"]["fields"][TIME_KEY])
    payload = payload_bytes(expected_ids, iron, time_strings)
    rows = validate_result(payload, expected_ids)
    if [row[TIME_KEY] for row in rows] != time_strings:
        raise ValueError("Time column strings changed")

    destination = output / CANDIDATE
    destination.mkdir(parents=True, exist_ok=False)
    package(destination, payload, expected_ids)

    reread = _read_zip(destination / ZIP_NAME)
    if reread["fields"][TIME_KEY] != inputs["parent"]["fields"][TIME_KEY]:
        raise ValueError("Packaged time strings changed")
    if [row[IRON_KEY] for row in reread["rows"]] != iron:
        raise ValueError("Packaged iron strings changed")
    if reread["fields"][IRON_KEY] == inputs["parent"]["fields"][IRON_KEY]:
        raise ValueError("Packaged iron column is unchanged")

    screen = screen_prediction(spec, inputs, mean3)
    report = {
        "candidate": CANDIDATE,
        "recipe": "iron = J42 + 2*(incumbent_iron - V32_iron) = mean of the three DE3 full-data members; time strings copied verbatim",
        "output": str(destination.relative_to(ROOT)),
        "zip_sha256": file_sha256(destination / ZIP_NAME),
        "rows": len(rows),
        "unique_ids": len({row["sample_id"] for row in rows}),
        "verification": checks,
        "tolerance": tolerance,
        "unchanged_time_string_mismatches": 0,
        "changed_iron_strings": int(sum(1 for a, b in zip(iron, inputs["parent"]["fields"][IRON_KEY]) if a != b)),
        "pre_upload_screen": screen,
        "budget_actual": {"new_fits": 0, "new_optimizers": 0, "model_loads": 0,
                          "cold_inferences": 0, "packages": 1, "desktop_writes": 0,
                          "agent_uploads": 0},
    }
    write_json(output / "report.json", report)
    write_json(output / "verification.json", {
        "inputs": {name: {"path": str(entry["path"].relative_to(ROOT)), "sha256": entry["sha256"]}
                   for name, entry in inputs.items() if isinstance(entry, dict) and "sha256" in entry},
        "checks": checks, "tolerance": tolerance,
        "reconstructed_q05_equals_incumbent": checks["reconstructed_q05_matches_incumbent"] == 0.0,
    })
    return report


def payload_bytes(ids: list[str], iron_strings: list[str], time_strings: list[str]) -> bytes:
    if not (len(ids) == len(iron_strings) == len(time_strings)):
        raise ValueError("Column length mismatch")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("sample_id", IRON_KEY, TIME_KEY))
    for sample_id, iron, time in zip(ids, iron_strings, time_strings):
        writer.writerow((sample_id, iron, time))
    return stream.getvalue().encode("utf-8")


def _read_zip(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as handle:
        if handle.namelist() != ["result.csv"]:
            raise ValueError("Unexpected zip members")
        if handle.testzip() is not None:
            raise ValueError("CRC failure")
        payload = handle.read("result.csv")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    return {
        "rows": rows,
        "fields": {IRON_KEY: [row[IRON_KEY] for row in rows],
                   TIME_KEY: [row[TIME_KEY] for row in rows]},
    }


def screen_prediction(spec: dict, inputs: dict, mean3: np.ndarray) -> dict:
    """Frozen slot rule plus the unvalidated test-side functional, recorded before upload."""
    parent_column = inputs["parent"]["values"][IRON_KEY]
    delta = mean3 - parent_column
    scale = float(np.abs(parent_column).mean())
    centred = parent_column - parent_column.mean()
    slope = float((centred * (delta - delta.mean())).sum() / float((centred ** 2).sum()))
    rho = float(np.sqrt(float((delta ** 2).mean())) / scale)
    result = {
        "changed_target": IRON_KEY,
        "rho": rho,
        "bias_pct": float(100.0 * delta.mean() / scale),
        "slope": slope,
        "slot_rule_conditions": {"non_expansive_change": slope <= 0.0,
                                 "bounded_perturbation_rho_le_0.01": rho <= 0.01},
    }
    result["slot_rule_passed"] = all(result["slot_rule_conditions"].values())
    weights_path = ROOT / spec["pre_upload_screen"]["fitted_functional"]
    if weights_path.exists():
        with np.load(weights_path, allow_pickle=False) as saved:
            key = f"{IRON_KEY}__weights"
            if key in saved.files:
                weights = saved[key]
                value = float(weights @ delta)
                result["fitted_functional"] = {
                    "path": spec["pre_upload_screen"]["fitted_functional"],
                    "predicted_platform_delta": value,
                    "predicted_sign": "positive" if value > 0 else "non_positive",
                    "status": "unvalidated_retrospective_fit_recorded_before_upload",
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/iron_strength_release/SPEC.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    spec = read_json(ROOT / args.spec)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = audit(spec, output) if args.audit else run(spec, output)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()


def audit(spec: dict, output: Path) -> dict:
    """Fresh-process readback: re-derive the column from frozen inputs and audit the ZIP."""
    expected_ids = template_ids()
    inputs = load_inputs(spec, expected_ids)
    derived = derive(inputs)
    archive = output / CANDIDATE / ZIP_NAME
    if not archive.exists():
        raise ValueError(f"Missing packaged archive: {archive}")
    reread = _read_zip(archive)
    rows = reread["rows"]
    ids = [row["sample_id"] for row in rows]
    values = np.array([[float(row[IRON_KEY]), float(row[TIME_KEY])] for row in rows])
    checks = {
        "zip_sha256": file_sha256(archive),
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "official_order": ids == expected_ids,
        "finite_nonnegative": bool(np.isfinite(values).all() and (values >= 0).all()),
        "time_strings_unchanged": reread["fields"][TIME_KEY] == inputs["parent"]["fields"][TIME_KEY],
        "iron_strings_equal_derivation": reread["fields"][IRON_KEY] == [format(v, ".17g") for v in derived["mean3"]],
        "iron_differs_from_parent": reread["fields"][IRON_KEY] != inputs["parent"]["fields"][IRON_KEY],
        "reconstructed_q05_matches_incumbent": derived["checks"]["reconstructed_q05_matches_incumbent"],
        "derived_mean3_matches_recorded_release_ensemble":
            derived["checks"]["derived_mean3_matches_recorded_release_ensemble"],
    }
    tolerance = spec["verification"]["max_absolute_difference"]
    passed = all([
        checks["rows"] == 322 and checks["unique_ids"] == 322, checks["official_order"],
        checks["finite_nonnegative"], checks["time_strings_unchanged"],
        checks["iron_strings_equal_derivation"], checks["iron_differs_from_parent"],
        checks["reconstructed_q05_matches_incumbent"] <= tolerance,
        checks["derived_mean3_matches_recorded_release_ensemble"] <= tolerance,
    ])
    report = {"candidate": CANDIDATE, "audit": "independent_process_readback",
              "checks": checks, "passed": passed, "tolerance": tolerance}
    write_json(output / "independent-audit.json", report)
    write_json(output / "terminal-reconciliation.json", {
        "candidate": CANDIDATE, "audit_passed": passed, "exceptions": [] if passed else ["audit_failed"],
        "new_fits": 0, "new_optimizers": 0, "model_loads": 0, "packages": 1,
        "desktop_writes": 0, "agent_uploads": 0})
    return report
