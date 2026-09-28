#!/usr/bin/env python3
"""V21 zero-fit time mixes of the accurate experts on the verified B0 parent.

Recovers the V36/N/V7m time endpoints field-exactly from the delivered A35/A60/V7
ZIPs, adds the V20 full-data P-LL member, checks the frozen weights on the four
split seeds and writes three single-target replacement packages.  Never trains,
never uploads and refuses to write outside ``local/runs/round2-v21``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .data import SUBMISSION_COLUMNS
from .submission import ZIP_NAME, package
from .v18_compose import (
    IRON_COLUMN,
    TIME_COLUMN,
    SourceTable,
    _check_payload_shape,
    read_sources,
    read_zip_result,
    recover_endpoints,
    sha256_bytes,
    sha256_file,
)
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import package_score, paired_summary, wmape
from .v5_spec import load_v5_spec
from .v17_confirm import confirmed_references

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v21/SPEC.yaml")
V18_SPEC = Path("configs/round2_v18/SPEC.yaml")
OUTPUT_DEFAULT = Path("local/runs/round2-v21/packages-r1")
IRON = "tap_iron"
TIME = "tap_time_len"


def load_member(path: Path, expected_sha256: str | None, rows: int) -> np.ndarray:
    payload = Path(path).read_bytes()
    if expected_sha256 and sha256_bytes(payload) != expected_sha256:
        raise ValueError("P-LL member file hash mismatch")
    values = np.load(io.BytesIO(payload), allow_pickle=False).astype(float)
    if values.shape != (rows,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid P-LL member predictions")
    return values


def endpoint_table(root: Path, spec: Mapping[str, Any], spec18: Mapping[str, Any]
                   ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tables = read_sources(root, spec18)
    endpoints, recovery = recover_endpoints(tables, spec18)
    rows = len(tables["a35"].ids)
    member_path = root / spec["pll_member"]["path"]
    expected = spec["pll_member"].get("sha256")
    if expected in (None, "pending_first_run_pin"):
        raise ValueError("The P-LL member hash must be pinned in the V21 spec before building")
    pll = load_member(member_path, expected, rows)
    endpoints["a35_time"] = tables["a35"].time.copy()
    endpoints["pll_time"] = pll
    recovery["pll_member"] = {"path": str(member_path), "sha256": sha256_file(member_path),
                              "shape": list(pll.shape), "minimum": float(pll.min())}
    return endpoints, recovery


def four_seed_check(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen-weight gains against B0 on the four split seeds (recorded caches)."""
    spec17 = yaml.safe_load((root / "configs/round2_v17/SPEC.yaml").read_text())
    dev_seeds = [int(seed) for seed in spec17["split_seeds"]]
    conf_seeds = [int(seed) for seed in spec17["confirmation_seeds"]]
    seeds = dev_seeds + conf_seeds
    frame = load_v5_training_frame(root)
    folds = {seed: fold_vector(root, frame, seed, load_v5_spec(root)) for seed in seeds}
    a35, b0, _ = confirmed_references(root, frame, folds, spec17)
    rows = len(frame)
    endpoints: dict[str, dict[int, np.ndarray]] = {name: {} for name in ("a35_time", "v36_time", "n_time",
                                                                        "v7_member_time", "pll_time")}
    for seed in seeds:
        v7 = np.full(rows, np.nan)
        pll = np.full(rows, np.nan)
        n_member = np.full(rows, np.nan)
        for fold in range(5):
            mask = folds[seed] == fold
            if seed in dev_seeds:
                v7[mask] = np.load(root / "local/runs/round2-v7-periodic-networks/development-r1" /
                                   f"tap_time_len-tabm_plr001-s{seed}-f{fold}.npy", allow_pickle=False).ravel()
                pll[mask] = np.load(root / "local/runs/round2-v17/development-r1" /
                                    f"P_LL_T-s{seed}-f{fold}.npy", allow_pickle=False).ravel()
                values = np.load(root / "local/runs/round2-v5-error-covariance/time-n-family-r1" /
                                 f"seed-{seed}/pred-v36-s1-N-0048.npy", allow_pickle=False)
                n_member[mask] = values[mask]
            else:
                v7[mask] = np.load(root / "local/runs/round2-v7-periodic-networks/confirmation-r1" /
                                   f"seed-{seed}-fold-{fold}.npy", allow_pickle=False).ravel()
                pll[mask] = np.load(root / "local/runs/round2-v17/confirmation-r1" /
                                    f"P_LL_T-s{seed}-f{fold}.npy", allow_pickle=False).ravel()
                cached = np.load(root / "local/runs/round2-v5-error-covariance/replication-r1" /
                                 f"seed-{seed}/fold-{fold}.npz")
                n_member[mask] = cached["candidate"][mask]
        for name, values in (("v7_member_time", v7), ("pll_time", pll), ("n_time", n_member)):
            if not np.isfinite(values).all():
                raise ValueError(f"Incomplete {name} reference for seed {seed}")
            endpoints[name][seed] = values
        endpoints["a35_time"][seed] = a35[seed][TIME]
        endpoints["v36_time"][seed] = (a35[seed][TIME] - 0.35 * n_member) / 0.65
    y_iron = frame[IRON].to_numpy()
    y_time = frame[TIME].to_numpy()
    base = {seed: package_score(wmape(y_iron, b0[seed][IRON]), wmape(y_time, b0[seed][TIME]))
            for seed in seeds}
    results: dict[str, Any] = {}
    for name, design in spec["designs"].items():
        gains = []
        for seed in seeds:
            column = np.zeros(rows)
            for endpoint, weight in design["time"].items():
                column = column + float(weight) * endpoints[endpoint][seed]
            candidate = package_score(wmape(y_iron, b0[seed][IRON]), wmape(y_time, column))
            gains.append(candidate - base[seed])
        summary = paired_summary(gains)
        dev_mean = float(np.mean([base[seed] + gains[index] for index, seed in enumerate(dev_seeds)]))
        results[name] = {
            "gains": {str(seed): round(gain, 6) for seed, gain in zip(seeds, gains)},
            "mean": round(float(summary["mean"]), 6),
            "paired_lcb95": round(float(summary["lcb95"]), 6),
            "positive_seeds": int(summary["positive"]),
            "development_mean_candidate_score": round(dev_mean, 6),
            "local_working_gate_met": bool(dev_mean >= 96.25),
            "promoted": bool(summary["positive"] == 4 and summary["lcb95"] > 0 and dev_mean >= 96.25),
            "b0_scores": {str(seed): round(base[seed], 6) for seed in seeds},
        }
    return results


def _build_payload(ids: Sequence[str], iron_text: Sequence[str], time_values: np.ndarray) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(list(SUBMISSION_COLUMNS))
    for position, sid in enumerate(ids):
        writer.writerow([sid, iron_text[position], format(float(time_values[position]), ".17g")])
    payload = stream.getvalue().encode("utf-8")
    _check_payload_shape(payload, ids)
    return payload


def _verify(payload: bytes, parent: SourceTable, design: Mapping[str, Any],
            endpoints: Mapping[str, np.ndarray], tolerance: float) -> dict[str, Any]:
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"), newline="")))
    if len(rows) != len(parent.ids) or [row["sample_id"] for row in rows] != list(parent.ids):
        raise ValueError("Read-back row count or order differs from the parent")
    observed_iron = tuple(row[IRON_COLUMN] for row in rows)
    mismatches = sum(left != right for left, right in zip(observed_iron, parent.iron_text))
    if mismatches:
        raise ValueError(f"Copied iron column is not byte-identical to the parent ({mismatches})")
    time = np.asarray([float(row[TIME_COLUMN]) for row in rows])
    if not np.isfinite(time).all() or (time < 0).any():
        raise ValueError("Read-back time predictions must be finite and non-negative")
    expected = np.zeros(len(rows))
    for endpoint, weight in design["time"].items():
        expected = expected + float(weight) * endpoints[endpoint]
    scale = max(1.0, float(np.max(np.abs(expected))))
    difference = float(np.max(np.abs(time - expected))) / scale
    if difference > tolerance:
        raise ValueError(f"Time blend read-back failed at relative {difference:.3e}")
    return {"rows": len(rows), "unique_ids": len({row["sample_id"] for row in rows}),
            "template_order": True, "iron_string_mismatches": 0,
            "time_blend_relative_difference": difference,
            "time_minimum": float(time.min())}


def run(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
        output: Path | str = OUTPUT_DEFAULT, check_only: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    spec_path = (root / spec_path).resolve()
    if not spec_path.is_relative_to(root / "configs/round2_v21"):
        raise ValueError("V21 specification required")
    spec = yaml.safe_load(spec_path.read_text())
    spec18 = yaml.safe_load((root / V18_SPEC).read_text())
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v21"):
        raise ValueError("V21 packages are private and must stay under local/runs/round2-v21")
    check = four_seed_check(root, spec)
    if check_only:
        print(json.dumps(check, indent=2))
        return check
    if not all(row["promoted"] for row in check.values()):
        raise ValueError("A frozen V21 design failed the four-seed gate")
    if out.exists():
        raise FileExistsError("V21 output already exists; refusing overwrite")
    endpoints, recovery = endpoint_table(root, spec, spec18)
    parent = read_zip_result(root / spec["parent"]["zip"], spec["parent"]["sha256"], "parent")
    out.mkdir(parents=True)
    verification = {"V18_source_recovery": recovery, "designs": check, "packages": {},
                    "new_fits": 0, "desktop_writes": 0, "agent_uploads": 0}
    tolerance = float(spec["verification"]["blend_relative_tolerance"])
    for name, design in spec["designs"].items():
        time_values = np.zeros(len(parent.ids))
        for endpoint, weight in design["time"].items():
            time_values = time_values + float(weight) * endpoints[endpoint]
        if (time_values < 0).any() or not np.isfinite(time_values).all():
            raise ValueError(f"{name}: invalid composed time column")
        payload = _build_payload(parent.ids, parent.iron_text, time_values)
        readback = _verify(payload, parent, design, endpoints, tolerance)
        package_dir = out / name
        package_dir.mkdir(exist_ok=False)
        package(package_dir, payload, list(parent.ids))
        import zipfile

        with zipfile.ZipFile(package_dir / ZIP_NAME) as archive:
            if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload:
                raise ValueError(f"{name}: ZIP read-back failed")
        record = {"design": json.loads(json.dumps(design)), "readback": readback,
                  "zip_sha256": sha256_file(package_dir / ZIP_NAME),
                  "result_csv_sha256": sha256_bytes((package_dir / "result.csv").read_bytes()),
                  "four_seed": check[name]}
        (package_dir / "manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        verification["packages"][name] = record
        print(json.dumps({"built": name, "zip_sha256": record["zip_sha256"],
                          "mean_gain": check[name]["mean"], "lcb95": check[name]["paired_lcb95"]}))
    (out / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return verification


def audit(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
          output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    spec18 = yaml.safe_load((root / V18_SPEC).read_text())
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v21"):
        raise ValueError("V21 audit only covers private local output")
    endpoints, recovery = endpoint_table(root, spec, spec18)
    parent = read_zip_result(root / spec["parent"]["zip"], spec["parent"]["sha256"], "parent")
    template = (root / "复赛_test/result_template.csv").read_text(encoding="utf-8-sig")
    template_ids = [row["sample_id"] for row in csv.DictReader(io.StringIO(template, newline=""))]
    if template_ids != list(parent.ids):
        raise ValueError("Parent order differs from the official result template")
    report: dict[str, Any] = {"template_order": True, "recovery": recovery, "packages": {}}
    for name, design in spec["designs"].items():
        package_dir = out / name
        payload = (package_dir / "result.csv").read_bytes()
        import zipfile

        with zipfile.ZipFile(package_dir / ZIP_NAME) as archive:
            if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload:
                raise ValueError(f"{name}: archive mismatch")
        report["packages"][name] = {
            "readback": _verify(payload, parent, design, endpoints,
                                float(spec["verification"]["blend_relative_tolerance"])),
            "zip_sha256": sha256_file(package_dir / ZIP_NAME),
        }
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "audited", "packages": len(report["packages"])}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["build", "audit", "check"], default="build")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    if args.mode == "audit":
        audit(args.root, args.spec, args.output)
    elif args.mode == "check":
        run(args.root, args.spec, args.output, check_only=True)
    else:
        run(args.root, args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
