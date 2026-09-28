#!/usr/bin/env python3
"""V18 zero-fit composition of the delivered A35/A60/V7/V12 prediction columns.

The module audits seven hash-pinned original ZIPs, recovers the frozen member
endpoints by exact field arithmetic, and writes a small pre-declared set of
convex-combination packages.  It never trains, never reads labels, never
overwrites an existing destination and never uploads anything.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .data import SUBMISSION_COLUMNS
from .submission import ZIP_NAME, package

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v18/SPEC.yaml")
OUTPUT_DEFAULT = Path("local/runs/round2-v18/packages-r1")

MAX_ZIP_BYTES = 16 * 1024 * 1024
MAX_CSV_BYTES = 8 * 1024 * 1024
ZIP_MEMBER = "result.csv"

IRON_COLUMN = "pred_tap_iron"
TIME_COLUMN = "pred_tap_time_len"
END_NAMES = ("v36_iron", "v12_member_iron", "v36_time", "n_time", "v7_member_time")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


@dataclass(frozen=True)
class SourceTable:
    """One original package kept as floats plus the exact CSV field text."""

    name: str
    path: str
    zip_sha256: str
    ids: tuple[str, ...]
    iron: np.ndarray
    iron_text: tuple[str, ...]
    time: np.ndarray
    time_text: tuple[str, ...]
    other_text: dict[str, tuple[str, ...]]


def read_zip_result(path: Path, expected_sha256: str, name: str) -> SourceTable:
    path = Path(path)
    payload = path.read_bytes()
    if len(payload) > MAX_ZIP_BYTES:
        raise ValueError(f"{name}: ZIP exceeds the size limit")
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"{name}: original ZIP SHA-256 mismatch; do not substitute a regenerated ZIP")
    import zipfile

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != [ZIP_MEMBER] or archive.testzip() is not None:
            raise ValueError(f"{name}: expected exactly {ZIP_MEMBER}")
        if archive.getinfo(ZIP_MEMBER).file_size > MAX_CSV_BYTES:
            raise ValueError(f"{name}: CSV exceeds the size limit")
        text = archive.read(ZIP_MEMBER).decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader)
    if list(header) != list(SUBMISSION_COLUMNS):
        raise ValueError(f"{name}: submission columns changed: {header}")
    index = {column: header.index(column) for column in SUBMISSION_COLUMNS}
    ids: list[str] = []
    iron: list[float] = []
    iron_text: list[str] = []
    time: list[float] = []
    time_text: list[str] = []
    other: dict[str, list[str]] = {column: [] for column in SUBMISSION_COLUMNS}
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"{name}: row width mismatch")
        ids.append(row[index["sample_id"]])
        iron_text.append(row[index[IRON_COLUMN]])
        time_text.append(row[index[TIME_COLUMN]])
        iron.append(float(row[index[IRON_COLUMN]]))
        time.append(float(row[index[TIME_COLUMN]]))
        for column in SUBMISSION_COLUMNS:
            other[column].append(row[index[column]])
    iron_array = np.asarray(iron, dtype=float)
    time_array = np.asarray(time, dtype=float)
    if not np.isfinite(iron_array).all() or not np.isfinite(time_array).all():
        raise ValueError(f"{name}: non-finite prediction")
    if (iron_array < 0).any() or (time_array < 0).any():
        raise ValueError(f"{name}: negative prediction")
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{name}: blank or duplicate sample_id")
    return SourceTable(
        name=name,
        path=str(path),
        zip_sha256=expected_sha256,
        ids=tuple(ids),
        iron=iron_array,
        iron_text=tuple(iron_text),
        time=time_array,
        time_text=tuple(time_text),
        other_text={column: tuple(values) for column, values in other.items()},
    )


def read_sources(root: Path, spec: Mapping[str, Any]) -> dict[str, SourceTable]:
    tables: dict[str, SourceTable] = {}
    for name, record in spec["sources"].items():
        tables[name] = read_zip_result(root / record["path"], record["sha256"], name)
    reference = tables["a35"]
    for name, table in tables.items():
        if table.ids != reference.ids:
            raise ValueError(f"{name}: sample_id order differs from a35")
        if len(table.ids) != len(reference.ids):
            raise ValueError(f"{name}: row count differs from a35")
        for column in SUBMISSION_COLUMNS:
            if column in (IRON_COLUMN, TIME_COLUMN):
                continue
            if table.other_text[column] != reference.other_text[column]:
                raise ValueError(f"{name}: metadata column {column!r} differs from a35")
    return tables


def n_line_columns(tables: Mapping[str, SourceTable], spec: Mapping[str, Any]
                   ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Solve the two-point affine N line and re-derive every cross-check package."""
    a35 = tables["a35"]
    a60 = tables["a60"]
    alpha_low = float(spec["sources"]["a35"]["n_alpha"])
    alpha_high = float(spec["sources"]["a60"]["n_alpha"])
    delta = alpha_high - alpha_low
    if delta <= 0:
        raise ValueError("N-line alphas must increase")
    v36 = (alpha_high * a35.time - alpha_low * a60.time) / delta
    member = ((1.0 - alpha_low) * a60.time - (1.0 - alpha_high) * a35.time) / delta
    residuals: dict[str, float] = {}
    for name, record in spec["sources"].items():
        if record.get("role") != "n_line_crosscheck":
            continue
        alpha = float(record["n_alpha"])
        predicted = (1.0 - alpha) * v36 + alpha * member
        residuals[name] = float(np.max(np.abs(predicted - tables[name].time)))
    return {"v36_time": v36, "n_time": member}, residuals


def recover_endpoints(tables: Mapping[str, SourceTable], spec: Mapping[str, Any]
                      ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    line, residuals = n_line_columns(tables, spec)
    tolerance = float(spec["composition"]["crosscheck_tolerance"])
    worst = max(residuals.values()) if residuals else 0.0
    if worst > tolerance:
        raise ValueError(f"N-line cross-check failed: worst absolute difference {worst:.3e}")
    a35 = tables["a35"]
    v12 = tables["v12"]
    v7 = tables["v7"]
    if v12.time_text != a35.time_text:
        raise ValueError("V12 is not an isolated iron change: time strings differ from A35")
    if v7.iron_text != a35.iron_text:
        raise ValueError("V7 is not an isolated time change: iron strings differ from A35")
    endpoints = {
        "v36_iron": a35.iron.copy(),
        "v12_member_iron": 2.0 * v12.iron - a35.iron,
        "v7_member_time": 2.0 * v7.time - a35.time,
        **line,
    }
    report = {
        "n_line_crosscheck": {name: round(value, 15) for name, value in sorted(residuals.items())},
        "n_line_worst_absolute_difference": worst,
        "v12_time_strings_equal_a35": True,
        "v7_iron_strings_equal_a35": True,
        "recovered_minimum": {name: float(value.min()) for name, value in sorted(endpoints.items())},
        "recovered_negative_rows": {name: int((value < 0).sum()) for name, value in sorted(endpoints.items())},
    }
    return endpoints, report


def _convex(weights: Mapping[str, float], endpoints: Mapping[str, np.ndarray],
            length: int) -> tuple[np.ndarray, dict[str, float]]:
    total = float(sum(weights.values()))
    unknown = set(weights) - set(END_NAMES)
    if unknown:
        raise ValueError(f"Unknown endpoint names: {sorted(unknown)}")
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"Convex weights must sum to 1, got {total!r}")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Convex weights must be non-negative")
    combined = np.zeros(length, dtype=float)
    for name, weight in weights.items():
        column = np.asarray(endpoints[name], dtype=float)
        if column.shape != (length,):
            raise ValueError(f"Endpoint {name} has shape {column.shape}, expected {(length,)}")
        combined += float(weight) * column
    negative = int((combined < 0).sum())
    if negative:
        raise ValueError(f"Composed column has {negative} negative rows; pre-declared floor is zero")
    return combined, {"weights": {k: float(v) for k, v in weights.items()}, "negative_rows": negative}


def _resolve(column: Mapping[str, Any], endpoints: Mapping[str, np.ndarray],
             tables: Mapping[str, SourceTable], length: int, kind: str
             ) -> tuple[np.ndarray, tuple[str, ...] | None, dict[str, Any]]:
    selector = column.get("kind")
    if selector == "copy":
        source = str(column["source"])
        table = tables[source]
        values = table.iron if kind == "iron" else table.time
        text = table.iron_text if kind == "iron" else table.time_text
        return values.copy(), tuple(text), {"kind": "copy", "source": source}
    if selector == "convex":
        values, meta = _convex({str(k): float(v) for k, v in column["weights"].items()},
                               endpoints, length)
        return values, None, {"kind": "convex", **meta}
    raise ValueError(f"Unknown column kind: {selector!r}")


def _check_payload_shape(payload: bytes, ids: Sequence[str]) -> None:
    reader = csv.reader(io.StringIO(payload.decode("utf-8"), newline=""))
    header = next(reader)
    if list(header) != list(SUBMISSION_COLUMNS):
        raise ValueError("Generated payload column order mismatch")
    observed = [row[0] for row in reader if row]
    if observed != list(ids) or len(set(observed)) != len(observed):
        raise ValueError("Generated payload sample_id order mismatch")


def build_design(design: Mapping[str, Any], endpoints: Mapping[str, np.ndarray],
                 tables: Mapping[str, SourceTable]) -> tuple[bytes, dict[str, Any]]:
    ids = list(tables["a35"].ids)
    length = len(ids)
    iron, iron_text, iron_meta = _resolve(design["iron"], endpoints, tables, length, "iron")
    time, time_text, time_meta = _resolve(design["time"], endpoints, tables, length, "time")
    if (iron < 0).any() or (time < 0).any():
        raise ValueError("Composed predictions must be non-negative")
    if not np.isfinite(iron).all() or not np.isfinite(time).all():
        raise ValueError("Composed predictions must be finite")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(list(SUBMISSION_COLUMNS))
    for position, sid in enumerate(ids):
        iron_field = iron_text[position] if iron_text is not None else format(float(iron[position]), ".17g")
        time_field = time_text[position] if time_text is not None else format(float(time[position]), ".17g")
        writer.writerow([sid, iron_field, time_field])
    payload = stream.getvalue().encode("utf-8")
    _check_payload_shape(payload, ids)
    meta = {
        "id": None,
        "iron": iron_meta,
        "time": time_meta,
        "rows": length,
        "iron_unchanged_by_copy": iron_text is not None,
        "time_unchanged_by_copy": time_text is not None,
    }
    return payload, meta


def verify_payload(payload: bytes, design: Mapping[str, Any], endpoints: Mapping[str, np.ndarray],
                   tables: Mapping[str, SourceTable], tolerance: float) -> dict[str, Any]:
    ids = list(tables["a35"].ids)
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"), newline="")))
    if len(rows) != len(ids) or [row["sample_id"] for row in rows] != ids:
        raise ValueError("Read-back row count or order differs from template")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Read-back contains duplicate sample_id")
    iron = np.asarray([float(row[IRON_COLUMN]) for row in rows], dtype=float)
    time = np.asarray([float(row[TIME_COLUMN]) for row in rows], dtype=float)
    if (iron < 0).any() or (time < 0).any() or not np.isfinite(iron).all() or not np.isfinite(time).all():
        raise ValueError("Read-back predictions are not finite and non-negative")
    report: dict[str, Any] = {"rows": len(rows), "unique_ids": len({row["sample_id"] for row in rows})}
    for kind, values, column in (("iron", iron, design["iron"]), ("time", time, design["time"])):
        if column.get("kind") == "copy":
            source = tables[str(column["source"])]
            expected_text = source.iron_text if kind == "iron" else source.time_text
            observed_text = tuple(row[IRON_COLUMN if kind == "iron" else TIME_COLUMN] for row in rows)
            mismatches = sum(left != right for left, right in zip(observed_text, expected_text))
            if mismatches:
                raise ValueError(f"{kind}: copied column is not byte-identical to its source ({mismatches})")
            report[f"{kind}_copy_source"] = str(column["source"])
            report[f"{kind}_copy_byte_mismatches"] = 0
        else:
            expected, _ = _convex({str(k): float(v) for k, v in column["weights"].items()},
                                  endpoints, len(ids))
            scale = max(1.0, float(np.max(np.abs(expected))))
            difference = float(np.max(np.abs(values - expected))) / scale
            if difference > tolerance:
                raise ValueError(f"{kind}: blend read-back failed at relative {difference:.3e}")
            report[f"{kind}_blend_relative_difference"] = difference
    return report


def run(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
        output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec_path = (root / spec_path).resolve()
    if not spec_path.is_relative_to(root / "configs/round2_v18"):
        raise ValueError("V18 specification required")
    spec = yaml.safe_load(spec_path.read_text())
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v18"):
        raise ValueError("V18 packages are private and must stay under local/")
    if out.exists():
        raise FileExistsError("V18 output already exists; refusing overwrite")
    tables = read_sources(root, spec)
    endpoints, recovery = recover_endpoints(tables, spec)
    out.mkdir(parents=True)
    report: dict[str, Any] = {
        "spec_sha256": sha256_file(spec_path),
        "sources": {name: {"path": table.path, "zip_sha256": table.zip_sha256, "rows": len(table.ids)}
                    for name, table in sorted(tables.items())},
        "recovery": recovery,
        "packages": {},
        "new_fits": 0,
        "agent_uploads": 0,
        "platform_observation": False,
    }
    tolerance = float(spec["verification"]["blend_relative_tolerance"])
    expected_rows = int(spec["verification"]["expected_rows"])
    if len(tables["a35"].ids) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(tables['a35'].ids)}")
    for name, design in spec["designs"].items():
        payload, meta = build_design(design, endpoints, tables)
        verification = verify_payload(payload, design, endpoints, tables, tolerance)
        package_dir = out / name
        package_dir.mkdir(exist_ok=False)
        package(package_dir, payload, list(tables["a35"].ids))
        import zipfile

        with zipfile.ZipFile(package_dir / ZIP_NAME) as archive:
            if archive.namelist() != [ZIP_MEMBER] or archive.read(ZIP_MEMBER) != payload:
                raise ValueError(f"{name}: ZIP read-back failed")
        manifest = {
            "id": name,
            "design": json.loads(json.dumps(design)),
            **{k: v for k, v in meta.items() if k != "id"},
            "verification": verification,
            "result_csv_sha256": sha256_bytes((package_dir / "result.csv").read_bytes()),
            "zip_sha256": sha256_file(package_dir / ZIP_NAME),
        }
        (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        report["packages"][name] = {
            "zip": str((package_dir / ZIP_NAME).relative_to(root)),
            "result_csv": str((package_dir / "result.csv").relative_to(root)),
            "manifest": str((package_dir / "manifest.json").relative_to(root)),
            "zip_sha256": manifest["zip_sha256"],
            "result_csv_sha256": manifest["result_csv_sha256"],
            "verification": verification,
        }
        print(json.dumps({"built": name, "zip_sha256": manifest["zip_sha256"],
                          "copy_columns": [k for k in ("iron_unchanged_by_copy", "time_unchanged_by_copy")
                                           if meta[k]]}), flush=True)
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "built", "packages": len(report["packages"]),
                      "report": str(report_path.relative_to(root)), "uploads": 0}, indent=2))
    return report


def audit(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
          output: Path | str = OUTPUT_DEFAULT, template: Path | str = "复赛_test/result_template.csv"
          ) -> dict[str, Any]:
    """Read-only post-write audit of an existing V18 output directory."""
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v18"):
        raise ValueError("V18 audit only covers private local output")
    tables = read_sources(root, spec)
    endpoints, recovery = recover_endpoints(tables, spec)
    tolerance = float(spec["verification"]["blend_relative_tolerance"])
    template_text = (root / template).read_text(encoding="utf-8-sig")
    template_ids = [row["sample_id"] for row in csv.DictReader(io.StringIO(template_text, newline=""))]
    if template_ids != list(tables["a35"].ids):
        raise ValueError("Package order differs from the official result template")
    report: dict[str, Any] = {"template_order": True, "packages": {}}
    for name, design in spec["designs"].items():
        package_dir = out / name
        result_bytes = (package_dir / "result.csv").read_bytes()
        import zipfile

        with zipfile.ZipFile(package_dir / ZIP_NAME) as archive:
            if archive.namelist() != [ZIP_MEMBER]:
                raise ValueError(f"{name}: archive member list changed")
            archived = archive.read(ZIP_MEMBER)
        if archived != result_bytes:
            raise ValueError(f"{name}: ZIP content differs from result.csv")
        verification = verify_payload(result_bytes, design, endpoints, tables, tolerance)
        report["packages"][name] = {
            "result_csv_sha256": sha256_bytes(result_bytes),
            "zip_sha256": sha256_file(package_dir / ZIP_NAME),
            "verification": verification,
        }
    report["recovery"] = recovery
    report["new_fits"] = 0
    report["agent_uploads"] = 0
    audit_path = out / "audit.json"
    if audit_path.exists():
        raise FileExistsError("V18 audit already exists; refusing overwrite")
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "audited", "packages": len(report["packages"]),
                      "audit": str(audit_path.relative_to(root))}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["build", "audit"], default="build")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    if args.mode == "audit":
        audit(args.root, args.spec, args.output)
    else:
        run(args.root, args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
