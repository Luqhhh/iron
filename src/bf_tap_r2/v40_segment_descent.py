#!/usr/bin/env python3
"""V40: the A60->V7m ridge and the raised bounded ceiling.

The third interior report (`0.75*A60 + 0.25*V7m`, coordinates
``(0.30, 0.45, 0.25)``) scored ``96.3727``, exactly tying ``0.5*A60 + 0.5*V7m``.
Those two points lie on the same ``A60 -> V7m`` segment (``lambda = 0.75`` and
``0.50``), so the tie is a plateau on a measured chord rather than a surprise.
Folding it in raises the bounded-region ceiling and **breaks the earlier
"96.4 is unreachable" verdict**: the bounded time maximum is now ``96.3882`` and
the iron head-room adds ``+0.0160``, giving ``96.4042``.

This module reports the updated bounds and composes the next zero-fit probes.
Never trains, never uploads, refuses to write outside ``local/runs/round2-v40``.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .submission import ZIP_NAME, package
from .v18_compose import (
    build_design,
    read_sources,
    recover_endpoints,
    sha256_bytes,
    sha256_file,
    verify_payload,
)
from .v32_family_ceiling import secant_bound

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v40/SPEC.yaml")
PROBES_DEFAULT = Path("configs/round2_v40/PROBES.yaml")
SOURCES_SPEC = Path("configs/round2_v18/SPEC.yaml")
PROBE_OUTPUT = Path("local/runs/round2-v40/probes-r1")
REPORT_OUTPUT = Path("local/runs/round2-v40/ridge-r1.json")
IRON_HEAD_ROOM = 0.0160


def _segment(lam: float) -> tuple[float, float, float]:
    """``lambda*A60 + (1-lambda)*V7m`` in ``(V36, N, V7m)`` coordinates."""
    return (round(0.4 * lam, 6), round(0.6 * lam, 6), round(1.0 - lam, 6))


#: Recorded platform scores lifted to the V12-iron reference, time coordinates
#: ``(V36, N, V7m)``.
MEASURED: dict[str, tuple[tuple[float, float, float], float, str]] = {
    "V36": ((1.00, 0.00, 0.00), 96.2734 + 0.0160, "V36 package"),
    "A35": ((0.65, 0.35, 0.00), 96.3366 + 0.0160, "A35 package"),
    "A45": ((0.55, 0.45, 0.00), 96.3438 + 0.0160, "A45 package"),
    "A60": ((0.40, 0.60, 0.00), 96.3465 + 0.0160, "A60 package"),
    "A72": ((0.28, 0.72, 0.00), 96.3425 + 0.0160, "A72 package"),
    "B0": ((0.325, 0.175, 0.50), 96.3679, "V18_B0"),
    "I2": (_segment(0.25), 96.3629, "V32_TIME_A60V7_75"),
    "I1": (_segment(0.50), 96.3727, "V32_TIME_A60V7_50"),
    "I5": (_segment(0.75), 96.3727, "V32_TIME_A60V7_25"),
}


def analysis(step: float = 0.05) -> dict[str, Any]:
    table = {name: (coords, value) for name, (coords, value, _) in MEASURED.items()}
    grid = [round(index * step, 6) for index in range(int(round(1.0 / step)) + 1)]
    rows: list[tuple[float, tuple[float, float, float], tuple[str, str, float]]] = []
    total = 0
    for a in grid:
        for b in grid:
            c = round(1.0 - a - b, 10)
            if c < -1e-9:
                continue
            total += 1
            bound = secant_bound((a, b, c), table)
            if bound is not None:
                rows.append((bound[0], (a, b, c), bound[1]))
    rows.sort(key=lambda row: -row[0])
    best = max(MEASURED.items(), key=lambda item: item[1][1])
    ceiling = rows[0][0] + IRON_HEAD_ROOM
    return {
        "step": step,
        "measured": {name: {"coordinates": list(coords), "score_with_V12_iron": round(value, 4),
                            "source": label}
                     for name, (coords, value, label) in sorted(MEASURED.items())},
        "measured_best": {"point": best[0], "score": round(best[1][1], 4), "source": best[1][2]},
        "ridge": {
            "segment": "lambda*A60 + (1-lambda)*V7m",
            "observed": {"lambda=0.25": 96.3629, "lambda=0.50": 96.3727, "lambda=0.75": 96.3727},
            "plateau": "the two equal points are the endpoints of a measured flat chord; by concavity every interior point of that chord is at least 96.3727",
            "lambda_0625_bracket": {"lower": 96.3727, "upper": 96.3776,
                                    "note": "bounded above by the I2->I1 chord extension and below by the flat I1->I5 chord"},
        },
        "grid_points": total,
        "bounded_grid_points": len(rows),
        "unbounded_grid_points": total - len(rows),
        "simplex_max_bound_with_V12_iron": round(rows[0][0], 6),
        "simplex_argmax_coordinates": list(rows[0][1]),
        "simplex_argmax_bound_source": {"from": rows[0][2][0], "to": rows[0][2][1],
                                        "lambda": round(rows[0][2][2], 4)},
        "iron_head_room": IRON_HEAD_ROOM,
        "bounded_ceiling_with_iron_endpoint": round(ceiling, 6),
        "ceiling_vs_target": round(ceiling - 96.4, 6),
        "target_provably_unreachable": bool(ceiling < 96.4),
        "runner_up_bounds": [{"bound": round(row[0], 4), "coordinates": list(row[1]),
                              "source": {"from": row[2][0], "to": row[2][1]}}
                             for row in rows[1:8]],
        "new_fits": 0,
        "agent_uploads": 0,
    }


def _context(root: Path, spec: Mapping[str, Any]):
    sources = yaml.safe_load((root / spec["sources_spec"]).read_text())
    tables = read_sources(root, sources)
    endpoints, recovery = recover_endpoints(tables, sources)
    return tables, endpoints, recovery


def build_probes(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = PROBES_DEFAULT,
                 output: Path | str = PROBE_OUTPUT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec_path = (root / spec_path).resolve()
    if not spec_path.is_relative_to(root / "configs/round2_v40"):
        raise ValueError("V40 probe specification required")
    spec = yaml.safe_load(spec_path.read_text())
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v40"):
        raise ValueError("V40 probes must stay under local/runs/round2-v40")
    if out.exists():
        raise FileExistsError("V40 probe output already exists; refusing overwrite")
    tables, endpoints, recovery = _context(root, spec)
    out.mkdir(parents=True)
    tolerance = float(spec["verification"]["blend_relative_tolerance"])
    report: dict[str, Any] = {"recovery": recovery, "packages": {}, "new_fits": 0,
                              "desktop_writes": 0, "agent_uploads": 0}
    for name, design in spec["designs"].items():
        payload, _ = build_design(design, endpoints, tables)
        verification = verify_payload(payload, design, endpoints, tables, tolerance)
        package_dir = out / name
        package_dir.mkdir(exist_ok=False)
        package(package_dir, payload, list(tables["a35"].ids))
        import zipfile

        with zipfile.ZipFile(package_dir / ZIP_NAME) as archive:
            if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload:
                raise ValueError(f"{name}: ZIP read-back failed")
        record = {"design": json.loads(json.dumps(design)), "readback": verification,
                  "zip_sha256": sha256_file(package_dir / ZIP_NAME),
                  "result_csv_sha256": sha256_bytes((package_dir / "result.csv").read_bytes())}
        (package_dir / "manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        report["packages"][name] = record
        print(json.dumps({"built": name, "zip_sha256": record["zip_sha256"]}))
    (out / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return report


def audit_probes(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = PROBES_DEFAULT,
                 output: Path | str = PROBE_OUTPUT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    out = (root / output).resolve()
    if not out.is_relative_to(root / "local/runs/round2-v40"):
        raise ValueError("V40 audit only covers private local output")
    tables, endpoints, recovery = _context(root, spec)
    template = (root / "复赛_test/result_template.csv").read_text(encoding="utf-8-sig")
    template_ids = [row["sample_id"] for row in csv.DictReader(io.StringIO(template, newline=""))]
    if template_ids != list(tables["a35"].ids):
        raise ValueError("Package order differs from the official result template")
    tolerance = float(spec["verification"]["blend_relative_tolerance"])
    report: dict[str, Any] = {"template_order": True, "recovery": recovery, "packages": {}}
    for name, design in spec["designs"].items():
        package_dir = out / name
        payload = (package_dir / "result.csv").read_bytes()
        import zipfile

        with zipfile.ZipFile(package_dir / ZIP_NAME) as archive:
            if archive.namelist() != ["result.csv"] or archive.read("result.csv") != payload:
                raise ValueError(f"{name}: archive mismatch")
        report["packages"][name] = {
            "readback": verify_payload(payload, design, endpoints, tables, tolerance),
            "zip_sha256": sha256_file(package_dir / ZIP_NAME)}
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "audited", "packages": len(report["packages"])}))
    return report


def run(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
        output: Path | str = REPORT_OUTPUT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    destination = (root / output).resolve()
    if not destination.is_relative_to(root / "local/runs/round2-v40"):
        raise ValueError("V40 evidence must stay private under local/runs/round2-v40")
    if destination.exists():
        raise FileExistsError("V40 output already exists; refusing overwrite")
    report = analysis(float(spec.get("grid_step", 0.05)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("measured_best", "simplex_max_bound_with_V12_iron",
                                             "simplex_argmax_coordinates",
                                             "bounded_ceiling_with_iron_endpoint",
                                             "ceiling_vs_target", "target_provably_unreachable",
                                             "bounded_grid_points", "unbounded_grid_points")}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["analyze", "probes", "audit"], default="analyze")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    parser.add_argument("--probes", type=Path, default=PROBES_DEFAULT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "probes":
        build_probes(args.root, args.probes, args.output or PROBE_OUTPUT)
    elif args.mode == "audit":
        audit_probes(args.root, args.probes, args.output or PROBE_OUTPUT)
    else:
        run(args.root, args.spec, args.output or REPORT_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
