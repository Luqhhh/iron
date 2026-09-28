#!/usr/bin/env python3
"""V39: the interior descent after the first two interior probes.

Two interior time-simplex packages were scored: ``0.5*A60 + 0.5*V7m``
(coordinates ``(0.20, 0.30, 0.50)``) at ``96.3727`` — a new platform best — and
``0.25*A60 + 0.75*V7m`` (``(0.10, 0.15, 0.75)``) at ``96.3629``.  This module
folds those two measurements into the concave bound, reports the new
bounded/unbounded split, the bound of every still-pending package, and the
measured slope of the N direction at the released ``V7m`` weight.  Zero fits.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .v32_family_ceiling import secant_bound

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v39/SPEC.yaml")
OUTPUT_DEFAULT = Path("local/runs/round2-v38/interior-descent-r1.json")

#: Recorded platform scores lifted to the V12-iron reference, with time-column
#: coordinates in ``(V36, N, V7m)``.
MEASURED = {
    "V36": ((1.00, 0.00, 0.00), 96.2734 + 0.0160, "V36 package"),
    "A35": ((0.65, 0.35, 0.00), 96.3366 + 0.0160, "A35 package"),
    "A45": ((0.55, 0.45, 0.00), 96.3438 + 0.0160, "A45 package"),
    "A60": ((0.40, 0.60, 0.00), 96.3465 + 0.0160, "A60 package"),
    "A72": ((0.28, 0.72, 0.00), 96.3425 + 0.0160, "A72 package"),
    "B0": ((0.325, 0.175, 0.50), 96.3679, "V18_B0"),
    "I1": ((0.20, 0.30, 0.50), 96.3727, "V32_TIME_A60V7_50"),
    "I2": ((0.10, 0.15, 0.75), 96.3629, "V32_TIME_A60V7_75"),
}

#: Packages still pending upload, with their time coordinates.
PENDING = {
    "3_TIME_V100": (0.00, 0.00, 1.00),
    "4_IRON_W100": (0.325, 0.175, 0.50),   # iron endpoint at the B0 time column
    "5_INTERIOR_A60V7_25": (0.30, 0.45, 0.25),
}

IRON_HEAD_ROOM = 0.0160


def design() -> dict[str, tuple[tuple[float, float, float], float]]:
    return {name: (coords, value) for name, (coords, value, _) in MEASURED.items()}


def analysis(step: float = 0.05) -> dict[str, Any]:
    table = design()
    grid = [round(index * step, 6) for index in range(int(round(1.0 / step)) + 1)]
    rows = []
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
    slope = (MEASURED["I1"][1] - MEASURED["B0"][1]) / (MEASURED["I1"][0][1] - MEASURED["B0"][0][1])
    pending = {}
    for name, coords in PENDING.items():
        bound = secant_bound(coords, table)
        entry = {"coordinates": list(coords),
                 "upper_bound": None if bound is None else round(bound[0], 6),
                 "bound_source": None if bound is None else
                 {"from": bound[1][0], "to": bound[1][1], "lambda": round(bound[1][2], 4)}}
        if name == "4_IRON_W100":
            # the iron endpoint keeps the B0 time column, so it is bounded by the
            # measured iron line (w=0 -> V7 package 96.3519, w=0.5 -> B0 96.3679),
            # not by the time simplex
            entry["iron_line_upper_bound"] = round(96.3679 + IRON_HEAD_ROOM, 6)
            entry["bound_source"] = {"from": "V7 package w=0", "to": "B0 w=0.5", "lambda": 1.0}
        pending[name] = entry
    return {
        "step": step,
        "grid_points": total,
        "bounded_grid_points": len(rows),
        "unbounded_grid_points": total - len(rows),
        "measured": {name: {"coordinates": list(coords), "score_with_V12_iron": round(value, 4),
                            "source": label}
                     for name, (coords, value, label) in sorted(MEASURED.items())},
        "measured_best": {"point": best[0], "score": round(best[1][1], 4), "source": best[1][2]},
        "simplex_max_bound_with_V12_iron": round(rows[0][0], 6),
        "simplex_argmax_coordinates": list(rows[0][1]),
        "simplex_argmax_bound_source": {"from": rows[0][2][0], "to": rows[0][2][1],
                                        "lambda": round(rows[0][2][2], 4)},
        "family_ceiling_with_iron_endpoint": round(rows[0][0] + IRON_HEAD_ROOM, 6),
        "ceiling_below_target": round(96.4 - (rows[0][0] + IRON_HEAD_ROOM), 6),
        "n_direction_slope_at_v7m_half": round(float(slope), 6),
        "runner_up_bounds": [{"bound": round(row[0], 4), "coordinates": list(row[1]),
                              "source": {"from": row[2][0], "to": row[2][1]}}
                             for row in rows[1:8]],
        "pending_packages": pending,
        "iron_endpoint_bound": round(96.3679 + IRON_HEAD_ROOM, 6),
        "best_plus_iron_endpoint_bound": round(rows[0][0] + IRON_HEAD_ROOM, 6),
        "new_fits": 0,
        "agent_uploads": 0,
    }


def run(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
        output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    destination = (root / output).resolve()
    if not destination.is_relative_to(root / "local/runs/round2-v38"):
        raise ValueError("V39 evidence must stay private under the historical local/runs/round2-v38 directory")
    if destination.exists():
        raise FileExistsError("V39 output already exists; refusing overwrite")
    report = analysis(float(spec.get("grid_step", 0.05)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("measured_best", "bounded_grid_points",
                                             "unbounded_grid_points",
                                             "simplex_max_bound_with_V12_iron",
                                             "simplex_argmax_coordinates",
                                             "family_ceiling_with_iron_endpoint",
                                             "ceiling_below_target",
                                             "n_direction_slope_at_v7m_half")}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--spec", type=Path, default=SPEC_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.root, args.spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
