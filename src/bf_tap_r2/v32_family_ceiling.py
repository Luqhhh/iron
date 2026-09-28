#!/usr/bin/env python3
"""V32: the proven platform ceiling of the whole measured family.

Given the documented additive metric and the recorded platform scores, the score
is a concave function of the column mixture weights (each absolute residual is
convex in the prediction, and the prediction is linear in the weights).  A secant
extended outside its chord is therefore an upper bound, and for any point that
lies beyond a measured chord the bound is computable.

This module evaluates that bound over the whole time simplex ``{V36, N, V7m}``
and adds the separately bounded iron-axis head-room, producing the tightest
provable ceiling of the measured family.  Zero fits; never uploads.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

ROOT_DEFAULT = Path("/home/lux1/iron")
SPEC_DEFAULT = Path("configs/round2_v32/SPEC.yaml")
OUTPUT_DEFAULT = Path("local/runs/round2-v32/family-ceiling-r1.json")

#: Recorded platform scores with their time-column coordinates in
#: ``(V36, N, V7m)``.  Each is lifted to the V12-iron reference by adding the
#: measured iron effect (+0.0160) where the score was recorded with V36 iron.
MEASURED_TIME = {
    "V36": ((1.00, 0.00, 0.00), 96.2734, 0.0160),
    "A35": ((0.65, 0.35, 0.00), 96.3366, 0.0160),
    "A45": ((0.55, 0.45, 0.00), 96.3438, 0.0160),
    "A60": ((0.40, 0.60, 0.00), 96.3465, 0.0160),
    "A72": ((0.28, 0.72, 0.00), 96.3425, 0.0160),
    "B0": ((0.325, 0.175, 0.50), 96.3679, 0.0),
}

#: Iron-axis head-room: the line is measured at w=0 (A35 iron) and w=0.5 (V12).
#: Concavity caps w=1.0 at twice the observed +0.0160 effect.
IRON_HEAD_ROOM = 0.0160


def measured_table() -> dict[str, tuple[tuple[float, float, float], float]]:
    return {name: (coords, base + lift) for name, (coords, base, lift) in MEASURED_TIME.items()}


def secant_bound(point: Sequence[float],
                 design: Mapping[str, tuple[Sequence[float], float]]) -> tuple[float, tuple[str, str, float]] | None:
    """Upper bound on a concave function at ``point`` from an extended chord."""
    names = list(design)
    target = np.asarray(point, dtype=float)
    best: float | None = None
    source: tuple[str, str, float] | None = None
    for left, right in itertools.permutations(names, 2):
        start = np.asarray(design[left][0], dtype=float)
        direction = np.asarray(design[right][0], dtype=float) - start
        offset = target - start
        active = [index for index in range(len(direction)) if abs(direction[index]) > 1e-12]
        if len(active) < 2:
            continue
        first = active[0]
        lam = float(offset[first] / direction[first])
        if any(abs(offset[index] / direction[index] - lam) > 1e-9 for index in active[1:]):
            continue
        if lam > 1.0 + 1e-9 or lam < -1e-9:
            value = design[left][1] + lam * (design[right][1] - design[left][1])
            if best is None or value < best:
                best = value
                source = (left, right, lam)
    return None if best is None else (float(best), source)


def simplex_ceiling(step: float = 0.05) -> dict[str, Any]:
    design = measured_table()
    grid = [round(index * step, 6) for index in range(int(round(1.0 / step)) + 1)]
    rows = []
    for a in grid:
        for b in grid:
            c = round(1.0 - a - b, 10)
            if c < -1e-9:
                continue
            bound = secant_bound((a, b, c), design)
            if bound is not None:
                rows.append((bound[0], (a, b, c), bound[1]))
    rows.sort(key=lambda row: -row[0])
    top = rows[0]
    measured_best = max(design.items(), key=lambda item: item[1][1])
    total_points = 0
    for a in grid:
        for b in grid:
            if (1.0 - a - b) >= -1e-9:
                total_points += 1
    return {
        "step": step,
        "grid_points": total_points,
        "bounded_grid_points": len(rows),
        "unbounded_grid_points": total_points - len(rows),
        "bounded_region_scope": ("the concave secant bound is only computable for points that lie beyond a "
                                 "measured chord; the unbounded points are interior mixtures (time-simplex "
                                 "V7m weight below 0.5). The ceiling below therefore bounds the beyond-chord "
                                 "region that contains the pure-V7m and iron-endpoint corner, NOT the whole "
                                 "simplex: an interior mixture is unmeasured and unconstrained."),
        "measured_points": {name: {"coordinates": list(coords), "score_with_V12_iron": round(value, 4)}
                            for name, (coords, value) in sorted(design.items())},
        "iron_head_room_to_full_weight": IRON_HEAD_ROOM,
        "simplex_max_bound_with_V12_iron": round(top[0], 6),
        "simplex_argmax_coordinates": list(top[1]),
        "simplex_argmax_source": {"from": top[2][0], "to": top[2][1], "lambda": round(top[2][2], 4)},
        "family_ceiling_with_iron_endpoint": round(top[0] + IRON_HEAD_ROOM, 6),
        "ceiling_below_target": round(96.4 - (top[0] + IRON_HEAD_ROOM), 6),
        "measured_best": {"point": measured_best[0], "score": round(measured_best[1][1], 4)},
        "runner_up_bounds": [{"bound": round(row[0], 4), "coordinates": list(row[1]),
                              "source": {"from": row[2][0], "to": row[2][1]}}
                             for row in rows[1:6]],
        "new_fits": 0,
        "agent_uploads": 0,
    }


def run(root: Path | str = ROOT_DEFAULT, spec_path: Path | str = SPEC_DEFAULT,
        output: Path | str = OUTPUT_DEFAULT) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = yaml.safe_load((root / spec_path).read_text())
    destination = (root / output).resolve()
    if not destination.is_relative_to(root / "local/runs/round2-v32"):
        raise ValueError("V32 evidence must stay private under local/runs/round2-v32")
    if destination.exists():
        raise FileExistsError("V32 output already exists; refusing overwrite")
    report = simplex_ceiling(float(spec.get("grid_step", 0.05)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "simplex_max_bound_with_V12_iron", "simplex_argmax_coordinates",
        "simplex_argmax_source", "family_ceiling_with_iron_endpoint",
        "ceiling_below_target", "measured_best")}, indent=2))
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
