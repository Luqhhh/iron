"""Conditional continuous concavity bounds, with rational dual certificates."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import yaml


def _solve_exact(matrix, rhs):
    """Solve a full-column-rank rectangular system using rational elimination."""
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    n = len(matrix[0])
    for col in range(n):
        pivot = next((k for k in range(col, len(rows)) if rows[k][col]), None)
        if pivot is None:
            raise ValueError("Non-basic dual certificate")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [v / scale for v in rows[col]]
        for k in range(len(rows)):
            if k != col:
                scale = rows[k][col]
                rows[k] = [v - scale * w for v, w in zip(rows[k], rows[col])]
    if any(not any(row[:-1]) and row[-1] for row in rows):
        raise ValueError("Inconsistent exact certificate")
    return [rows[k][-1] for k in range(n)]


def upper_at(design, anchor, query):
    """One-anchor upper bound; None means this anchor is unbounded."""
    names = list(design)
    coords = {k: [F(str(v)) for v in design[k][0]] for k in names}
    upper = F(str(design[anchor][1])) + F(str(design[anchor][2]))
    lower = {k: F(str(design[k][1])) - F(str(design[k][2])) for k in names}
    dim = len(coords[anchor])
    if len(query) != dim or any(len(v) != dim for v in coords.values()):
        raise ValueError("Coordinate dimensions differ")
    if any(F(str(design[k][2])) < 0 for k in names):
        raise ValueError("Negative uncertainty")
    a = [[x - y for x, y in zip(coords[anchor], coords[k])] for k in names]
    b = [upper - lower[k] for k in names]
    d = [F(str(x)) - y for x, y in zip(query, coords[anchor])]
    result = linprog(-np.array(d, float), A_ub=np.array(a, float),
                     b_ub=np.array(b, float), bounds=[(None, None)] * dim,
                     method="highs")
    if result.status == 3:
        return None
    if not result.success:
        raise ValueError(f"Inconsistent or failed concavity system: {result.message}")
    support = [i for i, v in enumerate(-result.ineqlin.marginals) if v > 1e-10]
    if support:
        weights = _solve_exact([[a[i][j] for i in support] for j in range(dim)], d)
    else:
        weights = []
    if any(v < 0 for v in weights) or any(
        sum((a[i][j] * v for i, v in zip(support, weights)), F(0)) != d[j]
        for j in range(dim)
    ):
        raise ValueError("Invalid rational dual certificate")
    bound = upper + sum((b[i] * v for i, v in zip(support, weights)), F(0))
    return {"upper": math.nextafter(float(bound), math.inf),
            "exact_upper": str(bound), "anchor": anchor,
            "dual_weights": {names[i]: str(v) for i, v in zip(support, weights)}}


def continuous_bound(design, vertices):
    """Bound the whole convex hull via the convex envelope of each anchor."""
    anchors = {}
    for anchor in design:
        certificates = [upper_at(design, anchor, vertex) for vertex in vertices]
        if all(c is not None for c in certificates):
            anchors[anchor] = {"upper": max(c["upper"] for c in certificates),
                               "vertex_certificates": certificates}
    if not anchors:
        raise ValueError("No anchor bounds the continuous domain")
    selected = min(anchors, key=lambda k: anchors[k]["upper"])
    return {"upper": anchors[selected]["upper"], "selected_anchor": selected,
            "anchors": anchors, "domain_vertices": vertices}


def analyze(spec):
    time = continuous_bound(spec["time_observations"], spec["time_domain_vertices"])
    iron = continuous_bound(spec["iron_observations"], spec["iron_domain_vertices"])
    _, score, error = spec["iron_observations"]["W05"]
    baseline_lower = math.nextafter(float(F(str(score)) - F(str(error))), -math.inf)
    iron_gain = math.nextafter(iron["upper"] - baseline_lower, math.inf)
    total = math.nextafter(max(0.0, time["upper"] + iron_gain), math.inf)
    probes = {name: min((c for a in spec["time_observations"]
                        if (c := upper_at(spec["time_observations"], a, point)) is not None),
                       key=lambda c: c["upper"])
              for name, point in {"H1": [0, .45], "H2": [.1, .45],
                                  "SEG625": [.25, .375], "N500": [0, .5]}.items()}
    return {"time": time, "iron": iron, "iron_gain_upper": iron_gain,
            "family_upper": total, "platform_target": spec["platform_target"],
            "target_excluded_under_assumptions": total < spec["platform_target"],
            "probe_upper_bounds": probes, "new_fits": 0, "packages": 0,
            "uploads": 0, "conditional_on": "documented metric, reported scores and package identities"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", default="local/runs/round2-v41/bound-r1.json")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / "local/runs/round2-v41"):
        raise ValueError("Evidence must stay in the V41 private run directory")
    spec = yaml.safe_load((root / "configs/round2_v41/SPEC.yaml").read_text())
    report = analyze(spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("family_upper", "iron_gain_upper",
                                            "target_excluded_under_assumptions")}))


if __name__ == "__main__":
    main()
