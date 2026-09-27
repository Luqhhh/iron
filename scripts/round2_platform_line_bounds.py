"""Conditional upper bounds for a fixed affine prediction line; never fits models.

The un-floored score 100 - 50*(WMAPE_iron + WMAPE_time) is concave
in an affine blend weight. A secant bounds it ABOVE only outside the
secant interval. This utility must not be used as an interpolation forecast.
Reported scores need a stated error interval and identical evaluation rows.
"""
from fractions import Fraction
from itertools import combinations


def _q(value):
    return Fraction(str(value))


def concave_upper_bound(points, domain=(0, 1), score_error="0.0001"):
    """Bound the supremum using robust secant extensions, with exact arithmetic.

    Each (alpha, reported_score) observation means the true, un-floored
    score is within +/- score_error. Infinity means the supplied secants
    cannot give a finite upper bound. No feasibility or platform claim is
    made; the result is conditional on those observations and concavity.
    """
    data = [(_q(a), _q(s)) for a, s in points]
    lo, hi = map(_q, domain)
    eps = _q(score_error)
    if len(data) < 2 or eps < 0 or lo >= hi:
        raise ValueError("Need two points, nonnegative error and a nonempty domain")
    if any(b[0] <= a[0] for a, b in zip(data, data[1:])):
        raise ValueError("Observation weights must be strictly increasing")
    edges = sorted({lo, hi, *(a for a, _ in data if lo < a < hi)})
    pieces = []
    for left, right in zip(edges, edges[1:]):
        lines = []
        for (a, sa), (b, sb) in combinations(data, 2):
            if b <= left:
                # Positive coefficient for sb, negative coefficient for sa.
                slope = (sb + eps - sa + eps) / (b - a)
                lines.append((slope, sb + eps - slope * b))
            elif a >= right:
                # Positive coefficient for sa, negative coefficient for sb.
                slope = (sb - eps - sa - eps) / (b - a)
                lines.append((slope, sa + eps - slope * a))
        if not lines:
            return {"upper_bound": float("inf"), "pieces": pieces}
        candidates = {left, right}
        for (m, c), (n, d) in combinations(lines, 2):
            if m != n:
                x = (d - c) / (m - n)
                if left <= x <= right:
                    candidates.add(x)
        # The minimum of affine bounds is piecewise affine and concave.
        value, at = max((min(m*x+c for m, c in lines), x) for x in candidates)
        pieces.append({"interval": [float(left), float(right)],
                       "upper_bound": float(value), "at": float(at),
                       "exact_upper_bound": str(value)})
    return {"upper_bound": max(p["upper_bound"] for p in pieces),
            "pieces": pieces, "score_error": float(eps),
            "meaning": "conditional upper bound, not attainable score or forecast"}
