"""Forward-only v0.3 gates and paired calendar-week diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..exceptions import ContractError


def acceptance(metrics: dict, summaries: dict, reference: str, policy: dict) -> dict:
    if reference not in {"E12-raw", "E12-CVcal"} or reference not in summaries:
        raise ContractError("v0.3 requires the globally selected complete reference")
    ref = summaries[reference]
    results = {}
    for candidate, summary in summaries.items():
        if candidate in {"E00", "E09", "E04", "B0", "B1", "E12-raw", "E12-CVcal"}:
            continue
        deltas = {h: summary["horizons"][h]["mean_loss"] - ref["horizons"][h]["mean_loss"]
                  for h in ("H1", "H2", "H3", "H4")}
        target_deltas = {t: float(np.mean([summary["horizons"][h][f"{t}_mean_wmape"] - ref["horizons"][h][f"{t}_mean_wmape"]
                                         for h in deltas])) for t in ("iron", "time")}
        long = metrics["DEV_LONG"]["candidates"]
        short = metrics["DEV_SHORT"]["candidates"]
        checks = {
            "J": ref["J"] - summary["J"] >= policy["min_J_improvement"],
            "improved_horizons": sum(d < 0 for d in deltas.values()) >= policy["min_improved_horizons"],
            "horizon_regression": max(deltas.values()) <= policy["max_any_horizon_regression"],
            "target_regression": max(target_deltas.values()) <= policy["per_target_mean_wmape_max_regression"],
            "DEV_LONG": long[candidate]["overall"]["loss"] < min(long[c]["overall"]["loss"] for c in ("B0", "B1")),
            "DEV_SHORT": short[candidate]["overall"]["loss"] - short[reference]["overall"]["loss"] <= policy["dev_short_max_regression"],
        }
        results[candidate] = {"pass": all(checks.values()), "checks": checks,
                              "J_improvement": ref["J"] - summary["J"],
                              "horizon_delta": deltas, "target_delta": target_deltas}
    return {"schema_version": "optimization-acceptance-v3", "C_ref": reference,
            "status": "PASS" if any(r["pass"] for r in results.values()) else "FAIL", "candidates": results}


def paired_week_intervals(errors: pd.DataFrame, candidate: str, reference: str,
                          *, repetitions: int = 1000, seed: int = 2026) -> dict:
    """Identical week multiplicities across origins; recompute cell ratios then J."""
    required = {"candidate", "origin", "horizon", "sample_id", "week", "tap_iron", "tap_time_len",
                "abs_error_tap_iron", "abs_error_tap_time_len"}
    if required - set(errors) or errors.duplicated(["candidate", "origin", "sample_id"]).any():
        raise ContractError("paired diagnostics require unique candidate/origin/sample rows")
    parts = {c: errors.loc[errors.candidate == c].copy() for c in (candidate, reference)}
    if (errors.groupby("sample_id")[["week", "tap_iron", "tap_time_len"]].nunique() > 1).any().any():
        raise ContractError("repeated samples have inconsistent calendar week or targets")
    keys = ["origin", "horizon", "sample_id", "week", "tap_iron", "tap_time_len"]
    if set(map(tuple, parts[candidate][keys].to_numpy())) != set(map(tuple, parts[reference][keys].to_numpy())):
        raise ContractError("paired diagnostics sample, label or scenario identity mismatch")
    if any(set(p.horizon) != {1, 2, 3, 4} for p in parts.values()):
        raise ContractError("paired diagnostics need complete four-horizon data")
    weeks = sorted(errors.week.unique())
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        counts = pd.Series(rng.choice(weeks, len(weeks), replace=True)).value_counts()
        scores = {}
        valid = True
        for c, part in parts.items():
            weighted = part.copy()
            w = weighted.week.map(counts).fillna(0).to_numpy()
            cols = ["tap_iron", "tap_time_len", "abs_error_tap_iron", "abs_error_tap_time_len"]
            weighted[cols] = weighted[cols].mul(w, axis=0)
            cells = weighted.groupby(["origin", "horizon"])[cols].sum()
            if (cells[["tap_iron", "tap_time_len"]] <= 0).any().any():
                valid = False
                break
            cells["loss"] = .5 * (cells.abs_error_tap_iron / cells.tap_iron + cells.abs_error_tap_time_len / cells.tap_time_len)
            scores[c] = cells.loss.groupby("horizon").mean().to_numpy()
        if valid:
            diff = scores[candidate] - scores[reference]
            deltas.append([*diff, diff.mean()])
    if not deltas:
        raise ContractError("no valid resamples with positive cell denominators")
    quantiles = np.quantile(deltas, [.025, .5, .975], axis=0)
    return {"candidate": candidate, "reference": reference, "seed": seed,
            "requested_repetitions": repetitions, "valid_repetitions": len(deltas),
            "deltas": {h: dict(zip(("p025", "median", "p975"), quantiles[:, i].tolist()))
                       for i, h in enumerate(("H1", "H2", "H3", "H4", "J"))},
            "scope": "development_stability_only_not_selection_adjusted_or_holdout"}
