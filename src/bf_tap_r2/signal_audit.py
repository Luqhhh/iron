"""Zero-fit signal diagnostics; no old-data access, model search or uploads."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml

from .audit import digest, write_json
from .cv import verify_audit
from .data import FEATURES, TARGETS, load_config, load_snapshot, resolve_file


def unit_centered(values):
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=0)
    norm = np.sqrt((centered ** 2).sum(axis=0))
    return np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)


def correlations(x, y):
    return unit_centered(x).T @ unit_centered(y)


def permutation_family(x, y, names, n_permutations, seed):
    """Same row permutation for both targets preserves their joint dependence."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or x.ndim != 2 or y.ndim != 2 or len(x) < 3:
        raise ValueError("Invalid correlation arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or n_permutations < 1:
        raise ValueError("Finite arrays and positive permutation count required")
    xs = [unit_centered(x), unit_centered(rankdata(x, axis=0))]
    ys = [unit_centered(y), unit_centered(rankdata(y, axis=0))]
    observed = np.stack([a.T @ b for a, b in zip(xs, ys)])
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for iteration in range(n_permutations):
        order = rng.permutation(len(y))
        null[iteration] = max(float(np.abs(a.T @ b[order]).max()) for a, b in zip(xs, ys))
    records = []
    for method, method_name in enumerate(("pearson", "spearman")):
        for i, name in enumerate(names):
            for j, target in enumerate(TARGETS):
                value = float(observed[method, i, j])
                p = float((1 + (null >= abs(value) - 1e-14).sum()) / (n_permutations + 1))
                records.append({"feature": name, "target": target, "method": method_name,
                                "correlation": value, "family_adjusted_p": p,
                                "three_scope_adjusted_p": min(1., 3 * p)})
    return {"n": len(y), "tests_in_family": observed.size, "permutations": n_permutations,
            "maximum_absolute_correlation": float(np.abs(observed).max()),
            "null_max_quantiles": {str(q): float(np.quantile(null, q)) for q in (.5, .9, .95, .99)},
            "records": records,
            "interpretation": "Permutation calibration assumes exchangeability within scope; not an independence proof"}


def constructions(frame):
    spout2 = frame.spout_no.eq(2).astype(float)
    return pd.DataFrame({
        "cold_air_press_minus_hot_air_press": frame.cold_air_press - frame.hot_air_press,
        "upper_press_diff_minus_lower_press_diff": frame.upper_press_diff - frame.lower_press_diff,
        "air_volume_times_oxygen": frame.air_volume * frame.oxygen,
        "spout_2_times_air_volume": spout2 * frame.air_volume,
        "spout_2_times_oxygen": spout2 * frame.oxygen,
        "spout_2_times_total_press_diff": spout2 * frame.total_press_diff,
        "spout_2_times_pig": spout2 * frame.pig,
    }, index=frame.index)


def independent_alignment(root, cfg, frame):
    """CSV dictionary lookups independently cross-check the pandas join by ID."""
    checked = 0
    for key, columns in (("train_samples", ("spout_no", *TARGETS)), ("train_features", FEATURES)):
        with resolve_file(root, cfg[key]).open(encoding="utf-8-sig", newline="") as handle:
            raw = list(csv.DictReader(handle))
        mapping = {row["sample_id"]: row for row in raw}
        if len(mapping) != len(raw) or set(mapping) != set(frame.sample_id):
            raise ValueError("Independent CSV ID set mismatch")
        for name in columns:
            expected = np.array([float(mapping[sid][name]) for sid in frame.sample_id])
            np.testing.assert_allclose(frame[name], expected, rtol=1e-14, atol=1e-12)
            checked += len(frame)
    return {"status": "pass", "numeric_cells_checked": checked, "method": "stdlib_csv_id_dictionary_lookup"}


def oof_diagnostics(parent, frame):
    result = {}
    for seed in (42, 3407):
        baseline = pd.read_csv(parent / "oof" / f"{seed}-median_global.csv").set_index("sample_id").loc[frame.sample_id]
        for route in ("L1", "L2", "S1", "C1", "C2"):
            for target in TARGETS:
                path = parent / "oof" / f"{seed}-{route}-{target}.csv"
                if not path.exists():
                    continue
                oof = pd.read_csv(path).set_index("sample_id")
                if not oof.index.is_unique or set(oof.index) != set(frame.sample_id):
                    raise ValueError("OOF identity mismatch")
                oof = oof.loc[frame.sample_id]
                y = frame[target].to_numpy()
                np.testing.assert_allclose(oof[target], y, rtol=1e-12)
                pred = oof[f"pred_{target}"].to_numpy()
                base = baseline[f"pred_{target}"].to_numpy()
                error_delta = np.abs(y-pred) - np.abs(y-base)
                boundary = (y == y.min()) | (y == y.max())
                result[f"{seed}-{route}-{target}"] = {
                    "prediction_std": float(pred.std()), "target_std": float(y.std()),
                    "oof_pearson": float(correlations(pred[:, None], y[:, None])[0, 0]),
                    "oof_spearman": float(correlations(rankdata(pred)[:, None], rankdata(y)[:, None])[0, 0]),
                    "fraction_lower_absolute_error_than_baseline": float((error_delta < 0).mean()),
                    "absolute_error_delta_sum": float(error_delta.sum()),
                    "boundary_rows": int(boundary.sum()),
                    "boundary_error_delta_sum": float(error_delta[boundary].sum()),
                    "interior_error_delta_sum": float(error_delta[~boundary].sum())}
    return result


def run(root, output):
    cfg_path = root / "configs/round2_v0_2/signal_audit.yaml"
    spec = yaml.safe_load(cfg_path.read_text())
    manifest = verify_audit(root, root / spec["audit_run"])
    parent = root / spec["parent_run"]
    verification = json.loads((parent / "cold_verification_with_test.json").read_text())
    if verification["status"] != "pass" or digest(parent / "summary.json") != verification["summary_sha256"]:
        raise ValueError("Parent evidence identity failed")
    if not output.resolve().is_relative_to(root.resolve() / "local/runs/round2-v0.2"):
        raise ValueError("Evidence must remain under the private round2-v0.2 directory")
    output.mkdir(parents=True, exist_ok=False)
    try:
        cfg = load_config(root / "configs/round2_v0_1/data.yaml")
        frame, _ = load_snapshot(root, cfg, "train")
        frame = frame.sort_values("sample_id").reset_index(drop=True)
        write_json(output / "manifest.json", {
            "version": spec["version"], "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "config_sha256": digest(cfg_path), "source_sha256": digest(Path(__file__)),
            "data_identity": manifest["files"], "parent_summary_sha256": digest(parent / "summary.json"),
            "uv_lock_sha256": digest(root / "uv.lock"), "model_fits": 0})
        output_config = output / "config.yaml"
        with output_config.open("xb") as handle:
            handle.write(cfg_path.read_bytes())
        audit = {"alignment": independent_alignment(root, cfg, frame), "raw_features": {}, "constructions": {},
                 "target_correlation": {"pearson": float(frame[list(TARGETS)].corr().iloc[0, 1]),
                                        "spearman": float(frame[list(TARGETS)].corr(method="spearman").iloc[0, 1])},
                 "spout_label_correlations": {t: float(frame[["spout_no", t]].corr().iloc[0, 1]) for t in TARGETS},
                 "model_fits": 0, "preprocessor_fits": 0, "platform_uploads": 0}
        for i, (scope, subset) in enumerate([("all", frame), *[(f"spout_{s}", f) for s, f in frame.groupby("spout_no")]]):
            for family, x in (("raw_features", subset[list(FEATURES)]), ("constructions", constructions(subset))):
                report = permutation_family(x.to_numpy(), subset[list(TARGETS)].to_numpy(), x.columns.tolist(),
                                            spec["permutations"], spec["permutation_seed"] + i)
                audit[family][scope] = report
                print(json.dumps({"family": family, "scope": scope,
                                  "max_abs_r": report["maximum_absolute_correlation"],
                                  "min_adjusted_p": min(r["three_scope_adjusted_p"] for r in report["records"])}), flush=True)
        audit["oof_diagnostics"] = oof_diagnostics(parent, frame)
        audit["limitations"] = ["Full-training exploratory audit, not held-out model selection",
                                "Permutation calibration requires within-scope exchangeability",
                                "Pearson/Spearman and seven constructions cannot rule out other nonlinear signals",
                                "Target association cannot be used as a true-target test input",
                                "Raw-feature and construction families are separate; do not pool their p-values as one corrected family"]
        write_json(output / "signal_audit.json", audit)
        return audit
    except Exception as exc:
        write_json(output / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
