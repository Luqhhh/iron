"""Bounded, zero-target-fit diagnostics for the 2026-09-26 strategy review.

Official target values are used only for descriptive correlation. Domain
classifiers predict train/test membership, never either competition target.
Historical scoring, release gates, predictions and cached models are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, TARGETS
from .v5_resolution import nested_blend


def audit_selection_boundary(selection_ids, evaluation_ids, prediction_training_ids):
    """Require row-disjoint selection and training provenance for an outer test.

    This is an attestation check, not proof that a supplied ledger is truthful.
    Selection OOF models must also exclude the outer evaluation rows.
    """
    selected, evaluated = set(selection_ids), set(evaluation_ids)
    if not selected or not evaluated:
        raise ValueError("Selection and evaluation IDs must be nonempty")
    if len(selected) != len(selection_ids) or len(evaluated) != len(evaluation_ids):
        raise ValueError("Duplicate row IDs")
    if selected & evaluated:
        raise ValueError("Evaluation labels occur in selection rows")
    training_sets = list(prediction_training_ids)
    if not training_sets:
        raise ValueError("Missing selection-prediction training provenance")
    for fitted_ids in training_sets:
        fitted = set(fitted_ids)
        if not fitted or not fitted <= selected or fitted & evaluated:
            raise ValueError("Selection prediction model includes outer evaluation rows")
    return {"selection_rows": len(selected), "evaluation_rows": len(evaluated),
            "provenance_records": len(training_sets), "row_boundary": "PASS"}


def shared_label_probe():
    """Show that legacy cross-seed alpha selection reads held-fold labels."""
    seeds = (42, 3407)
    folds = {42: np.repeat(np.arange(5), 2),
             3407: np.roll(np.repeat(np.arange(5), 2), 3)}
    base = {s: np.full(10, 10.0) for s in seeds}
    candidate = {s: np.full(10, 12.0) for s in seeds}
    before = np.full(10, 11.0)
    before[folds[42] == 0] = 10.0
    after = before.copy()
    after[folds[42] == 0] = 12.0
    first = nested_blend(before, folds, base, candidate, [0.0, 1.0])
    second = nested_blend(after, folds, base, candidate, [0.0, 1.0])
    return {"synthetic_only": True, "changed_rows": [0, 1],
            "changed_rows_are_seed42_evaluation_fold0": True,
            "prediction_arrays_unchanged": True,
            "seed42_alpha_before": first["alphas"][42],
            "seed42_alpha_after": second["alphas"][42],
            "interpretation": "cross-seed selection is not row-disjoint nested validation"}


def alpha_curve_audit(points):
    """Check the concavity implied by fixed-endpoint absolute-error scoring.

    A fixed test frame and fixed endpoint columns give a concave piecewise-linear
    score curve. A previous secant bounds later slopes, but cannot locate a peak.
    """
    ordered = sorted((float(a), float(s)) for a, s in points)
    if len(ordered) < 2 or any(not np.isfinite([a, s]).all() for a, s in ordered):
        raise ValueError("Need at least two finite curve observations")
    if any(a < 0 or a > 1 for a, _ in ordered):
        raise ValueError("Alpha must be in [0, 1]")
    if any(left[0] == right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Duplicate alpha")
    slopes = [(b[1] - a[1]) / (b[0] - a[0])
              for a, b in zip(ordered, ordered[1:])]
    # Four-decimal platform receipts have at most 0.00005 rounding error each.
    tolerance = [0.0001 / (b[0] - a[0])
                 for a, b in zip(ordered, ordered[1:])]
    consistent = all(b <= a + ta + tb for a, b, ta, tb in
                     zip(slopes, slopes[1:], tolerance, tolerance[1:]))
    last_alpha, last_score = ordered[-1]
    extension_bound = last_score + max(0.0, slopes[-1]) * (1 - last_alpha)
    return {"points": ordered, "secant_slopes": slopes,
            "consistent_with_concavity_at_receipt_precision": consistent,
            "maximum_score_on_remaining_alpha_interval_upper_bound": extension_bound,
            "upper_bound_is_approximate_due_to_receipt_rounding": True,
            "peak_location": "not_identified_by_these_points",
            "upper_bound_is_not_prediction": True}


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid p values")
    order = np.argsort(values, kind="stable")
    ranked = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    result = np.empty_like(values)
    result[order] = np.minimum(1.0, ranked)
    return result


def _read_inputs(root):
    frames = []
    for stage, expected in (("train", 2754), ("test", 322)):
        samples = pd.read_csv(root / f"复赛_{stage}/{stage}_samples.csv")
        features = pd.read_csv(root / f"复赛_{stage}/{stage}_features.csv")
        if set(features.columns) != {"sample_id", *FEATURES}:
            raise ValueError("Unexpected numeric feature schema")
        required = {"sample_id", "spout_no", *(TARGETS if stage == "train" else ())}
        if set(samples.columns) != required:
            raise ValueError("Unexpected sample schema")
        for table in (features, samples):
            if table.sample_id.isna().any() or table.sample_id.duplicated().any():
                raise ValueError("Invalid sample IDs")
            if not table.sample_id.str.startswith(f"R2S2_{stage.upper()}_").all():
                raise ValueError("Foreign snapshot IDs")
        if set(samples.sample_id) != set(features.sample_id):
            raise ValueError("Sample and feature IDs differ")
        merged = samples.merge(features, on="sample_id", validate="one_to_one", sort=False)
        if len(merged) != expected:
            raise ValueError("Unexpected snapshot row count")
        frames.append(merged)
    return frames


def distribution_audit(train, test):
    columns = [*FEATURES, "spout_no"]
    # IDs and target values cannot enter this feature-only domain classification.
    x = pd.concat([train[columns], test[columns]], ignore_index=True)
    domain = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
    split = list(StratifiedKFold(5, shuffle=True, random_state=926).split(x, domain))
    models = {
        "linear": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=1000, random_state=926)),
        "small_hgb": make_pipeline(SimpleImputer(strategy="median"),
                                  HistGradientBoostingClassifier(max_iter=80,
                                      max_leaf_nodes=7, min_samples_leaf=25,
                                      l2_regularization=10.0, learning_rate=0.05,
                                      early_stopping=False, random_state=926)),
    }
    domain_results = {}
    for name, model in models.items():
        pred = np.empty(len(domain))
        fold_auc = []
        for fitted, held in split:
            model.fit(x.iloc[fitted], domain[fitted])
            pred[held] = model.predict_proba(x.iloc[held])[:, 1]
            fold_auc.append(float(roc_auc_score(domain[held], pred[held])))
        domain_results[name] = {"oof_auc": float(roc_auc_score(domain, pred)),
                                "fold_auc": fold_auc}
    feature_results = []
    for name in columns:
        a, b = train[name].dropna().to_numpy(), test[name].dropna().to_numpy()
        if not len(a) or not len(b):
            raise ValueError("A feature is entirely missing")
        ks = ks_2samp(a, b)
        scale = float(np.std(a))
        feature_results.append({"feature": name,
            "standardized_mean_difference": float((np.mean(b) - np.mean(a)) / scale) if scale else None,
            "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
            "train_missing": int(train[name].isna().sum()),
            "test_missing": int(test[name].isna().sum()),
            "test_outside_training_range": int(((b < min(a)) | (b > max(a))).sum())})
    adjusted = holm_adjust([r["ks_pvalue"] for r in feature_results])
    for row, pvalue in zip(feature_results, adjusted):
        row["ks_holm_pvalue"] = float(pvalue)
    all_hashes = pd.util.hash_pandas_object(x, index=False)
    train_hashes, test_hashes = all_hashes.iloc[:len(train)], all_hashes.iloc[len(train):]
    duplicates = int(test_hashes.isin(set(train_hashes)).sum())
    target_corr = float(spearmanr(train[TARGETS[0]], train[TARGETS[1]]).statistic)
    return {"features_only_domain_classifier": domain_results,
            "feature_differences": sorted(feature_results, key=lambda r: -r["ks_statistic"]),
            "test_feature_rows_matching_training_hashes": duplicates,
            "descriptive_training_target_spearman": target_corr,
            "spout_proportions": {"train": train.spout_no.value_counts(normalize=True).to_dict(),
                                  "test": test.spout_no.value_counts(normalize=True).to_dict()},
            "interpretation_limit": "feature drift cannot identify unseen target concept shift; row-wise CV assumes exchangeability"}


def run(root, output):
    output = output.resolve()
    if not output.is_relative_to((root / "local").resolve()):
        raise ValueError("Diagnostic reports must stay under local/")
    if output.exists():
        raise ValueError("Never overwrite earlier evidence")
    train, test = _read_inputs(root)
    sources = [root / "EVIDENCE_STATUS.json", Path(__file__),
               root / "src/bf_tap_r2/v5_resolution.py", root / "src/bf_tap_r2/metrics.py"]
    sources += [root / f"复赛_{s}/{s}_{kind}.csv" for s in ("train", "test")
                for kind in ("samples", "features")]
    evidence = json.loads((root / "EVIDENCE_STATUS.json").read_text(encoding="utf-8"))
    result = {"verification_status": "ANALYZED_WITH_VERIFIED_SYNTHETIC_COUNTEREXAMPLE",
        "snapshot": evidence["round2_team_strategy_review_2026_09_26"]["team_snapshot_commit"],
        "official_target_model_fits": 0,
        "domain_classifier_fits": 10, "full_data_target_fits": 0, "packages": 0,
        "uploads": 0,
        "runtime": {"python": sys.version, "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn")},
            "not_the_frozen_v7_v13_neural_runtime": True},
        "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "row_counts": {"train": len(train), "test": len(test)},
        "current_platform_record": evidence["round2_current_platform_best"],
        "shared_label_probe": shared_label_probe(),
        "alpha_curve": alpha_curve_audit([(0, 96.2734), (0.05, 96.2844),
                                           (0.2, 96.3143), (0.35, 96.3366)]),
        "distribution": distribution_audit(train, test),
        "latest_private_caches_available_here": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"report": str(output), "probe": result["shared_label_probe"],
                      "domain_auc": result["distribution"]["features_only_domain_classifier"],
                      "alpha_curve": result["alpha_curve"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            parser.error(f"Set {name}=1 for this bounded diagnostic")
    run(Path.cwd(), args.output)


if __name__ == "__main__":
    main()
