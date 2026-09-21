"""Independent pairing, stratified random controls and saved-model diagnostics."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from .audit import digest
from .data import FEATURES, TARGETS, read_table, resolve_file
from .metrics import wmape
from .signal_audit import independent_alignment, unit_centered


def pairing_check(root, cfg, frame):
    report = independent_alignment(root, cfg, frame)
    samples = read_table(resolve_file(root, cfg["train_samples"]), ("sample_id", "spout_no", *TARGETS))
    features = read_table(resolve_file(root, cfg["train_features"]), ("sample_id", *FEATURES))
    rebuilt = pd.DataFrame({"sample_id": frame.sample_id})
    # Independent dictionaries, not join_snapshot or a shared row-index assumption.
    for raw in (samples, features):
        mapping = raw.set_index("sample_id").to_dict(orient="index")
        for column in raw.columns.drop("sample_id"):
            rebuilt[column] = [mapping[sid][column] for sid in frame.sample_id]
    pd.testing.assert_frame_equal(rebuilt[frame.columns].reset_index(drop=True), frame.reset_index(drop=True), check_dtype=False)
    for a, b in ((13, 71), (71, 13)):
        perturbed = samples.sample(frac=1, random_state=a).merge(
            features.sample(frac=1, random_state=b), on="sample_id", how="left", validate="one_to_one")
        perturbed = perturbed.sort_values("sample_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(perturbed[frame.columns], frame.reset_index(drop=True))
    return {**report, "independent_X_y_rebuild": "pass", "independently_shuffled_tables": "pass",
            "calls_join_snapshot": False}


def stratified_signal(frame, spec):
    records, prepared = [], []
    for scope, subset in [("all", frame), *[(f"spout_{s}", f) for s, f in frame.groupby("spout_no")]]:
        x, y = subset[list(FEATURES)].to_numpy(), subset[list(TARGETS)].to_numpy()
        xs = [unit_centered(x), unit_centered(rankdata(x, axis=0))]
        ys = [unit_centered(y), unit_centered(rankdata(y, axis=0))]
        values = [a.T @ b for a, b in zip(xs, ys)]
        for method, matrix in zip(("pearson", "spearman"), values):
            for i, feature in enumerate(FEATURES):
                for j, target in enumerate(TARGETS):
                    records.append({"scope": scope, "method": method, "feature": feature,
                                    "target": target, "correlation": float(matrix[i, j])})
        if scope != "all":
            prepared.append((xs, ys))
    observed = max(abs(r["correlation"]) for r in records if r["scope"] != "all")
    rng = np.random.default_rng(spec["seed"])
    null = []
    for _ in range(spec["count"]):
        value = 0.
        for xs, ys in prepared:
            order = rng.permutation(len(ys[0]))
            value = max(value, *(float(np.abs(x.T @ y[order]).max()) for x, y in zip(xs, ys)))
        null.append(value)
    p = (1 + np.count_nonzero(np.asarray(null) >= observed - 1e-14)) / (len(null) + 1)
    return {"records": records, "max_abs_within_spout": observed, "permutation_p_plus_one": p,
            "null_maxima": null, "null_max_p95": float(np.quantile(null, .95)),
            "scheme": spec, "target_pearson": float(frame[list(TARGETS)].corr().iloc[0, 1]),
            "limitations": "Diagnostic only; assumes within-spout exchangeability, not absence of all nonlinear signal"}


def saved_model_diagnostics(parent, frame, assignments):
    events = [json.loads(line) for line in (parent / "fit_ledger.jsonl").read_text().splitlines()]
    rows, coefficients = [], []
    for event in events:
        if event["kind"] != "regressor_fit_complete":
            continue
        seed, route, target, fold = (event[k] for k in ("seed", "route", "target", "fold"))
        folds = assignments.loc[assignments.seed == seed].set_index("sample_id").loc[frame.sample_id, "fold"].to_numpy()
        training, validation = frame.loc[folds != fold], frame.loc[folds == fold]
        path = parent / "models" / f"{seed}-{route}-{target}-fold{fold}.joblib"
        if digest(path) != event["model_sha256"]:
            raise ValueError("Parent model changed")
        model = joblib.load(path)
        with threadpool_limits(limits=1):
            fit_prediction = model.predict(training)
            valid_prediction = model.predict(validation)
        row = {k: event[k] for k in ("seed", "route", "target", "fold")}
        row.update({"training_wmape": wmape(training[target], fit_prediction),
                    "validation_wmape": wmape(validation[target], valid_prediction),
                    "train_prediction_std": float(np.std(fit_prediction)), "validation_prediction_std": float(np.std(valid_prediction)),
                    "train_prediction_quantiles": np.quantile(fit_prediction, [0, .05, .5, .95, 1]).tolist(),
                    "validation_prediction_quantiles": np.quantile(valid_prediction, [0, .05, .5, .95, 1]).tolist()})
        if route in ("L1", "L2", "S1"):
            pre = model.estimator_.named_steps["preprocess"]
            coef = model.estimator_.named_steps["regression"].coef_
            row["nonzero_coefficients_abs_gt_1e_10"] = int(np.count_nonzero(np.abs(coef) > 1e-10))
            for name, value in zip(pre.get_feature_names_out(), coef):
                coefficients.append({**{k: row[k] for k in ("seed", "route", "target", "fold")},
                                     "feature": name, "coefficient_original_target_units": float(value * model.target_scale_)})
        rows.append(row)
    table = pd.DataFrame(coefficients)
    stability = []
    for (seed, route, target, feature), values in table.groupby(["seed", "route", "target", "feature"]):
        coeff = values.coefficient_original_target_units.to_numpy()
        stability.append({"seed": int(seed), "route": route, "target": target, "feature": feature,
                          "positive_folds": int((coeff > 1e-10).sum()), "negative_folds": int((coeff < -1e-10).sum()),
                          "zero_folds": int((np.abs(coeff) <= 1e-10).sum()), "std": float(np.std(coeff)),
                          "mean": float(np.mean(coeff)), "sign_flip": bool((coeff > 1e-10).any() and (coeff < -1e-10).any())})
    return {"fold_models": rows, "coefficients": coefficients, "coefficient_stability": stability,
            "regressor_fits": 0, "models_read": len(rows),
            "coefficient_comparability": "L1/L2 standardized input units, restored target units; S1 spline knots differ per fold"}
