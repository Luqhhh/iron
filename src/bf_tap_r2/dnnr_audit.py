"""Independent saved-array inference and training-witness arithmetic audit.

No Regressor.fit, metric_epoch, Encoder.fit, neighbor helper or term function is
called. Local equation/gradient recomputations are mathematical audit work,
reported explicitly; they do not create/retrain estimators or select endpoints.
"""
import random
import json
from pathlib import Path

import numpy as np

from .data import FEATURES, TARGETS
from .dnnr_model import ARMS, Regressor, validate_frame
from .v3_4_bags import group_safe_inner_folds
from .v7_periodic import digest, file_hash


def _encode(frame, preprocessing):
    validate_frame(frame)
    num = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    num = (num-np.array(preprocessing["means"]))/np.array(preprocessing["stds"])
    vocabulary = preprocessing["spout_categories"]
    cat = np.zeros((len(frame), len(vocabulary)+1), dtype=np.float64)
    for i, v in enumerate(frame.spout_no.tolist()):
        cat[i, vocabulary.index(int(v))+1 if int(v) in vocabulary else 0] = 1.
    return np.concatenate((num, cat), axis=1)


def _neighbors(points, query, ids, count, *, anchor=None, include=False):
    distance = ((points-query)**2).sum(axis=1)
    rows = sorted((i for i in range(len(points)) if i != anchor),
                  key=lambda i: (float(distance[i]), ids[i]))
    if include:
        rows.insert(0, anchor)
    return np.asarray(rows[:count], dtype=np.int64)


def independent_predict(model, query):
    if set(model.ids_) & set(query.sample_id.astype(str)):
        raise ValueError("Audit query overlaps fitting identities")
    x = _encode(query, model.saved_metadata_["encoder"])*model.arrays_["scale"]
    train = model.arrays_["x"]*model.arrays_["scale"]
    answers = []
    for row in x:
        ids = _neighbors(train, row, model.ids_, model.settings.neighbors)
        votes = []
        for i in ids:
            value = model.arrays_["y"][i]
            if model.arm != "KNN_FIXED":
                value += float(np.dot(model.arrays_["derivatives"][i], row-train[i]))
            votes.append(value)
        answers.append(sum(votes)/len(votes))
    return np.array(answers)


def verify_saved(path, sha256, fitting, y, query, expected_prediction, *, tolerance=1e-8):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive finite audit tolerance required")
    model = Regressor.load(path, sha256)
    arrays, prep = model.arrays_, model.saved_metadata_["encoder"]
    validate_frame(fitting)
    if fitting.sample_id.astype(str).tolist() != model.ids_:
        raise ValueError("Saved/fresh fitting identity order mismatch")
    raw = fitting[list(FEATURES)].to_numpy(dtype=np.float64)
    std = raw.std(axis=0, ddof=0); std[std == 0] = 1.
    if (not np.array_equal(raw.mean(0), prep["means"])
            or not np.array_equal(std, prep["stds"])
            or sorted(set(int(v) for v in fitting.spout_no)) != prep["spout_categories"]):
        raise ValueError("Saved preprocessing differs from fitting rows alone")
    x = _encode(fitting, prep)
    if not np.array_equal(x, arrays["x"]) or not np.array_equal(y, arrays["y"]):
        raise ValueError("Saved training feature/label contents differ")
    derivative_difference, metric_difference = 0., 0.
    derivative_solves, metric_solves, updates = 0, 0, 0
    if model.metric_epochs_:
        n = len(fitting)
        order = list(range(n)); random.Random(model.settings.random_seed).shuffle(order)
        if not np.array_equal(order, arrays["metric_order"]):
            raise ValueError("Metric update order differs")
        scale = np.ones(x.shape[1])
        count = arrays["metric_graph"].shape[1]
        for step, i in enumerate(order):
            indices = _neighbors(x, x[i], model.ids_, count, anchor=i)
            if not np.array_equal(indices, arrays["metric_graph"][i]):
                raise ValueError("Metric graph differs from initial training-only coordinates")
            delta = x[indices]-x[i]
            normal = delta.T.dot(delta)
            coef = np.linalg.pinv(normal).dot(delta.T.dot(np.asarray(y)[indices]-y[i]))
            errors = abs(np.asarray(y)[indices]-y[i]-delta.dot(coef))
            # Reconstruct the cost and analytical derivative from saved trajectory.
            raw_h = np.sqrt(((delta*scale)**2).sum(axis=1))
            h = np.maximum(raw_h, model.settings.epsilon)
            a, b = errors-errors.mean(), h-h.mean()
            if np.allclose(a, 0) or np.allclose(b, 0):
                cost, gradient = 0., np.zeros(len(scale))
            else:
                na, nb = np.sqrt(a.dot(a)), np.sqrt(b.dot(b))
                cosine = a.dot(b)/(na*nb)
                derivative = -a/(na*nb)+cosine*b/(nb**2)
                derivative -= derivative.mean()
                derivative *= raw_h > model.settings.epsilon
                gradient = ((derivative/h)[:, None]*delta**2*scale).sum(axis=0)
                cost = -cosine
            differences = [np.max(abs(errors-arrays["metric_errors"][i])),
                np.max(abs(scale-arrays["metric_scales_before"][step])),
                np.max(abs(gradient-arrays["metric_gradients"][step])),
                abs(cost-arrays["metric_costs"][step])]
            metric_difference = max(metric_difference, *map(float, differences))
            scale = scale-model.settings.learning_rate*gradient
            metric_solves += 1; updates += 1
        metric_difference = max(metric_difference, float(np.max(abs(scale-arrays["scale"]))))
    scaled = x*arrays["scale"]
    if model.arm != "KNN_FIXED":
        count = arrays["derivative_graph"].shape[1]
        for i in range(len(fitting)):
            indices = _neighbors(scaled, scaled[i], model.ids_, count, anchor=i, include=True)
            if not np.array_equal(indices, arrays["derivative_graph"][i]):
                raise ValueError("Derivative graph differs from final scaled training coordinates")
            coef = np.linalg.lstsq(scaled[indices]-scaled[i], np.asarray(y)[indices]-y[i], rcond=None)[0]
            derivative_difference = max(derivative_difference, float(np.max(abs(coef-arrays["derivatives"][i]))))
            derivative_solves += 1
    cold = model.predict(query)
    independent = independent_predict(model, query)
    reverse = model.predict(query.iloc[::-1].reset_index(drop=True))[::-1]
    pieces = [model.predict(query.iloc[start:start+7].reset_index(drop=True)) for start in range(0, len(query), 7)]
    chunked = np.concatenate(pieces)
    expected = np.asarray(expected_prediction, dtype=np.float64)
    if expected.shape != cold.shape or not np.isfinite(expected).all():
        raise ValueError("Invalid externally supplied prediction witness")
    inference_difference = max(float(np.max(abs(cold-value))) for value in (independent, reverse, chunked, expected))
    if max(derivative_difference, metric_difference, inference_difference) > tolerance:
        raise ValueError("Saved DNNR witness/inference arithmetic differs beyond tolerance")
    return {"status": "passed", "sha256": sha256, "arm": model.arm,
        "metric_epochs": model.metric_epochs_, "fitting_rows": len(fitting), "query_rows": len(query),
        "cold_order_chunk_independent_max_difference": inference_difference,
        "derivative_witness_max_difference": derivative_difference,
        "metric_witness_max_difference": metric_difference,
        "numerical_audit_derivative_solutions": derivative_solves,
        "numerical_audit_metric_solutions": metric_solves,
        "numerical_audit_metric_updates_reconstructed": updates,
        "new_estimator_fits": 0, "new_metric_epoch_runs": 0,
        "query_ids_digest": digest(query.sample_id.astype(str).tolist())}


def verify_pair(directory, receipt_sha256, training, y, query, expected_predictions,
                *, tolerance=1e-8):
    """Verify all six states and endpoint selection against original training rows."""
    directory = Path(directory)
    path = directory / "receipt.json"
    if not receipt_sha256 or file_hash(path) != receipt_sha256:
        raise ValueError("Externally anchored pair receipt hash required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["format"] != "dnnr-matched-pair-v1":
        raise ValueError("Saved pair format mismatch")
    receipt, hashes = payload["receipt"], payload["model_hashes"]
    validate_frame(training); validate_frame(query)
    y = np.asarray(y, dtype=np.float64)
    if (y.shape != (len(training),) or not np.isfinite(y).all()
            or receipt["target"] not in TARGETS
            or set(hashes) != {f"{role}-{arm}.npz" for role in ("inner", "outer") for arm in ARMS}
            or set(expected_predictions) != set(ARMS)):
        raise ValueError("Pair target, array or model identity schema differs")
    if receipt["inner_seed"] != 42 or receipt["held_fold"] != 0:
        raise ValueError("Pair changed the fixed calibration split rule")
    assignment = group_safe_inner_folds(training, n_splits=5, seed=42)
    folds = assignment["fold"]; mask = folds != 0
    fitting, calibration = training.loc[mask].reset_index(drop=True), training.loc[~mask].reset_index(drop=True)
    checks = {"fitting_ids": fitting.sample_id.astype(str).tolist(),
        "calibration_ids": calibration.sample_id.astype(str).tolist(),
        "outer_ids": training.sample_id.astype(str).tolist(), "inner_folds": folds.tolist(),
        "inner_group_hash": assignment["group_hash"], "inner_fold_hash": assignment["inner_fold_hash"],
        "selection": "lower_raw_calibration_MAE_initial_wins_ties",
        "outer_refit": "fresh_unit_scale_relearn_selected_epoch_count", "outer_query_labels": "absent"}
    for key, value in checks.items():
        if receipt[key] != value:
            raise ValueError("Pair partition/selection contract differs")
    if set(receipt["calibration_predictions"]) != set(ARMS) or set(receipt["calibration_maes"]) != set(ARMS):
        raise ValueError("Pair calibration arms differ")
    models, reports, maes, settings = {}, {}, {}, None
    for arm in ARMS:
        name = f"inner-{arm}.npz"
        model = Regressor.load(directory/name, hashes[name])
        if model.arm != arm or model.metric_epochs_ != int(arm == "DNNR_LEARNED"):
            raise ValueError("Inner saved arm/epoch rule differs")
        if settings is not None and model.settings != settings:
            raise ValueError("Matched models have different numeric settings")
        settings = model.settings
        expected = receipt["calibration_predictions"][arm]
        if digest(expected) != receipt["calibration_prediction_digests"][arm]:
            raise ValueError("Pair calibration prediction digest differs")
        reports[name] = verify_saved(directory/name, hashes[name], fitting, y[mask], calibration,
                                    expected, tolerance=tolerance)
        prediction = independent_predict(model, calibration)
        maes[arm] = float(np.abs(prediction-y[~mask]).mean())
        if abs(maes[arm]-receipt["calibration_maes"][arm]) > tolerance:
            raise ValueError("Pair calibration MAE arithmetic differs")
        models[name] = model
    # Use the reconstructed original reduction order for the strict tie rule.
    # Independent inference is checked above; warm-equivalent NumPy reductions
    # avoid changing exact ties by a last-bit sum-order difference.
    original_maes = {arm: float(np.abs(models[f"inner-{arm}.npz"].predict(calibration)-y[~mask]).mean()) for arm in ARMS}
    selected = int(original_maes["DNNR_LEARNED"] < original_maes["DNNR_FIXED"])
    if type(receipt["selected_metric_epochs"]) is not int or receipt["selected_metric_epochs"] != selected:
        raise ValueError("Pair selected metric epoch differs from the fixed tie rule")
    for arm in ARMS:
        name = f"outer-{arm}.npz"
        model = Regressor.load(directory/name, hashes[name])
        if (model.arm != arm or model.settings != settings
                or model.metric_epochs_ != (selected if arm == "DNNR_LEARNED" else 0)):
            raise ValueError("Outer saved arm/settings/selected-epoch rule differs")
        reports[name] = verify_saved(directory/name, hashes[name], training, y, query,
                                    expected_predictions[arm], tolerance=tolerance)
        models[name] = model
    counts = {"estimator_runs": 6, "derivative_bank_runs": 4,
        "derivative_local_solutions": 2*len(fitting)+2*len(training),
        "metric_epoch_runs": 1+selected, "metric_updates": len(fitting)+selected*len(training),
        "metric_local_solutions": len(fitting)+selected*len(training)}
    for key, value in counts.items():
        if receipt[key] != value:
            raise ValueError("Pair computational accounting differs")
    elapsed = sum(m.elapsed_seconds_ for m in models.values())
    if abs(receipt["total_fit_seconds"]-elapsed) > tolerance:
        raise ValueError("Pair fit duration accounting differs")
    return {"status": "passed", "receipt_sha256": receipt_sha256,
        "selected_metric_epochs": selected, "calibration_maes": maes,
        "saved_models": 6, "counts": counts, "models": reports,
        "new_estimator_fits": 0, "new_metric_epoch_runs": 0}
