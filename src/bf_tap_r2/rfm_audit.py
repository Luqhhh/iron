"""Independent RFM equations and partition audit. No fitting or solving.

The oracle uses explicit blockwise differences in the original metric, not
the production square-root coordinate implementation. Official phase/source
and reference manifests still need separate binding by the phase runner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import FEATURES, TARGETS
from .rfm_model import SavedRFM, array_digest
from .rfm_protocol import canonical, file_hash
from .v3_4_bags import group_safe_inner_folds

TOLERANCE = 1e-8


def oracle_kernel_and_gradient(z, c, centers, center_c, metric, bandwidth, alpha=None):
    """Small query blocks cap the independent oracle's 3D temporary."""
    kernel = np.empty((len(z), len(centers)), dtype=np.float64)
    gradient = None if alpha is None else np.empty_like(z, dtype=np.float64)
    for start in range(0, len(z), 32):
        part = slice(start, start + 32)
        delta = z[part, None, :] - centers[None, :, :]
        q = np.einsum("bni,ij,bnj->bn", delta, metric, delta, optimize=False)
        q += np.square(c[part, None, :] - center_c[None, :, :]).sum(axis=2)
        radius = np.sqrt(np.maximum(q, 0))
        values = np.exp(-radius / bandwidth)
        kernel[part] = values
        if gradient is not None:
            coefficient = np.zeros_like(values)
            np.divide(-values * alpha[None, :], bandwidth * radius,
                      out=coefficient, where=radius > 1e-12)
            gradient[part] = np.einsum("bn,bni,ij->bj", coefficient, delta, metric, optimize=False)
    return kernel, gradient


def _assert_close(a, b, message, tolerance=TOLERANCE):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(message + ": invalid shape/value")
    difference = float(np.max(np.abs(a-b))) if a.size else 0.
    if difference > tolerance:
        raise ValueError(f"{message}: maximum difference {difference}")
    return difference


def _coordinates(frame, model):
    raw = frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    p = model.preprocessor
    z = ((raw - p.means_) / p.stds_).astype(np.float32).astype(np.float64)
    c = np.zeros((len(frame), p.n_spout_categories_), dtype=np.float64)
    codes = [p.spout_to_index_.get(int(v), 0) for v in frame.spout_no]
    c[np.arange(len(frame)), codes] = 1
    return z, c


def _verify_training(model, frame, y):
    raw = frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    if model.fit_ids != frame.sample_id.tolist():
        raise ValueError("saved training row order/membership mismatch")
    p = model.preprocessor
    _assert_close(p.means_, raw.mean(0), "training-only feature means", 1e-12)
    _assert_close(p.stds_, raw.std(0), "training-only feature stds", 1e-12)
    vocab = {v: i for i, v in enumerate(sorted({int(v) for v in frame.spout_no}), 1)}
    if p.spout_to_index_ != vocab or p.n_spout_categories_ != len(vocab) + 1:
        raise ValueError("training-only vocabulary mismatch")
    _assert_close(model.target_mean, y.mean(), "training-only target mean", 1e-12)
    _assert_close(model.target_std, y.std(), "training-only target std", 1e-12)
    z, c = _coordinates(frame, model)
    _assert_close(z, model.centers, "training center identity", 0.)
    _assert_close(c, model.center_categories, "category center identity", 0.)
    if (model.training_identity["features_sha256"] != array_digest(raw)
            or model.training_identity["target_sha256"] != array_digest(y)
            or model.training_identity["ids_sha256"] != hashlib.sha256(canonical(frame.sample_id.tolist())).hexdigest()):
        raise ValueError("original training identity mismatch")
    # Independent condensed distance construction. Only positive unordered
    # training pairs contribute; labels and query rows never set bandwidth.
    distances = []
    for i in range(len(z) - 1):
        q = np.square(z[i+1:] - z[i]).sum(1) + np.square(c[i+1:] - c[i]).sum(1)
        r = np.sqrt(q)
        distances.append(r[r > 0])
    positive = np.concatenate(distances)
    if not len(positive):
        raise ValueError("no positive training distances")
    _assert_close(model.bandwidth, np.median(positive), "training-only bandwidth", 1e-12)
    return z, c


def audit_path(directory, frame, y, arm, expected_models):
    directory = Path(directory)
    states = list(range(len(expected_models)))
    if not states or [a["state"] for a in expected_models] != states:
        raise ValueError("missing or unordered expected RFM state")
    if {p.name for p in directory.iterdir()} != {f"state-{s}" for s in states}:
        raise ValueError("missing or unexpected path artifact")
    models, max_difference = [], 0.
    expected_metric = np.eye(len(FEATURES))
    y = np.asarray(y, dtype=np.float64)
    for state, artifact in enumerate(expected_models):
        envelope = json.loads((directory / f"state-{state}" / "complete.json").read_text())
        if {k: v for k, v in artifact.items() if k not in ("state", "complete_sha256")} != envelope:
            raise ValueError("saved artifact descriptor mismatch")
        model = SavedRFM.load(directory / f"state-{state}", expected_sha256=artifact["complete_sha256"])
        if model.state != state or model.arm != arm:
            raise ValueError("model path arm/state mismatch")
        z, c = _verify_training(model, frame, y)
        max_difference = max(max_difference, _assert_close(model.metric, expected_metric, "AGOP metric recurrence"))
        kernel, gradient = oracle_kernel_and_gradient(z, c, z, c, model.metric, model.bandwidth, model.alpha)
        normalized_target = (y-model.target_mean)/model.target_std
        residual = _assert_close(kernel @ model.alpha + .01*model.alpha, normalized_target, "independent solve equation")
        # Inference arithmetic is checked independently from training residual.
        predicted = kernel @ model.alpha * model.target_std + model.target_mean
        max_difference = max(max_difference, residual,
            _assert_close(predicted, model.predict(frame), "independent cold prediction"),
            _assert_close(predicted[::-1], model.predict(frame.iloc[::-1]), "reverse cold prediction"))
        chunks = np.concatenate([model.predict(frame.iloc[i:i+7]) for i in range(0, len(frame), 7)])
        max_difference = max(max_difference, _assert_close(predicted, chunks, "chunked cold prediction"))
        if state < len(states) - 1:
            agop = sum(np.outer(g, g) for g in gradient) / len(gradient)
            agop = (agop + agop.T) / 2
            vals, vecs = np.linalg.eigh(agop)
            if vals.min() < -1e-10:
                raise ValueError("independent AGOP is not PSD")
            vals = np.maximum(vals, 0)
            if not np.isfinite(vals).all() or vals.sum() <= 0:
                raise ValueError("independent AGOP has no positive trace")
            expected_metric = .99 * ((vecs * (len(FEATURES)*vals/vals.sum())) @ vecs.T) + .01*np.eye(len(FEATURES))
        models.append(model)
    return models, max_difference


def audit_partition(directory, train, target, *, expected_selection_sha256):
    """Recover all states and selector decisions against supplied original rows.

    The caller must obtain expected_selection_sha256 from its frozen phase
    manifest, not compute it from an untrusted artifact immediately before
    this call. The same requirement applies to original data membership.
    """
    directory = Path(directory)
    if target not in TARGETS:
        raise ValueError("unknown target")
    if file_hash(directory / "selection.json") != expected_selection_sha256:
        raise ValueError("selection identity mismatch")
    if {p.name for p in directory.iterdir()} != {"inner", "refit", "selection.json"}:
        raise ValueError("unexpected partition artifact")
    selection = json.loads((directory / "selection.json").read_text())
    if selection["target"] != target or selection["inner_seed"] != 42 or selection["inner_validation_fold"] != 0:
        raise ValueError("selector definition mismatch")
    folds = group_safe_inner_folds(train.drop(columns=list(TARGETS), errors="ignore"), seed=42)
    mask = folds["fold"] != 0
    if (selection["inner_fold_hash"] != folds["inner_fold_hash"]
            or selection["group_hash"] != folds["group_hash"]
            or selection["training_ids"] != train.sample_id.tolist()
            or selection["inner_fit_ids"] != train.loc[mask, "sample_id"].tolist()
            or selection["calibration_ids"] != train.loc[~mask, "sample_id"].tolist()):
        raise ValueError("selector row/partition identity mismatch")
    y = train[target].to_numpy(dtype=np.float64)
    artifacts = selection["artifacts"]
    expected_inner = 4 if selection["arm"] == "FULL_RFM" else 1
    if len(artifacts["inner"]) != expected_inner:
        raise ValueError("incomplete selector state coverage")
    inner, max_inner = audit_path(directory / "inner", train.loc[mask], y[mask], selection["arm"], artifacts["inner"])
    errors = []
    maximum = max_inner
    query = train.loc[~mask].drop(columns=list(TARGETS), errors="ignore")
    for model in inner:
        z, c = _coordinates(query, model)
        kernel, _ = oracle_kernel_and_gradient(z, c, model.centers, model.center_categories, model.metric, model.bandwidth)
        independent = kernel @ model.alpha * model.target_std + model.target_mean
        cold = model.predict(query)
        maximum = max(maximum, _assert_close(independent, cold, "independent calibration prediction"))
        errors.append(float(np.mean(np.abs(cold-y[~mask]))))
    _assert_close(errors, selection["calibration_mae"], "saved selector MAEs", 0.)
    selected = int(np.argmin(errors))
    if selection["selected_state"] != selected or len(artifacts["refit"]) != selected + 1:
        raise ValueError("selected state/fresh refit path mismatch")
    refit, max_refit = audit_path(directory / "refit", train, y, selection["arm"], artifacts["refit"])
    for stage, path in (("inner", inner), ("outer", refit)):
        model = path[0]
        if selection[f"{stage}_preprocessor"] != model.preprocessor.metadata():
            raise ValueError("selection preprocessing record mismatch")
        if selection[f"{stage}_training_identity"] != model.training_identity:
            raise ValueError("selection training record mismatch")
        for field in ("target_mean", "target_std", "bandwidth"):
            _assert_close(selection[f"{stage}_{field}"], getattr(model, field), "selection scale record", 0.)
    return {"status": "passed", "target": target, "arm": selection["arm"],
            "models_checked": len(inner) + len(refit), "selected_state": selected,
            "maximum_difference": max(maximum, max_refit),
            "selection_sha256": expected_selection_sha256,
            "new_solver_calls": 0, "new_optimizer_calls": 0,
            "scope": "one supplied training partition; not full phase/reference/gate admission"}
