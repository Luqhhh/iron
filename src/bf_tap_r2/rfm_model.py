"""Partition-local RFM fitting and hash-bound, inference-only model loading.

There is no dataset loader, training CLI, automatic retry or release function.
Every procedure, solve and metric update requires an external reservation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .rfm_kernel import laplace_kernel, numeric_gradient, solve_kernel, training_bandwidth, update_metric
from .rfm_protocol import ARMS, canonical, file_hash, write_new
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor


def array_digest(value):
    value = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(canonical(list(value.shape)) + value.tobytes()).hexdigest()


def source_identity():
    names = ("rfm_model.py", "rfm_kernel.py", "rfm_protocol.py", "data.py",
             "splits.py", "v3_4_bags.py", "v3_6_networks.py")
    return {name: file_hash(Path(__file__).parent / name) for name in names}


def runtime_identity():
    return {"python": platform.python_version(), **{
        p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn")}}


def feature_frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError("a DataFrame with unique column names is required")
    required = [*FEATURES, "spout_no"]
    if not set(required) <= set(frame.columns):
        raise ValueError("missing prediction features")
    view = frame.loc[:, required].copy()
    values = view.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite prediction features")
    spout = values[:, -1]
    if not np.equal(spout, np.floor(spout)).all():
        raise ValueError("spout category must be integer valued")
    return view


def fit_ids(frame):
    if "sample_id" not in frame:
        raise ValueError("training sample IDs are required")
    ids = frame["sample_id"]
    if ids.isna().any() or ids.duplicated().any() or not all(isinstance(v, str) for v in ids):
        raise ValueError("unique, non-null string training IDs required")
    return ids.tolist()


def restore_preprocessor(metadata):
    p = NumericPreprocessor(structure="raw_tabm")
    if (metadata.get("structure") != "raw_tabm" or metadata.get("feature_names") != list(FEATURES)
            or metadata.get("uses_ple") is not False or metadata.get("ple_bins_lengths") is not None
            or metadata.get("n_bins") != 16 or metadata.get("d_embedding") != 8):
        raise ValueError("saved preprocessing definition mismatch")
    p.means_ = np.asarray(metadata["means"], dtype=np.float64)
    p.stds_ = np.asarray(metadata["stds"], dtype=np.float64)
    if (p.means_.shape != (len(FEATURES),) or p.stds_.shape != (len(FEATURES),)
            or not np.isfinite(p.means_).all() or not np.isfinite(p.stds_).all()
            or (p.stds_ <= 0).any()):
        raise ValueError("invalid saved feature scales")
    p.spout_to_index_ = {int(k): v for k, v in metadata["spout_vocabulary"].items()}
    expected = {v: i for i, v in enumerate(sorted(p.spout_to_index_), 1)}
    if not expected or p.spout_to_index_ != expected:
        raise ValueError("invalid saved spout vocabulary")
    p.n_spout_categories_ = metadata["n_spout_categories"]
    if p.n_spout_categories_ != len(expected) + 1:
        raise ValueError("missing reserved unknown category")
    return p


@dataclass
class SavedRFM:
    arm: str
    state: int
    preprocessor: NumericPreprocessor
    centers: np.ndarray
    center_categories: np.ndarray
    metric: np.ndarray
    alpha: np.ndarray
    bandwidth: float
    target_mean: float
    target_std: float
    fit_ids: list[str]
    training_identity: dict
    solver_residual: float

    def inputs(self, frame):
        numeric, category = self.preprocessor.transform_mlp(feature_frame(frame))
        numeric = numeric.astype(np.float64)
        category = category.astype(np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("nonfinite transformed numeric input")
        return numeric, category

    def validate(self):
        if self.arm not in ARMS or type(self.state) is not int or not 0 <= self.state <= 3:
            raise ValueError("invalid RFM arm/state")
        if self.arm == "FIXED_KRR" and self.state != 0:
            raise ValueError("fixed kernel cannot update its metric")
        n, d = len(self.fit_ids), len(FEATURES)
        if (not n or len(set(self.fit_ids)) != n or not all(isinstance(i, str) for i in self.fit_ids)
                or self.centers.shape != (n, d)
                or self.center_categories.shape != (n, self.preprocessor.n_spout_categories_)
                or self.metric.shape != (d, d) or self.alpha.shape != (n,)):
            raise ValueError("invalid saved model shape/IDs")
        for a in (self.centers, self.center_categories, self.metric, self.alpha):
            if a.dtype != np.float64 or not np.isfinite(a).all():
                raise ValueError("model arrays must be finite float64")
        if (not np.isin(self.center_categories, (0, 1)).all()
                or not np.equal(self.center_categories.sum(1), 1).all()
                or np.any(self.center_categories[:, 0] != 0)):
            raise ValueError("invalid training categorical centers")
        scales = (self.bandwidth, self.target_mean, self.target_std, self.solver_residual)
        if (not np.isfinite(scales).all() or self.bandwidth <= 0 or self.target_std <= 0
                or not 0 <= self.solver_residual < 1e-8):
            raise ValueError("invalid saved scales/solver residual")
        if self.state == 0:
            if not np.array_equal(self.metric, np.eye(d)):
                raise ValueError("state zero must use the identity metric")
        elif (not np.allclose(self.metric, self.metric.T, rtol=0, atol=1e-12)
              or abs(np.trace(self.metric) - d) > 1e-10
              or np.linalg.eigvalsh(self.metric).min() < .01 - 1e-10):
            raise ValueError("invalid updated metric")
        for name, value in (("centers", self.centers), ("categories", self.center_categories)):
            if self.training_identity[name + "_sha256"] != array_digest(value):
                raise ValueError("saved center identity mismatch")
        if self.training_identity["ids_sha256"] != hashlib.sha256(canonical(self.fit_ids)).hexdigest():
            raise ValueError("saved row identity mismatch")

    def predict(self, frame):
        z, c = self.inputs(frame)
        prediction = (laplace_kernel(z, c, self.centers, self.center_categories,
                                     self.metric, self.bandwidth) @ self.alpha)
        prediction = prediction * self.target_std + self.target_mean
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite RFM prediction")
        return prediction

    def save(self, directory):
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {"centers": self.centers, "center_categories": self.center_categories,
                  "metric": self.metric, "alpha": self.alpha}
        with (directory / "arrays.npz").open("xb") as handle:
            np.savez(handle, **arrays)
        metadata = {"format": "rfm-model-v1", "arm": self.arm, "state": self.state,
                    "preprocessor": self.preprocessor.metadata(), "bandwidth": self.bandwidth,
                    "target_mean": self.target_mean, "target_std": self.target_std,
                    "fit_ids": self.fit_ids, "training_identity": self.training_identity,
                    "solver_residual": self.solver_residual,
                    "source_identity": source_identity(), "runtime_identity": runtime_identity()}
        write_new(directory / "metadata.json", metadata)
        completion = {"format": "rfm-model-v1", "hashes": {
            p: file_hash(directory / p) for p in ("arrays.npz", "metadata.json")}}
        write_new(directory / "complete.json", completion)
        return {"complete_sha256": file_hash(directory / "complete.json"), **completion}

    @classmethod
    def load(cls, directory, *, expected_sha256):
        directory = Path(directory)
        if file_hash(directory / "complete.json") != expected_sha256:
            raise ValueError("model completion identity mismatch")
        completion = json.loads((directory / "complete.json").read_text())
        if (completion.get("format") != "rfm-model-v1"
                or set(completion.get("hashes", {})) != {"arrays.npz", "metadata.json"}):
            raise ValueError("unexpected model completion format")
        for name, expected in completion["hashes"].items():
            if file_hash(directory / name) != expected:
                raise ValueError("saved model file hash mismatch")
        metadata = json.loads((directory / "metadata.json").read_text())
        if (metadata["format"] != "rfm-model-v1" or metadata["source_identity"] != source_identity()
                or metadata["runtime_identity"] != runtime_identity()):
            raise ValueError("saved model source/runtime identity mismatch")
        with np.load(directory / "arrays.npz", allow_pickle=False) as saved:
            if set(saved.files) != {"centers", "center_categories", "metric", "alpha"}:
                raise ValueError("unexpected saved array names")
            arrays = {name: saved[name] for name in saved.files}
        model = cls(arm=metadata["arm"], state=metadata["state"], **arrays,
                    preprocessor=restore_preprocessor(metadata["preprocessor"]),
                    bandwidth=metadata["bandwidth"], target_mean=metadata["target_mean"],
                    target_std=metadata["target_std"], fit_ids=metadata["fit_ids"],
                    training_identity=metadata["training_identity"], solver_residual=metadata["solver_residual"])
        model.validate()
        return model


class RFMRegressor:
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError("unknown RFM arm")
        self.arm = arm

    def fit_path(self, frame, y, updates, reserve, *, directory=None):
        if type(updates) is not int or not 0 <= updates <= 3 or (self.arm == "FIXED_KRR" and updates):
            raise ValueError("invalid frozen update count")
        with reserve("procedure", None, {"arm": self.arm, "updates": updates}) as procedure:
            if directory is not None:
                directory = Path(directory)
                directory.mkdir(parents=True, exist_ok=False)
            ids = fit_ids(frame)
            features = feature_frame(frame)
            y = np.asarray(y, dtype=np.float64)
            if y.shape != (len(frame),) or not np.isfinite(y).all() or not len(y):
                raise ValueError("invalid training target")
            mean, std = float(y.mean()), float(y.std())
            if not np.isfinite([mean, std]).all() or std <= 0:
                raise ValueError("constant or invalid target scale")
            target = (y - mean) / std
            p = NumericPreprocessor(structure="raw_tabm").fit(features)
            z, c = (a.astype(np.float64) for a in p.transform_mlp(features))
            identity = {"ids_sha256": hashlib.sha256(canonical(ids)).hexdigest(),
                        "centers_sha256": array_digest(z), "categories_sha256": array_digest(c),
                        "target_sha256": array_digest(y),
                        "features_sha256": array_digest(features.loc[:, FEATURES].to_numpy())}
            bandwidth = training_bandwidth(z, c)
            metric, models = np.eye(len(FEATURES)), []
            artifacts = []
            for state in range(updates + 1):
                with reserve("solve", state, {"training_identity": identity, "arm": self.arm,
                             "metric_sha256": array_digest(metric), "bandwidth": bandwidth}) as solve:
                    kernel = laplace_kernel(z, c, z, c, metric, bandwidth)
                    alpha = solve_kernel(kernel, target)
                    residual = float(np.max(np.abs(kernel @ alpha + .01 * alpha - target)))
                    if not residual < 1e-8:
                        raise ValueError("kernel solve residual exceeds 1e-8")
                    model = SavedRFM(self.arm, state, p, z, c, metric, alpha, bandwidth,
                                     mean, std, ids, identity, residual)
                    model.validate()
                    if directory is not None:
                        artifact = model.save(directory / f"state-{state}")
                        artifacts.append({"state": state, **artifact})
                        solve["saved_model"] = artifact
                    models.append(model)
                    solve["residual"] = residual
                    solve["alpha_sha256"] = array_digest(alpha)
                if state < updates:
                    with reserve("update", state + 1, {"from_state": state}) as update:
                        gradients = numeric_gradient(z, c, z, c, metric, bandwidth, alpha)
                        metric = update_metric(gradients)
                        update["metric_sha256"] = array_digest(metric)
                        update["gradient_sha256"] = array_digest(gradients)
            procedure.update(training_identity=identity, states=len(models), artifacts=artifacts)
            self.last_artifacts_ = artifacts
            return models


def fit_partition(train, target, arm, reserve, *, directory=None):
    """Select on inner fold zero, then fit a fresh identity-metric outer path.

    Only ``train`` is accepted, so outer query rows cannot be passed as a
    separate validation argument. The future phase runner must prove its
    membership against the frozen outer split before calling this function.
    """
    if target not in TARGETS or target not in train:
        raise ValueError("invalid target")
    model = RFMRegressor(arm)
    ids = fit_ids(train)
    features = feature_frame(train)
    splitter_frame = features.assign(sample_id=ids)
    folds = group_safe_inner_folds(splitter_frame, seed=42)
    mask = folds["fold"] != 0
    if not mask.any() or mask.all():
        raise ValueError("empty inner training or calibration partition")
    y = train[target].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("nonfinite training target")
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
    def scoped(stage):
        return lambda kind, state, payload: reserve(kind, (stage, state), payload)
    inner = model.fit_path(splitter_frame.loc[mask].reset_index(drop=True), y[mask],
                           3 if arm == "FULL_RFM" else 0, scoped("inner"),
                           directory=None if directory is None else directory / "inner")
    inner_artifacts = model.last_artifacts_
    calibration = splitter_frame.loc[~mask].reset_index(drop=True)
    errors = [float(np.mean(np.abs(m.predict(calibration) - y[~mask]))) for m in inner]
    if not np.isfinite(errors).all():
        raise ValueError("nonfinite calibration scores")
    selected = int(np.argmin(errors))  # exact ties choose the first state
    refit = model.fit_path(splitter_frame, y, selected, scoped("refit"),
                           directory=None if directory is None else directory / "refit")
    metadata = {"target": target, "arm": arm, "inner_seed": 42, "inner_validation_fold": 0,
                "inner_fold_hash": folds["inner_fold_hash"], "group_hash": folds["group_hash"],
                "training_ids": ids, "inner_fit_ids": inner[0].fit_ids,
                "calibration_ids": calibration.sample_id.tolist(),
                "calibration_mae": errors, "selected_state": selected,
                "inner_training_identity": inner[0].training_identity,
                "outer_training_identity": refit[0].training_identity,
                "inner_preprocessor": inner[0].preprocessor.metadata(),
                "outer_preprocessor": refit[0].preprocessor.metadata(),
                "inner_target_mean": inner[0].target_mean, "inner_target_std": inner[0].target_std,
                "outer_target_mean": refit[0].target_mean, "outer_target_std": refit[0].target_std,
                "inner_bandwidth": inner[0].bandwidth, "outer_bandwidth": refit[0].bandwidth}
    if directory is not None:
        metadata["artifacts"] = {"inner": inner_artifacts, "refit": model.last_artifacts_}
        write_new(directory / "selection.json", metadata)
    return refit[-1], metadata
