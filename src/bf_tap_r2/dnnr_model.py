"""Training-only first-order DNNR and its two matched controls.

Code-inspired variant of Nader et al. (ICML 2022), not an exact package
reproduction. The metric graph is frozen at the epoch's initial coordinates;
the current scale changes distances inside that fixed graph only. See DESIGN.
No data loader, incumbent residual, target clipping or release logic lives here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path
import json
import random
import time

import numpy as np

from .data import FEATURES, TARGETS
from .dnnr_terms import (first_order_coefficients, local_linear_errors,
                         scaling_cost_gradient)
from .v7_periodic import digest, file_hash

ARMS = ("KNN_FIXED", "DNNR_FIXED", "DNNR_LEARNED")


@dataclass(frozen=True)
class Settings:
    neighbors: int = 3
    derivative_multiple: int = 3
    metric_multiple: int = 8
    learning_rate: float = .01
    epsilon: float = 1e-6
    random_seed: int = 42

    def __post_init__(self):
        for name in ("neighbors", "derivative_multiple", "metric_multiple"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer neighborhood sizes required")
        if (type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32
                or not np.isfinite([self.learning_rate, self.epsilon]).all()
                or self.learning_rate <= 0 or self.epsilon <= 0):
            raise ValueError("Invalid metric update constants")


def validate_frame(frame):
    if (set(frame.columns) != {"sample_id", "spout_no", *FEATURES}
            or len(frame.columns) != len(set(frame.columns)) or not len(frame)):
        raise ValueError("Nonempty, unlabeled round-two feature schema required")
    if (frame.sample_id.isna().any()
            or frame.sample_id.astype(str).duplicated().any()
            or (frame.sample_id.astype(str).str.len() == 0).any()):
        raise ValueError("Unique nonempty sample identities required")
    x = frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    cat = frame.spout_no.to_numpy(dtype=np.float64)
    if (not np.isfinite(x).all() or not np.isfinite(cat).all()
            or not np.equal(cat, np.floor(cat)).all()
            or (np.abs(cat) > 2**53).any()):
        raise ValueError("Finite numeric features and exact integer spout values required")
    return x, cat.astype(np.int64)


class Encoder:
    """Float64 train-only numeric z scores and unit one-hot spout encoding."""
    def fit(self, frame):
        x, cat = validate_frame(frame)
        self.means_, self.stds_ = x.mean(0), x.std(0, ddof=0)
        self.stds_[self.stds_ == 0] = 1.
        self.categories_ = sorted(set(cat.tolist()))
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        return self

    def transform(self, frame):
        x, cat = validate_frame(frame)
        vocabulary = {v: i+1 for i, v in enumerate(self.categories_)}
        codes = np.array([vocabulary.get(v, 0) for v in cat], dtype=np.int64)
        one_hot = np.zeros((len(frame), len(self.categories_)+1), dtype=np.float64)
        one_hot[np.arange(len(frame)), codes] = 1.
        value = np.concatenate([(x-self.means_)/self.stds_, one_hot], axis=1)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite encoded inputs")
        return value

    def metadata(self):
        return {"numeric_features": list(FEATURES), "means": self.means_.tolist(),
                "stds": self.stds_.tolist(), "spout_categories": self.categories_,
                "unknown_spout_code": 0, "fit_ids": self.fit_ids_}

    @classmethod
    def from_metadata(cls, metadata):
        obj = cls()
        if (metadata["numeric_features"] != list(FEATURES)
                or metadata["unknown_spout_code"] != 0):
            raise ValueError("Saved encoder schema mismatch")
        obj.means_, obj.stds_ = np.array(metadata["means"]), np.array(metadata["stds"])
        categories = metadata["spout_categories"]
        if (obj.means_.shape != (len(FEATURES),) or obj.stds_.shape != obj.means_.shape
                or not np.isfinite([obj.means_, obj.stds_]).all()
                or (obj.stds_ <= 0).any() or not categories
                or any(type(v) is not int for v in categories)
                or categories != sorted(set(categories))):
            raise ValueError("Invalid saved encoder statistics/vocabulary")
        obj.categories_, obj.fit_ids_ = categories, metadata["fit_ids"]
        if (not obj.fit_ids_ or any(type(v) is not str or not v for v in obj.fit_ids_)
                or len(set(obj.fit_ids_)) != len(obj.fit_ids_)):
            raise ValueError("Invalid saved fitting identities")
        return obj


def exact_neighbors(points, query, ids, count, *, exclude=None, include_anchor=None):
    """Exact squared distances; ties resolved by string sample identity.

    An anchor must be supplied by its actual row index. Coincident feature
    vectors do not justify removing an arbitrary first neighbor. Derivative
    neighborhoods include the anchor's zero equation plus count-1 other rows.
    """
    if (type(count) is not int or count < 1 or len(ids) != len(points)
            or points.ndim != 2 or query.shape != (points.shape[1],)
            or not np.isfinite(points).all() or not np.isfinite(query).all()):
        raise ValueError("Invalid exact neighbor request")
    if exclude is not None and include_anchor is not None:
        raise ValueError("Conflicting anchor handling")
    anchor = exclude if exclude is not None else include_anchor
    if anchor is not None and (type(anchor) is not int or not 0 <= anchor < len(points)):
        raise ValueError("Invalid anchor index")
    delta = points-query
    squared = np.sum(delta*delta, axis=1)
    if not np.isfinite(squared).all():
        raise ValueError("Nonfinite neighbor distances")
    order = np.lexsort((np.asarray(ids, dtype=str), squared))
    if anchor is not None:
        order = order[order != anchor]
    if include_anchor is not None:
        order = np.r_[include_anchor, order]
    if count > len(order):
        raise ValueError("Neighborhood exceeds available fitting rows")
    return order[:count].astype(np.int64)


def reservation(observer, kind, metadata):
    context = observer(kind, metadata)
    return nullcontext() if context is None else context


def metric_epoch(x, y, ids, settings, observer):
    """One SGD epoch with a training-only graph frozen at initial unit scale."""
    n, d = x.shape
    count = min(n-1, settings.metric_multiple*d-1)
    if count < 2:
        raise ValueError("At least two other training neighbors required")
    with reservation(observer, "metric_epoch", {"rows": n, "updates": n, "local_solutions": n}):
        return _metric_epoch(x, y, ids, settings, count)


def _metric_epoch(x, y, ids, settings, count):
    n, d = x.shape
    # Fix both graph and local linearization errors before any scale update.
    graph = np.stack([exact_neighbors(x, x[i], ids, count, exclude=i) for i in range(n)])
    errors = np.empty((n, count), dtype=np.float64)
    for i, neighbors in enumerate(graph):
        _, errors[i] = local_linear_errors(x[i], y[i], x[neighbors], y[neighbors])
    order = list(range(n))
    random.Random(settings.random_seed).shuffle(order)
    scale = np.ones(d, dtype=np.float64)
    costs, gradients, before = [], [], []
    for i in order:
        cost, grad = scaling_cost_gradient(scale, x[graph[i]]-x[i], errors[i], settings.epsilon)
        before.append(scale.copy()); costs.append(cost); gradients.append(grad)
        scale = scale-settings.learning_rate*grad
        if not np.isfinite(scale).all() or np.linalg.norm(scale) <= settings.epsilon:
            raise ValueError("Nonfinite or degenerate learned metric; do not silently fall back")
    return scale, {"metric_graph": graph, "metric_errors": errors,
        "metric_order": np.asarray(order, dtype=np.int64), "metric_costs": np.asarray(costs),
        "metric_gradients": np.stack(gradients), "metric_scales_before": np.stack(before)}


class Regressor:
    def __init__(self, arm, settings=None):
        if arm not in ARMS:
            raise ValueError("Unknown DNNR arm")
        self.arm, self.settings = arm, settings or Settings()
        self.attempted_ = False

    def fit(self, frame, y, *, metric_epochs=0, observer=None):
        if self.attempted_:
            raise ValueError("No repeat fit or implicit resume, including failed initialization")
        self.attempted_ = True
        y = np.array(y, dtype=np.float64, copy=True)
        if (type(metric_epochs) is not int or metric_epochs not in (0, 1)
                or (self.arm != "DNNR_LEARNED" and metric_epochs != 0)):
            raise ValueError("Only learned Taylor may consume the one-epoch metric budget")
        validate_frame(frame)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or len(frame) < 3:
            raise ValueError("Finite matching training labels and at least three rows required")
        if self.settings.neighbors > len(frame):
            raise ValueError("Too few training rows for fixed vote count")
        observer = observer or (lambda _kind, _metadata: None)
        with reservation(observer, "estimator", {"arm": self.arm, "metric_epochs": metric_epochs, "rows": len(frame)}):
            return self._fit(frame, y, metric_epochs, observer)

    def _fit(self, frame, y, metric_epochs, observer):
        start = time.perf_counter()
        self.encoder_ = Encoder().fit(frame)
        self.ids_ = frame.sample_id.astype(str).tolist()
        x = self.encoder_.transform(frame)
        self.arrays_ = {"x": x, "y": y, "scale": np.ones(x.shape[1])}
        if metric_epochs:
            scale, witnesses = metric_epoch(x, y, self.ids_, self.settings, observer)
            self.arrays_.update(witnesses)
            self.arrays_["scale"] = scale
        scaled = x*self.arrays_["scale"]
        if self.arm != "KNN_FIXED":
            count = min(len(frame), self.settings.derivative_multiple*x.shape[1])
            with reservation(observer, "derivative_bank", {"rows": len(frame), "local_solutions": len(frame), "neighbors": count}):
                graph = np.stack([exact_neighbors(scaled, scaled[i], self.ids_, count,
                                                   include_anchor=i) for i in range(len(frame))])
                derivatives = np.stack([first_order_coefficients(scaled[i], y[i], scaled[g], y[g])
                                        for i, g in enumerate(graph)])
                self.arrays_.update(derivative_graph=graph, derivatives=derivatives)
        self.metric_epochs_, self.elapsed_seconds_ = metric_epochs, time.perf_counter()-start
        self.fitted_ = True
        return self

    def predict(self, frame):
        if not getattr(self, "fitted_", False):
            raise ValueError("DNNR has no successfully fitted state")
        if set(self.ids_) & set(frame.sample_id.astype(str)):
            raise ValueError("Prediction identities overlap fitting rows")
        query = self.encoder_.transform(frame)*self.arrays_["scale"]
        points = self.arrays_["x"]*self.arrays_["scale"]
        result = []
        for q in query:
            neighbors = exact_neighbors(points, q, self.ids_, self.settings.neighbors)
            values = self.arrays_["y"][neighbors].copy()
            if self.arm != "KNN_FIXED":
                values += np.sum(self.arrays_["derivatives"][neighbors]*(q-points[neighbors]), axis=1)
            result.append(float(values.mean()))
        result = np.array(result)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite Taylor extrapolation; no silent clipping")
        return result

    def metadata(self):
        if not getattr(self, "fitted_", False):
            raise ValueError("No completed model to serialize")
        return {"format": "dnnr-first-order-v1", "arm": self.arm,
            "settings": asdict(self.settings), "encoder": self.encoder_.metadata(),
            "metric_epochs": self.metric_epochs_, "elapsed_seconds": self.elapsed_seconds_,
            "array_digests": {k: digest(v.tolist()) for k, v in self.arrays_.items()},
            "metric_graph_policy": "initial_unit_scale_fixed_all_epoch",
            "derivative_coordinates": "final_scaled", "query_targets": "absent",
            "postprocessing": "none"}

    def save(self, path):
        metadata = self.metadata()
        with Path(path).open("xb") as stream:
            np.savez(stream, metadata=np.array(json.dumps(metadata, sort_keys=True)), **self.arrays_)
        return file_hash(path)

    @classmethod
    def load(cls, path, expected_sha256):
        if not expected_sha256 or file_hash(path) != expected_sha256:
            raise ValueError("Externally anchored saved DNNR hash required")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            arrays = {k: saved[k].copy() for k in saved.files if k != "metadata"}
        if (metadata["format"] != "dnnr-first-order-v1"
                or metadata["metric_graph_policy"] != "initial_unit_scale_fixed_all_epoch"
                or metadata["derivative_coordinates"] != "final_scaled"
                or metadata["query_targets"] != "absent" or metadata["postprocessing"] != "none"
                or type(metadata["metric_epochs"]) is not int or metadata["metric_epochs"] not in (0, 1)):
            raise ValueError("Saved model contract mismatch")
        obj = cls(metadata["arm"], Settings(**metadata["settings"]))
        obj.encoder_ = Encoder.from_metadata(metadata["encoder"])
        obj.ids_ = obj.encoder_.fit_ids_
        n, d = len(obj.ids_), len(FEATURES)+len(obj.encoder_.categories_)+1
        expected = {"x": (n, d), "y": (n,), "scale": (d,)}
        if obj.arm != "KNN_FIXED":
            expected.update(derivatives=(n, d),
                            derivative_graph=(n, min(n, obj.settings.derivative_multiple*d)))
        if metadata["metric_epochs"]:
            if obj.arm != "DNNR_LEARNED":
                raise ValueError("Fixed arm has a learned metric witness")
            count = min(n-1, obj.settings.metric_multiple*d-1)
            expected.update(metric_graph=(n, count), metric_errors=(n, count),
                            metric_order=(n,), metric_costs=(n,), metric_gradients=(n, d),
                            metric_scales_before=(n, d))
        if (set(arrays) != set(expected) or set(metadata["array_digests"]) != set(expected)
                or n < 3 or obj.settings.neighbors > n):
            raise ValueError("Saved model array schema mismatch")
        for key, shape in expected.items():
            value = arrays[key]
            if (value.shape != shape or not np.isfinite(value).all()
                    or digest(value.tolist()) != metadata["array_digests"][key]):
                raise ValueError("Saved model shape/value/digest mismatch")
            if key.endswith("graph") or key == "metric_order":
                if value.dtype != np.int64 or (value < 0).any() or (value >= n).any():
                    raise ValueError("Invalid saved neighborhood/order index")
            elif value.dtype != np.float64:
                raise ValueError("Float64 model arrays required")
        if not metadata["metric_epochs"] and not np.array_equal(arrays["scale"], np.ones(d)):
            raise ValueError("Fixed metric must be exactly unit scale")
        if np.linalg.norm(arrays["scale"]) <= obj.settings.epsilon:
            raise ValueError("Degenerate saved metric")
        if not np.isfinite(metadata["elapsed_seconds"]) or metadata["elapsed_seconds"] < 0:
            raise ValueError("Invalid saved fit duration")
        obj.arrays_, obj.metric_epochs_, obj.elapsed_seconds_ = arrays, metadata["metric_epochs"], metadata["elapsed_seconds"]
        obj.fitted_, obj.attempted_ = True, True
        obj.saved_metadata_ = metadata
        return obj
