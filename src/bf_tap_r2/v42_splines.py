"""Deterministic adaptive hinge products with training-only basis construction."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import resource

import numpy as np

from .data import FEATURES, TARGETS
from .v7_periodic import digest
from .v30_deep_kernel import select_weight


def least_squares(matrix, y, rtol):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = s > rtol*s[0]
    q, singular, right = u[:, keep], s[keep], vt[keep]
    coef = right.T @ ((q.T @ y)/singular)
    residual = y-matrix@coef
    return coef, float(residual@residual), q, singular, right


def pair_gains(q, residual, left, right, rtol):
    """RSS reduction after adding each pair to the current column space."""
    a, b = left-q@(q.T@left), right-q@(q.T@right)
    gram = np.empty((left.shape[1], 2, 2))
    gram[:, 0, 0], gram[:, 1, 1] = np.sum(a*a, axis=0), np.sum(b*b, axis=0)
    gram[:, 0, 1] = gram[:, 1, 0] = np.sum(a*b, axis=0)
    eigen, vectors = np.linalg.eigh(gram)
    # Squaring the SVD cutoff alone is below roundoff for a 2x2 Gram matrix.
    threshold = np.maximum(eigen[:, -1]*max(rtol**2, 1e-14), 1e-20)
    inv = np.divide(1., eigen, out=np.zeros_like(eigen), where=eigen > threshold[:, None])
    rhs = np.column_stack([a.T@residual, b.T@residual])
    rotated = np.einsum("nki,nk->ni", vectors, rhs)
    return np.sum(rotated**2*inv, axis=1)


def deletion_costs(coef, singular, right):
    """Exact least-squares deletion increase, including redundant columns."""
    null_diagonal = np.maximum(0., 1-np.sum(right*right, axis=0))
    inverse_diagonal = np.sum((right/singular[:, None])**2, axis=0)
    costs = np.divide(coef**2, inverse_diagonal, out=np.zeros_like(coef), where=inverse_diagonal > 0)
    costs[null_diagonal > 1e-8] = 0.
    costs[0] = np.inf  # Keep the intercept.
    return costs


class SplinePreprocessor:
    def fit(self, frame):
        x = frame[list(FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite training features")
        self.mean_, self.scale_ = x.mean(axis=0), x.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.
        self.categories_ = sorted(int(v) for v in frame.spout_no.unique())
        self.fit_ids_digest_ = digest(frame.sample_id.astype(str).tolist())
        return self

    def transform(self, frame):
        x = frame[list(FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite query features")
        return np.column_stack([(x-self.mean_)/self.scale_,
                                *[(frame.spout_no.to_numpy() == c).astype(float) for c in self.categories_]])

    def metadata(self):
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
                "categories": self.categories_, "fit_ids_digest": self.fit_ids_digest_}

    @classmethod
    def restore(cls, data):
        obj = cls()
        obj.mean_, obj.scale_ = np.array(data["mean"]), np.array(data["scale"])
        obj.categories_, obj.fit_ids_digest_ = data["categories"], data["fit_ids_digest"]
        return obj


def basis_matrix(x, terms):
    columns = []
    for factors in terms:
        column = np.ones(len(x))
        for feature, knot, sign in factors:
            column *= np.maximum(sign*(x[:, feature]-knot), 0.)
        columns.append(column)
    return np.column_stack(columns)


class SplineRegressor:
    def __init__(self, recipe, settings):
        self.recipe, self.settings = recipe, deepcopy(settings)

    def fit(self, frame, y, validation=None, selected_count=None):
        y = np.asarray(y, float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or y.std() <= 0:
            raise ValueError("Invalid training target")
        self.preprocessor_ = SplinePreprocessor().fit(frame)
        self.mean_, self.std_ = float(y.mean()), float(y.std())
        target = (y-self.mean_)/self.std_
        x = self.preprocessor_.transform(frame)
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        self.terms_, self.forward_ = [[]], []
        degree = self.settings["max_degree"][self.recipe]
        rtol = self.settings["rank_rtol"]
        matrix = np.ones((len(frame), 1))
        while len(self.terms_)+2 <= self.settings["max_terms"]:
            coef, rss, q, _, _ = least_squares(matrix, target, rtol)
            residual, best = target-matrix@coef, None
            for parent, factors in enumerate(self.terms_):
                if len(factors) >= degree:
                    continue
                used = {min(f[0], len(FEATURES)) for f in factors}
                active = matrix[:, parent] > 0
                for feature in range(x.shape[1]):
                    if min(feature, len(FEATURES)) in used:
                        continue
                    values = x[active, feature]
                    if len(values) < 2*self.settings["min_side_rows"]:
                        continue
                    knots = ([self.settings["categorical_knot"]] if feature >= len(FEATURES)
                             else np.unique(np.quantile(values, self.settings["knot_quantiles"])))
                    knots = [float(k) for k in knots
                             if min(np.sum(values < k), np.sum(values > k)) >= self.settings["min_side_rows"]]
                    if not knots:
                        continue
                    left = matrix[:, parent, None]*np.maximum(x[:, feature, None]-knots, 0.)
                    right = matrix[:, parent, None]*np.maximum(np.array(knots)-x[:, feature, None], 0.)
                    gains = pair_gains(q, residual, left, right, rtol)
                    index = int(np.argmax(gains))
                    if best is None or gains[index] > best[0]:
                        best = (float(gains[index]), parent, feature, knots[index], left[:, index], right[:, index])
            if best is None or best[0] <= self.settings["min_relative_improvement"]*max(rss, 1e-20):
                break
            gain, parent, feature, knot, left, right = best
            self.terms_ += [self.terms_[parent]+[[feature, knot, 1]],
                            self.terms_[parent]+[[feature, knot, -1]]]
            matrix = np.column_stack([matrix, left, right])
            _, after, _, _, _ = least_squares(matrix, target, rtol)
            if after > rss+1e-8*max(1., rss):
                raise ValueError("Forward least squares increased RSS")
            self.forward_.append({"terms": len(self.terms_), "parent": parent, "feature": feature,
                                  "knot": knot, "predicted_gain": gain, "rss": after})
        self.path_ = []
        active = list(range(len(self.terms_)))
        while active:
            coef, rss, _, singular, right = least_squares(matrix[:, active], target, rtol)
            self.path_.append({"count": len(active), "active": active.copy(), "coef": coef.tolist(), "rss": rss})
            if len(active) == 1:
                break
            costs = deletion_costs(coef, singular, right)
            active.pop(int(np.argmin(costs)))
        self.selection_ = []
        if validation is not None:
            query, actual = validation
            qb = basis_matrix(self.preprocessor_.transform(query), self.terms_)
            counts = set(self.settings["selection_counts"]) | {len(self.terms_)}
            for row in self.path_:
                if row["count"] in counts:
                    pred = qb[:, row["active"]]@np.array(row["coef"])*self.std_+self.mean_
                    self.selection_.append({"count": row["count"], "calibration_mae": float(np.abs(actual-pred).mean())})
            chosen = min(self.selection_, key=lambda r: (r["calibration_mae"], r["count"]))["count"]
        elif selected_count is not None and selected_count >= 1:
            chosen = min(int(selected_count), len(self.terms_))
        else:
            raise ValueError("Calibration or selected complexity required")
        self.selected_ = next(row for row in self.path_ if row["count"] == chosen)
        self.requested_count_ = selected_count
        return self

    def predict(self, frame):
        x = self.preprocessor_.transform(frame)
        matrix = basis_matrix(x, [self.terms_[i] for i in self.selected_["active"]])
        result = matrix@np.array(self.selected_["coef"])*self.std_+self.mean_
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite prediction")
        return result

    def metadata(self):
        return {"recipe": self.recipe, "fit_ids_digest": digest(self.fit_ids_), "fit_rows": len(self.fit_ids_),
                "preprocessing": self.preprocessor_.metadata(), "target_mean": self.mean_, "target_std": self.std_,
                "selected_count": self.selected_["count"], "requested_count": self.requested_count_,
                "forward_terms": len(self.terms_), "forward": self.forward_, "selection": self.selection_,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}

    def save(self, path):
        with Path(path).open("x") as stream:
            json.dump({"recipe": self.recipe, "settings": self.settings,
                       "preprocessing": self.preprocessor_.metadata(), "mean": self.mean_, "std": self.std_,
                       "terms": self.terms_, "path": self.path_, "selected": self.selected_}, stream, allow_nan=False)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        obj = cls(data["recipe"], data["settings"])
        obj.preprocessor_ = SplinePreprocessor.restore(data["preprocessing"])
        obj.mean_, obj.std_ = data["mean"], data["std"]
        obj.terms_, obj.path_, obj.selected_ = data["terms"], data["path"], data["selected"]
        return obj


def fit_partition(fitting, calibration, outer_training, query, target, recipe, settings, calibration_base, grid):
    if (set(fitting.sample_id) & set(calibration.sample_id)
            or set(outer_training.sample_id) != set(fitting.sample_id) | set(calibration.sample_id)
            or set(outer_training.sample_id) & set(query.sample_id)):
        raise ValueError("Invalid nested partitions")
    if any(t in query for t in TARGETS):
        raise ValueError("Query labels must be removed")
    selector = SplineRegressor(recipe, settings).fit(fitting, fitting[target].to_numpy(),
                                                    validation=(calibration, calibration[target].to_numpy()))
    cp = selector.predict(calibration)
    weight, losses = select_weight(calibration[target], calibration_base, cp, grid)
    final = SplineRegressor(recipe, settings).fit(outer_training, outer_training[target].to_numpy(),
                                               selected_count=selector.selected_["count"])
    final.calibration_model_ = selector
    return final, final.predict(query), {"weight": weight, "calibration_mae_by_weight": losses,
                                        "calibration": selector.metadata(), "refit": final.metadata()}, cp
