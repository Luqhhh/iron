"""One fixed Q2 route; preserves the frozen v0.1 estimator implementations."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from .data import FEATURES
from .models import inputs


def q2_preprocessor(spec):
    return ColumnTransformer([
        ("numeric", Pipeline([("scale_input", StandardScaler()),
                              ("poly", PolynomialFeatures(**spec["polynomial"])),
                              ("scale_expanded", StandardScaler())]), list(FEATURES)),
        ("spout", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["spout_no"])
    ], remainder="drop")


class Q2Regressor:
    def __init__(self, spec):
        self.spec = spec

    def fit(self, frame, target):
        y = np.asarray(target, dtype=float)
        if y.ndim != 1 or len(y) != len(frame) or not np.isfinite(y).all():
            raise ValueError("Invalid Q2 target")
        self.target_scale_ = float(np.median(y))
        if self.target_scale_ <= 0:
            raise ValueError("Positive training-fold target median required")
        self.estimator_ = Pipeline([("preprocess", q2_preprocessor(self.spec)),
                                    ("regression", QuantileRegressor(**self.spec["regressor"]))])
        self.estimator_.fit(inputs(frame), y / self.target_scale_)
        return self

    def predict(self, frame):
        prediction = self.estimator_.predict(inputs(frame)) * self.target_scale_
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite Q2 predictions")
        return prediction


def s25_predictions(medians, old):
    medians, old = np.asarray(medians, dtype=float), np.asarray(old, dtype=float)
    if medians.shape != old.shape or not np.isfinite(medians).all() or not np.isfinite(old).all():
        raise ValueError("S25 requires equally sized finite aligned arrays")
    return .75 * medians + .25 * old


def assess_candidate(scores, anchors, rules):
    tolerance = rules["numerical_equality_tolerance"]
    seeds = sorted(anchors)
    overall = [scores[s]["wmape"] < anchors[s]["wmape"] - tolerance for s in seeds]
    fold_wins = sum(scores[s]["by_fold"][f] < anchors[s]["by_fold"][f] - tolerance
                    for s in seeds for f in anchors[s]["by_fold"])
    worst_spout = max(scores[s]["by_spout"][k] - anchors[s]["by_spout"][k]
                      for s in seeds for k in anchors[s]["by_spout"])
    return {"eligible": bool(all(overall) and fold_wins >= rules["minimum_improved_folds"]
                              and worst_spout <= rules["max_spout_wmape_degradation"] + tolerance),
            "pooled_improvements": {str(s): bool(v) for s, v in zip(seeds, overall)},
            "improved_folds": fold_wins, "worst_spout_delta": worst_spout,
            "mean_wmape": float(np.mean([scores[s]["wmape"] for s in seeds]))}
