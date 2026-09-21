"""One frozen RBF SVR with fold-local transformations and convergence gate."""
import warnings
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from .data import FEATURES
from .models import inputs


class KernelRegressor:
    def __init__(self, spec):
        self.spec = dict(spec)

    def fit(self, frame, target):
        y = np.asarray(target, dtype=float)
        if y.ndim != 1 or len(y) != len(frame) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Invalid target")
        self.median_ = float(np.median(y))
        if self.median_ <= 0:
            raise ValueError("Positive training median required")
        self.preprocessor_ = ColumnTransformer([
            ("numeric", StandardScaler(), list(FEATURES)),
            ("spout", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["spout_no"])
        ])
        x = self.preprocessor_.fit_transform(inputs(frame))
        self.regressor_ = SVR(**self.spec)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.regressor_.fit(x, (y - self.median_) / self.median_)
        self.fit_report_ = {
            "fit_status": int(self.regressor_.fit_status_),
            "iterations": int(self.regressor_.n_iter_),
            "support_vectors": int(self.regressor_.n_support_.sum()),
            "gamma": float(self.regressor_._gamma),
            "warnings": [str(w.message) for w in caught],
            "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
        }
        if self.fit_report_["fit_status"] or self.fit_report_["convergence_warning"]:
            raise RuntimeError("K1 engineering gate failed: " + str(self.fit_report_))
        return self

    def predict_raw(self, frame):
        z = self.regressor_.predict(self.preprocessor_.transform(inputs(frame)))
        result = self.median_ * (1 + z)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite prediction")
        return result

    def predict(self, frame):
        return np.maximum(0., self.predict_raw(frame))
