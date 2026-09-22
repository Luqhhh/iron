"""Optional pytest plugin counting observed estimator fits without changing behavior.

Set BF_TAP_TEST_FIT_REPORT to an unused private JSON path; load with -p pytest_fit_audit.
Counts describe this pytest process only. Nested fits are separately reported.
"""
from collections import Counter
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import threading

COUNTS = Counter()
OUTER = Counter()
STATE = threading.local()
PATCHES = []


def pytest_sessionstart(session):
    classes = {
        "catboost": ["CatBoostRegressor"],
        "lightgbm": ["LGBMRegressor"],
        "sklearn.neural_network": ["MLPRegressor"],
        "sklearn.kernel_ridge": ["KernelRidge"],
        "sklearn.linear_model": ["Ridge", "QuantileRegressor", "LinearRegression"],
        "sklearn.svm": ["SVR"],
        "sklearn.ensemble": ["RandomForestRegressor", "ExtraTreesRegressor", "HistGradientBoostingRegressor", "GradientBoostingRegressor"],
        "sklearn.tree": ["DecisionTreeRegressor", "ExtraTreeRegressor"],
    }
    for module, names in classes.items():
        imported = importlib.import_module(module)
        for name in names:
            cls = getattr(imported, name)
            original = cls.fit
            key = module + "." + name

            def wrap(original, key):
                @wraps(original)
                def fit(self, *args, **kwargs):
                    depth = getattr(STATE, "depth", 0)
                    COUNTS[key] += 1
                    if depth == 0:
                        OUTER[key] += 1
                    STATE.depth = depth + 1
                    try:
                        return original(self, *args, **kwargs)
                    finally:
                        STATE.depth = depth
                return fit

            PATCHES.append((cls, original))
            cls.fit = wrap(original, key)


def pytest_sessionfinish(session, exitstatus):
    for cls, original in reversed(PATCHES):
        cls.fit = original
    target = os.environ.get("BF_TAP_TEST_FIT_REPORT")
    if target:
        path = Path(target).resolve()
        if not path.is_relative_to(Path.cwd().resolve() / "local"):
            raise ValueError("Synthetic fit accounting must stay private")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump({"pytest_exitstatus": int(exitstatus), "observed_outer_estimator_fit_calls": dict(OUTER),
                       "observed_all_estimator_fit_calls": dict(COUNTS),
                       "scope": "Instrumented estimator entries in this pytest process; subprocesses and uninstrumented estimators excluded; nested calls not additional top-level fits",
                       "real_competition_training": False}, handle, indent=2)
