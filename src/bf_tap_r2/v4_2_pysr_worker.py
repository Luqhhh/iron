"""Standalone PySR search worker for the V4.2 S line.

This file is executed by the **independent** V4.2 PySR interpreter
(``local/envs/v4_2_pysr/bin/python``) created for this round, never by the
frozen V3.6 environment.  It therefore imports nothing from ``bf_tap_r2``.

It reads a job description from ``argv[1]`` and writes its result to the path
named there.  Example data downloads are never used: only the caller-supplied
design matrix and target are fitted.
"""
from __future__ import annotations

import json
import sys
import time


def build_operators(requested):
    """Translate the frozen V4.2 operator set into PySR operator declarations."""
    binary = ["+", "-", "*"]
    unary = []
    for name in requested:
        if name == "safe_ratio":
            binary.append("safe_ratio(a, b) = a / (1 + b * b)")
        elif name == "log1p_abs":
            unary.append("log1p_abs(a) = log1p(abs(a))")
        elif name == "tanh":
            unary.append("tanh")
        elif name in {"add", "sub", "mul"}:
            continue
        else:
            raise ValueError(f"Unsupported V4.2 symbolic operator for PySR: {name!r}")
    return binary, unary


def main(argv):
    job = json.loads(open(argv[1], encoding="utf-8").read())
    import numpy as np
    import sympy
    from pysr import PySRRegressor
    import pysr as pysr_module

    X = np.asarray(job["X"], dtype=np.float64)
    y = np.asarray(job["y"], dtype=np.float64).reshape(-1)
    binary, unary = build_operators(job["operators"])
    # The pre-registered no-self-nesting rules: nonlinear unary operators may
    # not nest inside each other and safe_ratio may not nest inside itself.
    # Only operators that are actually declared may appear in the constraints,
    # otherwise PySR rejects the option set.
    declared = [
        name for name in ("tanh", "log1p_abs", "safe_ratio") if name in job["operators"]
    ]
    nested_constraints = {}
    for name in declared:
        if name == "safe_ratio":
            nested_constraints[name] = {"safe_ratio": 0}
        else:
            nested_constraints[name] = {
                other: 0 for other in declared if other != "safe_ratio"
            }

    extra_sympy_mappings = {}
    if "safe_ratio" in job["operators"]:
        extra_sympy_mappings["safe_ratio"] = lambda a, b: a / (1 + b * b)
    if "log1p_abs" in job["operators"]:
        extra_sympy_mappings["log1p_abs"] = lambda a: sympy.log(1 + sympy.Abs(a))

    kwargs = {
        "niterations": 10**7,
        "populations": int(job["populations"]),
        "population_size": int(job["population_size"]),
        "tournament_selection_n": int(job["tournament_selection_n"]),
        "maxsize": int(job["maxsize"]),
        "maxdepth": int(job["maxdepth"]),
        "max_evals": int(job["eval_budget"]),
        "binary_operators": binary,
        "unary_operators": unary,
        "nested_constraints": nested_constraints,
        "extra_sympy_mappings": extra_sympy_mappings,
        "elementwise_loss": "loss(prediction, target) = abs(prediction - target)",
        "random_state": int(job["seed"]),
        "deterministic": True,
        # Deterministic searches require the modern serial parallelism switch.
        # ``procs``/``multithreading`` must not be mixed with it.
        "parallelism": "serial",
        "progress": False,
        "verbosity": 0,
        "temp_equation_file": True,
        # ``temp_equation_file`` and ``output_directory`` are mutually exclusive
        # in PySR 2.x; the temporary-file mode keeps every artefact outside the
        # repository without writing a hall-of-fame file next to the sources.
        "model_selection": "best",
        "delete_tempfiles": True,
    }
    result = {
        "ok": False,
        "pysr_version": str(getattr(pysr_module, "__version__", "unknown")),
        "requested": {
            "eval_budget": int(job["eval_budget"]),
            "maxsize": int(job["maxsize"]),
            "maxdepth": int(job["maxdepth"]),
            "operators": list(job["operators"]),
            "seed": int(job["seed"]),
        },
    }
    import inspect

    allowed = set(inspect.signature(PySRRegressor.__init__).parameters)
    dropped = sorted(set(kwargs) - allowed)
    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    result["dropped_unsupported_parameters"] = dropped
    started = time.perf_counter()
    try:
        model = PySRRegressor(**kwargs)
        model.fit(X, y)
        frame = model.equations_
        equations = []
        for _, row in frame.iterrows():
            equations.append({
                "equation": str(row["equation"]),
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
            })
        result.update({
            "ok": True,
            "equations": equations,
            "n_equations": len(equations),
            "effective": {
                "niterations": kwargs["niterations"],
                "populations": kwargs["populations"],
                "population_size": kwargs["population_size"],
                "max_evals": kwargs["max_evals"],
                "maxsize": kwargs["maxsize"],
                "maxdepth": kwargs["maxdepth"],
                "deterministic": True,
                "parallelism": "serial",
            },
        })
    except Exception as exc:  # noqa: BLE001
        result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    result["seconds"] = round(float(time.perf_counter() - started), 3)
    with open(job["output"], "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
