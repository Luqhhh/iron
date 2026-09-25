"""V4.2 S-line PySR backend adapter.

The verified feasibility record for this round is:

* ``pysr 2.5.0`` installs into an **independent** environment
  (``local/envs/v4_2_pysr``) with its own Julia depot
  (``local/envs/julia-depot``); the frozen V3.6 environment is not touched;
* ``PySRRegressor`` accepts an exact ``max_evals`` budget, so the
  pre-registered per-training-subset expression-evaluation cap can be enforced
  rather than approximated;
* a warm 20000-evaluation search on a 1800 x 25 design matrix took about
  1.3 s, projecting roughly 13 s for the full 200000-evaluation budget, which
  is affordable for this round's slot count.

The adapter therefore runs PySR as the published S backend, and converts every
returned hall-of-fame equation back into the frozen V4.2 AST language so that
constant re-estimation, non-finite rejection and NumPy export all use exactly
one code path.

Example data downloads are never used: only the caller-supplied design matrix
and target are handed to the worker.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "PysrSettings",
    "pysr_worker_path",
    "run_pysr_search",
    "sympy_text_to_ast",
    "unsupported_sympy_names",
]


def pysr_worker_path(root: Path | str = ".") -> Path:
    return (Path(root).resolve() / "src/bf_tap_r2/v4_2_pysr_worker.py")


@dataclass(frozen=True)
class PysrSettings:
    python_path: Path
    julia_depot: Path
    eval_budget: int = 200_000
    maxsize: int = 24
    maxdepth: int = 5
    populations: int = 16
    population_size: int = 40
    tournament_selection_n: int = 5
    seed: int = 42
    timeout_seconds: int = 3600

    @classmethod
    def default(cls, root: Path | str = ".") -> "PysrSettings":
        base = Path(root).resolve()
        return cls(
            python_path=base / "local/envs/v4_2_pysr/bin/python",
            julia_depot=base / "local/envs/julia-depot",
        )

    def available(self) -> tuple[bool, str]:
        if not self.python_path.is_file():
            return False, f"missing independent PySR interpreter: {self.python_path}"
        if not os.access(self.python_path, os.X_OK):
            return False, f"PySR interpreter is not executable: {self.python_path}"
        if not self.julia_depot.is_dir():
            return False, f"missing Julia depot: {self.julia_depot}"
        return True, "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_path": str(self.python_path),
            "julia_depot": str(self.julia_depot),
            "eval_budget": int(self.eval_budget),
            "maxsize": int(self.maxsize),
            "maxdepth": int(self.maxdepth),
            "populations": int(self.populations),
            "population_size": int(self.population_size),
            "tournament_selection_n": int(self.tournament_selection_n),
            "seed": int(self.seed),
            "timeout_seconds": int(self.timeout_seconds),
            "deterministic": True,
            "parallelism": "serial",
        }


def run_pysr_search(
    x: np.ndarray,
    y: np.ndarray,
    operators: Sequence[str],
    *,
    settings: PysrSettings,
    root: Path | str = ".",
    scratch_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run one PySR search in the independent interpreter and return its hall of fame."""
    ok, reason = settings.available()
    if not ok:
        return {"ok": False, "status": "backend_unavailable", "error": reason,
                "settings": settings.as_dict()}
    base = Path(root).resolve()
    scratch = Path(scratch_dir) if scratch_dir is not None else base / "local/tmp/v4_2_pysr_jobs"
    scratch = scratch if scratch.is_absolute() else base / scratch
    scratch.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}"
    job_path = scratch / f"job-{stamp}.json"
    out_path = scratch / f"out-{stamp}.json"
    job = {
        "X": np.asarray(x, dtype=np.float64).tolist(),
        "y": np.asarray(y, dtype=np.float64).reshape(-1).tolist(),
        "operators": [str(op) for op in operators],
        "eval_budget": int(settings.eval_budget),
        "maxsize": int(settings.maxsize),
        "maxdepth": int(settings.maxdepth),
        "populations": int(settings.populations),
        "population_size": int(settings.population_size),
        "tournament_selection_n": int(settings.tournament_selection_n),
        "seed": int(settings.seed),
        "output": str(out_path),
    }
    job_path.write_text(json.dumps(job), encoding="utf-8")
    environment = dict(os.environ)
    environment["JULIA_DEPOT_PATH"] = str(settings.julia_depot)
    environment["PYTHONPATH"] = ""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(settings.python_path), str(pysr_worker_path(base)), str(job_path)],
            cwd=str(base), env=environment, capture_output=True, text=True,
            timeout=int(settings.timeout_seconds), check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "status": "timeout", "settings": settings.as_dict(),
                "seconds": float(time.perf_counter() - started)}
    elapsed = float(time.perf_counter() - started)
    if not out_path.is_file():
        return {
            "ok": False, "status": "worker_failed", "settings": settings.as_dict(),
            "seconds": elapsed, "returncode": int(completed.returncode),
            "stderr_tail": (completed.stderr or "")[-2000:],
            "stdout_tail": (completed.stdout or "")[-1000:],
        }
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["settings"] = settings.as_dict()
    payload["adapter_seconds"] = round(elapsed, 3)
    payload["returncode"] = int(completed.returncode)
    for path in (job_path, out_path):
        try:
            path.unlink()
        except OSError:
            pass
    return payload


# ---------------------------------------------------------------------------
# PySR equation text -> frozen V4.2 AST
# ---------------------------------------------------------------------------

class UnsupportedSympyExpression(ValueError):
    """Raised when a PySR equation uses a form outside the frozen operator set."""


def unsupported_sympy_names() -> tuple[str, ...]:
    return ("sin", "cos", "exp", "sqrt", "Abs", "sign", "Min", "Max", "Piecewise")


def _sympy():
    try:
        import sympy
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("sympy is required to convert PySR expressions") from exc
    return sympy


def sympy_text_to_ast(text: str, n_variables: int) -> tuple:
    """Convert one PySR equation string into a V4.2 AST tuple."""
    sympy = _sympy()

    class safe_ratio(sympy.Function):  # noqa: N801 - mirrors the PySR operator name
        nargs = 2

    class log1p_abs(sympy.Function):  # noqa: N801
        nargs = 1

    expression = sympy.sympify(
        text, locals={"safe_ratio": safe_ratio, "log1p_abs": log1p_abs}
    )
    counter = [0]

    def convert(node) -> tuple:
        if isinstance(node, sympy.Symbol):
            name = str(node)
            if not name.startswith("x") or not name[1:].isdigit():
                raise UnsupportedSympyExpression(f"unexpected symbol {name!r}")
            index = int(name[1:])
            if not 0 <= index < int(n_variables):
                raise UnsupportedSympyExpression(f"symbol {name!r} outside the design matrix")
            return ("var", index)
        if isinstance(node, sympy.Number):
            return ("const", float(node))
        if isinstance(node, sympy.Add):
            terms = [convert(arg) for arg in node.args]
            result = terms[0]
            for term in terms[1:]:
                result = ("add", result, term)
            return result
        if isinstance(node, sympy.Mul):
            factors = [convert(arg) for arg in node.args]
            result = factors[0]
            for factor in factors[1:]:
                result = ("mul", result, factor)
            return result
        if isinstance(node, sympy.Pow):
            base, exponent = node.args
            if not isinstance(exponent, sympy.Integer):
                raise UnsupportedSympyExpression("non-integer exponent is forbidden")
            power = int(exponent)
            if power < 0 or power > 4:
                raise UnsupportedSympyExpression(f"exponent {power} is forbidden")
            converted = convert(base)
            result = converted
            for _ in range(power - 1):
                result = ("mul", result, converted)
            return result if power >= 1 else ("const", 1.0)
        if isinstance(node, sympy.tanh):
            return ("tanh", convert(node.args[0]))
        if node.func is safe_ratio or str(getattr(node, "func", "")) == "safe_ratio":
            arguments = [convert(arg) for arg in node.args]
            if len(arguments) != 2:
                raise UnsupportedSympyExpression("safe_ratio requires two arguments")
            return ("safe_ratio", arguments[0], arguments[1])
        if isinstance(node, log1p_abs) or str(getattr(node, "func", "")) == "log1p_abs":
            return ("log1p_abs", convert(node.args[0]))
        if isinstance(node, sympy.log):
            argument = node.args[0]
            if isinstance(argument, sympy.Abs):
                return ("log1p_abs", convert(argument.args[0]))
            if isinstance(argument, sympy.Add) and len(argument.args) == 2:
                one, other = argument.args
                if isinstance(one, sympy.Integer) and int(one) == 1 and isinstance(other, sympy.Abs):
                    return ("log1p_abs", convert(other.args[0]))
                if isinstance(other, sympy.Integer) and int(other) == 1 and isinstance(one, sympy.Abs):
                    return ("log1p_abs", convert(one.args[0]))
            raise UnsupportedSympyExpression("only log(1 + |a|) is allowed")
        counter[0] += 1
        raise UnsupportedSympyExpression(
            f"expression form outside the frozen operator set: {type(node).__name__}"
        )

    return convert(expression)
