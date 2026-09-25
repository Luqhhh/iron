"""V4.2 shared training convention and bookkeeping.

Every V4.2 fit records the requested parameters, the parameters that actually
took effect, the stopping reason, the randomness control, the fit-row ID/group
hashes and the input/target transformation hash.  R and N use the same fixed
MAE scheme; S uses an absolute-error objective.  Early stopping always monitors
the original-unit WMAPE.

A trial that reaches ``max_epochs`` while the early-stopping criterion has not
fired is reported as ``budget_limited`` rather than as a permanent family
failure.
"""
from __future__ import annotations

import math
import resource
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .metrics import wmape

__all__ = [
    "DEFAULT_TRAIN",
    "TrainConfig",
    "TrainOutcome",
    "WmapeEarlyStopping",
    "peak_memory_report",
]

#: The pre-registered V4.2 neural starting point.  These are this round's design
#: values, not an external paper's recommended optimum.
DEFAULT_TRAIN: dict[str, Any] = {
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "max_epochs": 1000,
    "early_stopping_patience": 50,
    "min_delta": 0.0,
    "seed": 42,
}

#: R and N share this fixed objective; S optimises absolute error as well.
NEURAL_LOSS = "mae"
SYMBOLIC_LOSS = "absolute_error"


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 256
    max_epochs: int = 1000
    early_stopping_patience: int = 50
    min_delta: float = 0.0
    seed: int = 42
    loss: str = NEURAL_LOSS

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None = None) -> "TrainConfig":
        merged = dict(DEFAULT_TRAIN)
        if payload:
            merged.update({str(k): v for k, v in payload.items()})
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = sorted(set(merged) - allowed)
        if unknown:
            raise ValueError(f"Unknown V4.2 training parameters: {unknown}")
        # Fields absent from both the mapping and DEFAULT_TRAIN keep their
        # dataclass default (for example ``loss``).
        values = {k: merged[k] for k in allowed if k in merged}
        values["batch_size"] = max(1, int(values["batch_size"]))
        values["max_epochs"] = max(1, int(values["max_epochs"]))
        values["early_stopping_patience"] = max(1, int(values["early_stopping_patience"]))
        return cls(**values)  # type: ignore[arg-type]

    def requested(self) -> dict[str, Any]:
        return {
            "optimizer": str(self.optimizer),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "early_stopping_patience": int(self.early_stopping_patience),
            "min_delta": float(self.min_delta),
            "seed": int(self.seed),
            "loss": str(self.loss),
        }


def peak_memory_report(
    tracemalloc_peak_bytes: int, rss_before_kb: int, rss_after_kb: int
) -> dict[str, Any]:
    """Honest peak-memory reporting: python-allocator peak and process RSS delta."""
    return {
        "kind": "tracemalloc_peak_plus_process_ru_maxrss_delta",
        "tracemalloc_peak_mb": round(float(tracemalloc_peak_bytes) / (1024.0 * 1024.0), 4),
        "process_maxrss_before_mb": round(float(rss_before_kb) / 1024.0, 4),
        "process_maxrss_after_mb": round(float(rss_after_kb) / 1024.0, 4),
        "process_maxrss_delta_mb": round(float(rss_after_kb - rss_before_kb) / 1024.0, 4),
        "note": "ru_maxrss is a process high-water mark and never decreases within a process.",
    }


def _maxrss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class WmapeEarlyStopping:
    """Original-unit WMAPE early stopping with an explicit stopping reason."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.best_value = math.inf
        self.best_epoch = 0
        self.best_state: Any = None
        self.stale = 0
        self.stopped_epoch = 0
        self.fired = False

    def update(self, value: float, epoch: int, state: Any) -> bool:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Early stopping received a nonfinite validation WMAPE")
        improved = value < self.best_value - self.min_delta
        if improved:
            self.best_value = value
            self.best_epoch = int(epoch)
            self.best_state = state
            self.stale = 0
        else:
            self.stale += 1
        self.stopped_epoch = int(epoch)
        if self.stale >= self.patience:
            self.fired = True
        return improved

    def stop_reason(self, max_epochs: int) -> str:
        if self.fired:
            return "early_stopping"
        # The criterion never fired inside the epoch budget: the trial did not
        # stabilise in time, which is a budget limitation, not a family failure.
        if (int(max_epochs) - int(self.best_epoch)) < self.patience:
            return "budget_limited"
        return "max_epochs_stable"


@dataclass
class TrainOutcome:
    """Bookkeeping for one V4.2 fit."""

    requested: dict[str, Any]
    effective: dict[str, Any]
    stop_reason: str
    best_epoch: int
    stopped_epoch: int
    best_validation_wmape: float
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0
    peak_memory: dict[str, Any] = field(default_factory=dict)
    randomness: dict[str, Any] = field(default_factory=dict)
    fit_row_id_hash: str = ""
    fit_group_hash: str = ""
    transform_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "requested_parameters": dict(self.requested),
            "effective_parameters": dict(self.effective),
            "stop_reason": str(self.stop_reason),
            "budget_limited": bool(self.stop_reason == "budget_limited"),
            "best_epoch": int(self.best_epoch),
            "stopped_epoch": int(self.stopped_epoch),
            "best_validation_wmape": float(self.best_validation_wmape),
            "seconds": round(float(self.seconds), 4),
            "peak_memory": dict(self.peak_memory),
            "randomness": dict(self.randomness),
            "fit_row_id_hash": str(self.fit_row_id_hash),
            "fit_group_hash": str(self.fit_group_hash),
            "transform_hash": str(self.transform_hash),
            "history_length": int(len(self.history)),
        }
        payload.update(self.extra)
        return payload


class FitTimer:
    """Context helper measuring wall time and peak memory around a fit."""

    def __init__(self) -> None:
        self.started = 0.0
        self.rss_before = 0
        self.tracemalloc_peak = 0
        self._active = False

    def __enter__(self) -> "FitTimer":
        self.rss_before = _maxrss_kb()
        self.started = time.perf_counter()
        tracemalloc.start()
        self._active = True
        return self

    def __exit__(self, *exc: object) -> None:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._active = False
        self.tracemalloc_peak = int(peak)

    @property
    def seconds(self) -> float:
        return float(time.perf_counter() - self.started)

    def report(self) -> dict[str, Any]:
        return peak_memory_report(self.tracemalloc_peak, self.rss_before, _maxrss_kb())


def wmape_of(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(wmape(actual, predicted))


def regression_epochs(
    *,
    n_rows: int,
    config: TrainConfig,
    epoch_step: Callable[[int, np.ndarray], float],
    epoch_score: Callable[[], float],
    state_getter: Callable[[], Any],
    state_setter: Callable[[Any], None],
    rng: np.random.Generator,
    on_epoch: Callable[[int, float], None] | None = None,
) -> tuple[WmapeEarlyStopping, list[dict[str, float]]]:
    """Shared epoch loop: shuffled mini-batches plus original-unit WMAPE stopping."""
    stopper = WmapeEarlyStopping(config.early_stopping_patience, config.min_delta)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config.max_epochs) + 1):
        order = rng.permutation(int(n_rows))
        losses: list[float] = []
        for start in range(0, len(order), int(config.batch_size)):
            losses.append(float(epoch_step(epoch, order[start:start + int(config.batch_size)])))
        score = float(epoch_score())
        history.append({
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)) if losses else math.inf,
            "validation_wmape": score,
        })
        improved = stopper.update(score, epoch, state_getter())
        if improved:
            state_setter(stopper.best_state)
        if on_epoch is not None:
            on_epoch(epoch, score)
        if stopper.fired:
            break
    return stopper, history
