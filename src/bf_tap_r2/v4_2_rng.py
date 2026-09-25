"""Explicit, stage-scoped randomness control for the V4.2 neural lines.

Why this module exists
----------------------
The first V4.2 execution created each network *before* any seed was set::

    stage1_net = _make_network(...)      # torch.randn, no explicit seed
    stage1 = self._run_epochs(stage1_net, ...)   # only here: torch.manual_seed

``_make_network`` draws ``torch.randn`` for the feature logits, thresholds and
leaf weights, so the stage-1 initialisation was controlled by whatever global
torch RNG state the worker process happened to have.  Setting a seed *after* the
weights exist does not reset them.  The selected epoch of stage 1 then decides
how many epochs stage 2 runs, so the contamination propagated into the reported
score.  A ``training_seed: 42`` log line alone therefore did **not** prove that
initialisation was controlled by 42.

What this module guarantees
---------------------------
* every network is built inside an explicit ``torch_seed_context`` so the
  initialisation seed is applied *before* the first random draw;
* both stages use the one registered training seed for their initialisation, as
  the repair task book requires, and the two stages have explicitly registered,
  independent batch-order streams;
* no seed is derived from a worker id, a process id, a task-scheduling index or
  Python's randomised ``hash()``; the batch-order seeds are plain arithmetic
  offsets from the registered training seed;
* the ambient torch RNG state is saved and restored around each stage, so an
  unrelated task executed earlier in the same process cannot move the model;
* an order-sensitive SHA-256 over the parameter state gives a comparable
  ``initialisation_hash`` / ``state_hash`` that can be checked across new
  processes and repeated in-process fits.

Determinism is reported, never overstated: identical results are expected for a
fixed source revision, dependency set, thread count and data.  Floating-point
reduction order depends on the thread count, so a different ``torch_threads``
setting is a different numeric environment and is recorded as such.
"""
from __future__ import annotations

import hashlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BATCH_ORDER_SEED_OFFSET",
    "V42RandomnessPlan",
    "dependency_identity",
    "initialisation_hash",
    "parameter_state_hash",
    "source_digest",
    "torch_seed_context",
]

#: Plain arithmetic offset, so the batch-order stream is independent of the
#: initialisation seed without depending on any process-local value.  The offset
#: is registered here and repeated verbatim in the fit ledger.
BATCH_ORDER_SEED_OFFSET = 1_000_003

#: Torch seeds are taken modulo ``2 ** 32 - 1`` exactly as the original code did.
_TORCH_SEED_MODULUS = 2**32 - 1


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("The V4.2 randomness control requires torch") from exc
    return torch


@dataclass(frozen=True)
class V42RandomnessPlan:
    """The registered randomness identity of one V4.2 neural fit.

    ``init_seed`` is reused for *both* stages on purpose: the repair task book
    forbids choosing a stage-2 seed by trying several and keeping the better
    one.  The batch-order streams are separate and derived by fixed arithmetic.
    """

    training_seed: int
    inner_split_seed: int
    torch_threads: int | None = None

    @classmethod
    def from_training_seed(
        cls, training_seed: int, *, inner_split_seed: int, torch_threads: int | None = None
    ) -> "V42RandomnessPlan":
        return cls(
            training_seed=int(training_seed),
            inner_split_seed=int(inner_split_seed),
            torch_threads=None if torch_threads is None else int(torch_threads),
        )

    @property
    def init_seed(self) -> int:
        return int(self.training_seed)

    @property
    def batch_order_seed_stage1(self) -> int:
        return int(self.training_seed) + BATCH_ORDER_SEED_OFFSET

    @property
    def batch_order_seed_stage2(self) -> int:
        return int(self.training_seed) + BATCH_ORDER_SEED_OFFSET + 1

    def stage_init_seed(self, stage: str) -> int:
        key = _stage_key(stage)
        # Both stages deliberately share the registered initialisation seed.
        return int(self.init_seed)

    def stage_batch_order_seed(self, stage: str) -> int:
        key = _stage_key(stage)
        if key == "stage1":
            return self.batch_order_seed_stage1
        return self.batch_order_seed_stage2

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": "V42RandomnessPlan",
            "training_seed": int(self.training_seed),
            "init_seed": int(self.init_seed),
            "init_seed_shared_across_stages": True,
            "batch_order_seed_stage1": int(self.batch_order_seed_stage1),
            "batch_order_seed_stage2": int(self.batch_order_seed_stage2),
            "batch_order_seed_offset": int(BATCH_ORDER_SEED_OFFSET),
            "batch_order_rng": "numpy_default_rng",
            "inner_split_seed": int(self.inner_split_seed),
            "torch_threads": self.torch_threads,
            "seed_derivation": "arithmetic_offset_never_process_local",
            "note": (
                "Initialisation happens inside torch_seed_context before the first "
                "random draw; batch order uses a separate numpy Generator."
            ),
        }


def _stage_key(stage: str) -> str:
    key = str(stage).strip().lower()
    if key in {"stage1", "1", "u", "early_stopping"}:
        return "stage1"
    if key in {"stage2", "2", "refit", "full"}:
        return "stage2"
    raise ValueError(f"Unknown V4.2 training stage: {stage!r}")


@contextmanager
def torch_seed_context(seed: int):
    """Run a block with an explicit torch seed, then restore the ambient state.

    Saving and restoring means an earlier unrelated task in the same process
    cannot move this network, and this network cannot move a later one.
    """
    torch = _torch()
    saved = torch.random.get_rng_state()
    cuda_saved = None
    cuda_available = bool(getattr(torch.cuda, "is_available", lambda: False)())
    if cuda_available:
        cuda_saved = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(int(seed) % _TORCH_SEED_MODULUS)
        yield
    finally:
        torch.random.set_rng_state(saved)
        if cuda_available and cuda_saved is not None:
            torch.cuda.set_rng_state_all(cuda_saved)


def _digest_tensors(items: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in items:
        array = tensor.detach().to("cpu").contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(str(tuple(int(v) for v in array.shape)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(array.numpy().tobytes())
        digest.update(b"\x01")
    return digest.hexdigest()


def initialisation_hash(net) -> str:
    """Order-sensitive SHA-256 of a freshly constructed network's state.

    Comparable across new processes.  Covers every parameter and persistent
    buffer, so it detects a seed that did not actually reach ``torch.randn``.
    """
    state = net.state_dict()
    items = [(name, state[name]) for name in sorted(state)]
    if not items:
        raise ValueError("Cannot hash an empty network state")
    return _digest_tensors(items)


#: Alias with the name used when hashing a fitted model.
parameter_state_hash = initialisation_hash


def dependency_identity() -> dict[str, Any]:
    """The numeric dependency set a fit's reproducibility is conditional on."""
    from importlib import metadata

    packages = (
        "torch", "numpy", "pandas", "catboost", "lightgbm",
        "scikit-learn", "sympy", "PyYAML",
    )
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = str(metadata.version(name))
        except Exception:  # noqa: BLE001 - an absent optional package is recorded as null
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "note": (
            "Identical results are conditional on this dependency set and on the "
            "recorded torch thread count; a different thread count changes "
            "floating-point reduction order and is a different numeric environment."
        ),
    }


def source_digest(paths: Sequence[Path | str]) -> str:
    """SHA-256 over the V4.2 model/training source files, in sorted order."""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(Path(path).name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(Path(path).read_bytes())
        digest.update(b"\x01")
    return digest.hexdigest()


def default_source_files() -> tuple[Path, ...]:
    """The model and training sources whose content defines the N-line identity."""
    here = Path(__file__).resolve().parent
    names = (
        "v4_2_rng.py",
        "v4_2_n_node.py",
        "v4_2_prep.py",
        "v4_2_train.py",
        "v3_6_networks.py",
    )
    return tuple(here / name for name in names)


def source_digest_payload() -> dict[str, Any]:
    """A small, loggable description of the model/training source identity."""
    files = default_source_files()
    return {
        "files": [f.name for f in files],
        "sha256": source_digest(files),
    }


def assert_mapping_finite(payload: Mapping[str, Any]) -> None:
    """Small guard used by the identity builders: reject non-finite markers."""
    import math

    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Identity field {key!r} is not finite")
