"""Replay the frozen V3.1 S1 schedule without its run ledger.

``v3_1_sampler.sample_s1`` mutates six parent trials loaded by ``load_centers``
from ``local/runs/round2-v3-local-search/coarse-catboost-r1/fit_ledger.jsonl``.
That ledger is not on this machine, so ``load_centers`` raises and the whole
V3.1 schedule -- which supplies L1's member pool -- looked unrecoverable.

It is recoverable.  ``v3_run.trial_details`` shows the ledger carries nothing
but trial definitions, and all six parents are V3 first-batch trials, which
``v3_local_search.sample_trials`` reproduces deterministically from the frozen
search space.  So the parents can be looked up instead of read.

Nothing here writes to ``local/runs``: no run record is fabricated, and the
reconstruction is a lookup in the frozen schedule rather than an invention.

The assumption underneath -- that the original coarse run used the default
sample seed -- has since been checked against the real ledgers, which were
recovered separately.  All six centres and all 320 S1 trials are identical to
the originals, parameter for parameter.  See
``docs/round2_next_phase/RESULTS.md`` section 9.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .v3_1_sampler import IRON_CENTERS, TIME_CENTERS, sample_s1
from .v3_local_search import sample_trials

V3_CONFIG = Path("configs/round2_v3/experiment.yaml")

# docs/round2_v3_1/RESULTS.md, the L1 member pool.
L1_IRON_MEMBERS: tuple[str, ...] = (
    "v31-s1-expr-iron-0018",
    "v31-s1-expr-iron-0012",
)
L1_TIME_MEMBERS: tuple[str, ...] = (
    "v31-s1-time-0021-0050",
    "v31-s1-time-0021-0031",
    "v31-s1-time-0021-0039",
    "v31-s1-time-0021-0002",
)


def reconstructed_centers(root: Path | str) -> dict[str, dict[str, Any]]:
    """The six V3.1 parents, resolved from the frozen V3 schedule."""
    root = Path(root)
    spec = yaml.safe_load((root / V3_CONFIG).read_text(encoding="utf-8"))
    by_id = {trial["trial_id"]: trial for trial in sample_trials(spec)}
    wanted = set(TIME_CENTERS) | set(IRON_CENTERS)
    missing = wanted - set(by_id)
    if missing:
        raise ValueError(f"Centres absent from the frozen V3 schedule: {sorted(missing)}")
    return {key: dict(by_id[key]) for key in wanted}


def v31_s1_trials(root: Path | str) -> list[dict[str, Any]]:
    """The full frozen V3.1 S1 schedule, replayed without its ledger."""
    return sample_s1(root, centers=reconstructed_centers(root))


def v31_trial_by_id(root: Path | str, trial_id: str) -> dict[str, Any]:
    for trial in v31_s1_trials(root):
        if trial["trial_id"] == trial_id:
            return trial
    raise ValueError(f"{trial_id} is absent from the replayed V3.1 schedule")
