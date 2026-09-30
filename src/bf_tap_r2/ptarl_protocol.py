"""Append-only reservations for paired PTaRL partition procedures.

The shared ledger engine is unchanged. No official scheduling is exposed here.
"""
from pathlib import Path

from .data import TARGETS
from .rfm_protocol import ReservationLedger as Engine, file_hash, write_new


def phase_tasks(phase, eligible_targets=None):
    if phase == 'development':
        if eligible_targets is not None:
            raise ValueError('Development covers both targets')
        targets, seeds = TARGETS, (42, 3407)
    elif phase == 'confirmation':
        if eligible_targets is None:
            raise ValueError('Explicit earned confirmation targets required')
        targets = tuple(eligible_targets)
        if len(set(targets)) != len(targets) or not set(targets) <= set(TARGETS):
            raise ValueError('Invalid confirmation targets')
        targets = tuple(t for t in TARGETS if t in targets)
        seeds = (7777, 12011)
    else:
        raise ValueError('Unknown PTaRL phase')
    return [dict(target=t, seed=s, fold=f) for t in targets for s in seeds for f in range(5)]


def phase_limits(phase, eligible_targets=None):
    count = len(phase_tasks(phase, eligible_targets))
    return dict(pair_unit=count, optimizer=6*count, kmeans=2*count)


class ReservationLedger(Engine):
    @classmethod
    def create(cls, root, limits):
        if (set(limits) != {'pair_unit', 'optimizer', 'kmeans'}
                or any(type(v) is not int or v < 0 for v in limits.values())):
            raise ValueError('Invalid PTaRL reservation budget')
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        (root/'events').mkdir()
        (root/'lock').touch(exist_ok=False)
        write_new(root/'policy.json', dict(version=1, limits=limits))
        return cls.open(root, file_hash(root/'policy.json'))
