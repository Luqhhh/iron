"""Fixed paired tasks and count-only reservations; no official-data scheduler."""
from pathlib import Path

from .data import TARGETS
from .modernnca_ledger import ReservationLedger as Engine, file_hash, write_new


def phase_tasks(phase, eligible_targets=None):
    if phase == 'development':
        if eligible_targets is not None:
            raise ValueError('Development covers both targets')
        targets, seeds = TARGETS, (42, 3407)
    elif phase == 'confirmation':
        if (eligible_targets is None or len(set(eligible_targets)) != len(eligible_targets)
                or not set(eligible_targets) <= set(TARGETS)):
            raise ValueError('Explicit earned confirmation targets required')
        targets = tuple(t for t in TARGETS if t in eligible_targets)
        seeds = (7777, 12011)
    else:
        raise ValueError('Invalid phase')
    return [dict(target=t, seed=s, fold=f) for t in targets for s in seeds for f in range(5)]


def phase_limits(phase, eligible_targets=None):
    count = len(phase_tasks(phase, eligible_targets))
    return dict(pair_unit=count, estimator=4*count, optimizer=2*count)


def task_name(task):
    if (set(task) != {'target', 'seed', 'fold'} or task['target'] not in TARGETS
            or type(task['seed']) is not int or task['seed'] not in (42, 3407, 7777, 12011)
            or type(task['fold']) is not int or task['fold'] not in range(5)):
        raise ValueError('Invalid paired task')
    return f"{task['target']}-s{task['seed']}-f{task['fold']}"


class ReservationLedger(Engine):
    @classmethod
    def create(cls, root, limits):
        if (set(limits) != {'pair_unit', 'estimator', 'optimizer'}
                or any(type(v) is not int or v < 0 for v in limits.values())):
            raise ValueError('Invalid paired fit-count limits')
        return super().create(Path(root), limits)
