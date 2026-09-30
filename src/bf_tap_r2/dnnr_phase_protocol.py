"""Frozen DNNR pool, earned confirmation arms and operation budgets."""
from .data import TARGETS
from .dnnr_model import ARMS

DEV = (42, 3407)
CONFIRM = (7777, 12011)
ELIGIBLE = ('DNNR_FIXED', 'DNNR_LEARNED')
CONTROLS = {'DNNR_FIXED': 'KNN_FIXED', 'DNNR_LEARNED': 'DNNR_FIXED'}


def validate_eligible(pairs):
    pairs = [tuple(p) for p in pairs]
    if len(set(pairs)) != len(pairs) or any(len(p) != 2 or p[0] not in TARGETS or p[1] not in ELIGIBLE for p in pairs):
        raise ValueError('Invalid earned target/method pairs')
    return [(t,a) for t in TARGETS for a in ELIGIBLE if (t,a) in pairs]


def phase_tasks(phase, eligible=None):
    if phase == 'development':
        if eligible is not None:
            raise ValueError('Development always covers the frozen complete pool')
        targets, seeds = TARGETS, DEV
    elif phase == 'confirmation':
        if eligible is None:
            raise ValueError('Confirmation requires independently earned methods')
        eligible = validate_eligible(eligible)
        targets, seeds = tuple(t for t in TARGETS if any(p[0] == t for p in eligible)), CONFIRM
    else:
        raise ValueError('Unknown DNNR phase')
    return [dict(target=t, seed=s, fold=f) for t in targets for s in seeds for f in range(5)]


def required_arms(target, phase, eligible=None):
    if target not in TARGETS:
        raise ValueError('Unknown target')
    if phase == 'development':
        if eligible is not None:
            raise ValueError('No development selection supplied before evaluation')
        return ARMS
    if phase != 'confirmation' or eligible is None:
        raise ValueError('Explicit earned confirmation methods required')
    needed = set()
    for t, a in validate_eligible(eligible):
        if t == target:
            needed.update((a, CONTROLS[a]))
    if not needed:
        raise ValueError('Target has no earned candidate')
    return tuple(a for a in ARMS if a in needed)


def phase_limits(phase, eligible=None):
    tasks = phase_tasks(phase, eligible)
    counts = dict(pair_unit=len(tasks), estimator=0, derivative_bank=0, metric_epoch=0)
    for task in tasks:
        arms = required_arms(task['target'], phase, eligible)
        counts['estimator'] += 2*len(arms)
        counts['derivative_bank'] += 2*sum(a != 'KNN_FIXED' for a in arms)
        counts['metric_epoch'] += 2*int('DNNR_LEARNED' in arms)
    return counts


def task_name(task):
    return f"{task['target']}-s{task['seed']}-f{task['fold']}"
