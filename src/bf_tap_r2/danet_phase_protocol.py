"""Complete DANet pool and earned target-only confirmation budgets."""
from .data import TARGETS
from .danet_model import ARMS

DEV=(42,3407)
CONFIRM=(7777,12011)
ELIGIBLE=('DANET_LEARNED',)
CONTROLS={'DANET_LEARNED':'DANET_FIXED'}


def validate_eligible(pairs):
    pairs=[tuple(p) for p in pairs]
    if len(set(pairs))!=len(pairs) or any(len(p)!=2 or p[0] not in TARGETS or p[1] not in ELIGIBLE for p in pairs):
        raise ValueError('Invalid earned target/method pairs')
    return [(t,a) for t in TARGETS for a in ELIGIBLE if (t,a) in pairs]


def phase_tasks(phase,eligible=None):
    if phase=='development':
        if eligible is not None:raise ValueError('Development always covers the frozen complete pool')
        targets,seeds=TARGETS,DEV
    elif phase=='confirmation':
        if eligible is None:raise ValueError('Confirmation requires independently earned methods')
        eligible=validate_eligible(eligible)
        targets,seeds=tuple(t for t in TARGETS if any(p[0]==t for p in eligible)),CONFIRM
    else:raise ValueError('Unknown DANet phase')
    return [dict(target=t,seed=s,fold=f) for t in targets for s in seeds for f in range(5)]


def required_arms(target,phase,eligible=None):
    if target not in TARGETS:raise ValueError('Unknown target')
    if phase=='development':
        if eligible is not None:raise ValueError('No pre-evaluation development selection')
        return ARMS
    if phase!='confirmation' or eligible is None:raise ValueError('Explicit earned confirmation methods required')
    if not any(t==target for t,a in validate_eligible(eligible)):raise ValueError('Target has no earned candidate')
    return ARMS


def phase_limits(phase,eligible=None):
    tasks=phase_tasks(phase,eligible)
    return dict(pair_unit=len(tasks),estimator=4*len(tasks),optimizer=4*len(tasks))


def task_name(task):
    return f"{task['target']}-s{task['seed']}-f{task['fold']}"
