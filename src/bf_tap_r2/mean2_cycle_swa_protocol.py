"""Frozen bookkeeping and independent seed decisions for cycle-tail SWA."""
import math
import numpy as np
from scipy.stats import t
from .tabm_swa_protocol import Ledger,sha,digest,write_new,verify_tree,child

BUDGETS={"engineering":{"state":2,"optimizer":2},"development":{"state":20,"optimizer":20},"confirmation":{"state":20,"optimizer":20}}
CANDIDATE="SWA_CYCLE_MEAN2"
CONTROL="SWA_CYCLE_INIT42_CONTROL"
SEEDS=(42,3407,271828,314159)
def decide(gains,controls,confirmation):
    a,b=np.asarray(gains,float),np.asarray(controls,float);n=4 if confirmation else 2
    if a.shape!=(n,) or b.shape!=a.shape or not np.isfinite([a,b]).all():
        raise ValueError("Incomplete paired seed evidence")
    mean=math.fsum(a)/n;mechanism=math.fsum(a-b)/n
    lcb=float(mean-t.ppf(.95,n-1)*np.std(a,ddof=1)/np.sqrt(n)) if confirmation else None
    failures=[]
    if not all(v>0 for v in a):failures.append("nonpositive_seed_gain")
    if mechanism<=0:failures.append("nonpositive_mechanism_advantage")
    if confirmation and lcb<=0:failures.append("nonpositive_seed_lcb95")
    passed=not failures
    return dict(seed_gains=a.tolist(),control_gains=b.tolist(),mean_gain=mean,mechanism_mean=mechanism,seed_lcb95=lcb,
                selected_for_confirmation=[CANDIDATE] if passed and not confirmation else [],
                formal_promoted=bool(passed and confirmation),failure_reasons=failures,
                classification="formal_candidate_not_release_authorized" if passed and confirmation else "exploration_only_not_release_authorized",
                release_authorized=False,packages=0,uploads=0)

from .cycle_swa_protocol import MECHANISMS as CONTROL_MECHANISMS
MECHANISMS=dict(CONTROL_MECHANISMS)

def epoch_lr(epoch):
    if not isinstance(epoch,int) or epoch<1 or epoch>240:raise ValueError('Epoch outside frozen LR grid')
    if epoch<=80:return .001
    j=(epoch-81)%20
    return .0001+.0009*(1+math.cos(math.pi*j/19))/2

def cycle_epochs(selected):
    if selected not in range(100,241,20):raise ValueError('Not a completed cycle end')
    return list(range(100,selected+1,20))[-5:]

def verify_cycle_selection(trace,settings,actual):
    history=trace['history']
    if len(history)!=240 or trace['stopped_epoch']!=240:raise ValueError('Incomplete frozen cycle grid')
    best=float('inf');selected=0
    for number,row in enumerate(history,1):
        if row['epoch']!=number or abs(row['learning_rate']-epoch_lr(number))>1e-15:raise ValueError('Learning rate differs')
        end=number in range(100,241,20)
        if ('validation_mae' in row)!=end:raise ValueError('Calibration outside cycle ends')
        if end:
            if not math.isfinite(row['validation_mae']):raise ValueError('Nonfinite cycle metric')
            if row['validation_mae']<best-settings['min_delta']:best=row['validation_mae'];selected=number
    if selected!=trace['selected_epoch'] or not math.isfinite(actual) or abs(best-actual)>1e-6:raise ValueError('Cycle selected checkpoint differs')
    return selected

def verify_cycle_window(saved,witness):
    selected=saved['trace']['selected_epoch'];expected=cycle_epochs(selected)
    if witness['window']!=5 or witness['warmup_epochs']!=80 or witness['cycle_length']!=20 or witness['selected_epoch']!=selected or witness['epochs']!=expected or len(witness['states'])!=len(expected):raise ValueError('Sparse cycle window identity differs')
    maximum=0.
    for k,actual in saved['state'].items():
        if any(s.keys()!=saved['state'].keys() or s[k].shape!=actual.shape or s[k].dtype!=actual.dtype for s in witness['states']):raise ValueError('Sparse window state identity differs')
        values=[s[k].detach().cpu().numpy() for s in witness['states']];a=actual.detach().cpu().numpy()
        if actual.is_floating_point():
            mean=np.stack(values).astype(np.float64).mean(0).astype(a.dtype)
            np.testing.assert_allclose(a,mean,rtol=1e-6,atol=5e-7);maximum=max(maximum,float(np.max(np.abs(a-mean))))
        else:
            for v in values:np.testing.assert_array_equal(a,v)
    return maximum


def member_settings(control_settings):
    """Train a second RNG seed while retaining the original inner partition."""
    if control_settings.get("random_seed")!=42 or control_settings.get("inner_seed")!=42:
        raise ValueError("Unexpected control or inner seed")
    result=dict(control_settings)
    result["random_seed"]=3407
    return result


def mean2_predictions(control_ids,control,member_ids,member):
    a,b=np.asarray(control_ids,dtype=str),np.asarray(member_ids,dtype=str)
    c,m=np.asarray(control,dtype=float),np.asarray(member,dtype=float)
    if a.ndim!=1 or not len(a) or a.shape!=b.shape or c.shape!=a.shape or m.shape!=b.shape:
        raise ValueError("Incomplete member shape")
    if len(np.unique(a))!=len(a) or not np.array_equal(a,b):
        raise ValueError("Member IDs differ; cross-split/reordered averaging forbidden")
    if not np.isfinite([c,m]).all():
        raise ValueError("Nonfinite ensemble member")
    result=.5*c+.5*m
    if not np.isfinite(result).all():raise ValueError("Nonfinite ensemble mean")
    return result
