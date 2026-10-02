"""Frozen bookkeeping and independent seed decisions for time-MAE SWA."""
import math
import numpy as np
from scipy.stats import t
from .tabm_swa_protocol import Ledger,sha,digest,write_new,verify_tree,child

BUDGETS={"engineering":{"state":2,"optimizer":2},"development":{"state":20,"optimizer":20},"confirmation":{"state":20,"optimizer":20}}
CANDIDATE="SWA_TIME_MAE"
CONTROL="SWA_MSE_CONTROL"
SEEDS=(42,3407,271828,314159)
def expected_mechanisms(arm):
    if arm==CONTROL:return {"uniform_epoch_window":10}
    if arm==CANDIDATE:return {"uniform_epoch_window":10,"time_loss":"mae","iron_loss":"mse","loss_coefficients":[.5,.5]}
    raise ValueError("Unknown arm")
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
