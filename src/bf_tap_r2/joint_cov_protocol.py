"""Frozen bookkeeping and independent seed decisions for fixed half-shrinkage joint covariance MSE."""
import math
import numpy as np
from scipy.stats import t
from .tabm_swa_protocol import Ledger,sha,digest,write_new,verify_tree,child

BUDGETS={"engineering":{"state":2,"optimizer":2},"development":{"state":20,"optimizer":20},"confirmation":{"state":20,"optimizer":20}}
CANDIDATE="JOINT_COV_MSE"
CONTROL="NATIVE_MSE_CONTROL"
SEEDS=(42,3407,271828,314159)
def expected_mechanisms(arm,phase):
    if arm==CONTROL:return {"target_metric_weight":0.0}
    if arm==CANDIDATE:return {"joint_cov_shrinkage":0.5,"loss":"half_quadratic_precision_per_head"}
    raise ValueError("Unknown arm")

def covariance_metric(y):
    y=np.asarray(y,dtype=np.float64)
    if y.ndim!=2 or y.shape[1]!=2 or len(y)<2 or not np.isfinite(y).all():
        raise ValueError("Finite nonconstant row/two-target fit matrix required")
    std=y.std(0)
    if not np.isfinite(std).all() or not (std>0).all():raise ValueError("Constant target")
    z=(y-y.mean(0))/std
    c=.5*(z.T@z/len(z))+.5*np.eye(2)
    if not np.isfinite(c).all() or np.linalg.eigvalsh(c).min()<.5-1e-12:raise ValueError("Invalid shrinkage matrix")
    return c,np.linalg.inv(c)

def select_target(rows):
    order=("tap_time_len","tap_iron")
    eligible=[t for t in order if rows[t]["selected_for_confirmation"]]
    return max(eligible,key=lambda t:rows[t]["mean_gain"]) if eligible else None

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
