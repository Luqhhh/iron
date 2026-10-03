"""Frozen bookkeeping and independent seed decisions for fixed independent per-row/head Poisson1 bootstrap."""
import math
import numpy as np
from scipy.stats import t
from .tabm_swa_protocol import Ledger,sha,digest,write_new,verify_tree,child

BUDGETS={"engineering":{"state":2,"optimizer":2},"development":{"state":20,"optimizer":20},"confirmation":{"state":20,"optimizer":20}}
CANDIDATE="FIXED_HEAD_BOOTSTRAP"
CONTROL="HEAD_BOOTSTRAP_CONTROL"
SEEDS=(42,3407,271828,314159)
def expected_mechanisms(arm,phase):
    if arm==CONTROL:return {"bootstrap_rate":1.0,"bootstrap_seed_offset":1000003,"loss":"fixed_denominator_per_row_head_poisson_mse"}
    if arm==CANDIDATE:return {"bootstrap_rate":1.0,"bootstrap_seed_offset":1000003,"loss":"fixed_denominator_per_row_head_poisson_mse","bootstrap_draw_policy":"once_per_actual_fit_row_head"}
    raise ValueError("Unknown arm")

class FixedBootstrapPlan:
    def __init__(self,seed,ids,heads):
        import hashlib
        self.ids=np.asarray(list(ids),dtype=str)
        if not len(self.ids) or len(set(self.ids))!=len(self.ids) or heads<1:raise ValueError("Unique nonempty actual-fit row/head pool required")
        self.heads=int(heads);self.fit_digest=digest(self.ids.tolist());self.index={v:i for i,v in enumerate(self.ids)}
        self.weights=np.random.default_rng(int(seed)+1000003).poisson(1.0,(len(self.ids),self.heads)).astype('<i8')
        self.weights.setflags(write=False);self.plan_hash=hashlib.sha256(self.weights.tobytes()).hexdigest()
        self.hash=hashlib.sha256();self.counts={"draws":0,"weight_sum":0,"zero_weights":0,"zero_batches":0}
    def draw(self,ids,heads):
        ids=list(ids)
        if not ids or heads!=self.heads or len(set(ids))!=len(ids) or any(v not in self.index for v in ids):raise ValueError("Batch must be unique actual-fit IDs and unchanged heads")
        weights=self.weights[[self.index[v] for v in ids]].copy()
        self.hash.update(digest(ids).encode('ascii'));self.hash.update(weights.tobytes())
        self.counts["draws"]+=weights.size;self.counts["weight_sum"]+=int(weights.sum());self.counts["zero_weights"]+=int((weights==0).sum());self.counts["zero_batches"]+=int(not weights.any())
        return weights
    def certificate(self):
        return dict(self.counts,weight_and_order_sha256=self.hash.hexdigest(),plan_sha256=self.plan_hash,unique_poisson_draws=self.weights.size,fit_ids_digest=self.fit_digest,draw_policy="once_per_actual_fit_row_head")
    def save(self,path):
        from pathlib import Path
        with Path(path).open('xb') as f:np.savez(f,ids=self.ids,weights=self.weights)

def select_target(rows):
    order=("tap_time_len",)
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
