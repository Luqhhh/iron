"""compact iron MAE fixed-budget bookkeeping and isolated iron evaluation; no neural imports."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import hashlib
import json
import time
import numpy as np
from scipy.stats import t

BUDGETS={"engineering":{"state":4,"optimizer":4},"development":{"state":40,"optimizer":40},"confirmation":{"state":40,"optimizer":40}}
SEEDS=(42,3407,7777,12011)
ARMS=("MSE_CONTROL","COMPACT_MAE")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def write_new(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.flush()
        import os
        os.fsync(f.fileno())
def child(root,name):
    root=Path(root).resolve(); rel=Path(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts: raise ValueError("Path escapes reference root")
    path=root/rel
    if not path.resolve().is_relative_to(root): raise ValueError("Symlink escapes reference root")
    return path

def verify_tree(root,hashes):
    for name,h in hashes.items():
        if sha(child(root,name))!=h: raise ValueError("Frozen dependency changed: "+name)

class Ledger:
    def __init__(self,root,budgets=None):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.budgets=BUDGETS if budgets is None else budgets
        self.path=self.root/"fits.jsonl"
    @contextmanager
    def locked(self):
        with (self.root/"fits.lock").open("a") as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(f,fcntl.LOCK_UN)
    def events(self):
        return [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
    def _append(self,e):
        import os
        with self.path.open("a") as f:
            f.write(json.dumps(dict(e,time=time.time()),sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())
    def reserve(self,phase,kind,key):
        with self.locked():
            es=self.events(); match=[e for e in es if e["phase"]==phase and e["kind"]==kind]
            if any(e["key"]==key for e in match): raise ValueError("Consumed reservation cannot retry")
            if sum(e["event"]=="reserved" for e in match)>=self.budgets[phase][kind]: raise ValueError("Frozen budget exhausted")
            self._append(dict(event="reserved",phase=phase,kind=kind,key=key))
    def complete(self,phase,kind,key):
        with self.locked():
            es=[e for e in self.events() if (e["phase"],e["kind"],e["key"])==(phase,kind,key)]
            if len(es)!=1 or es[0]["event"]!="reserved": raise ValueError("Completion lacks one reservation")
            self._append(dict(event="completed",phase=phase,kind=kind,key=key))
    def counts(self,phase):
        es=[e for e in self.events() if e["phase"]==phase]
        return {event:{kind:sum(e["event"]==event and e["kind"]==kind for e in es) for kind in ("state","optimizer")} for event in ("reserved","completed")}

def decision(gains,controls,*,confirmation):
    a,b=np.asarray(gains,float),np.asarray(controls,float)
    if a.shape!=(4 if confirmation else 2,) or b.shape!=a.shape or not np.isfinite([a,b]).all(): raise ValueError("Incomplete paired seed evidence")
    mechanism=float((a-b).mean()); mean=float(a.mean())
    lcb=float(mean-t.ppf(.95,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if confirmation else None
    positive=bool((a>0).all()); passed=positive and mechanism>0 and (not confirmation or lcb>0)
    reasons=[]
    if not positive: reasons.append("nonpositive_seed_gain")
    if mechanism<=0: reasons.append("nonpositive_mechanism_advantage")
    if confirmation and lcb<=0: reasons.append("nonpositive_seed_lcb95")
    return dict(mean_gain=mean,mechanism_mean=mechanism,seed_gains=a.tolist(),control_gains=b.tolist(),seed_lcb95=lcb,
        selected_for_confirmation=["COMPACT_MAE"] if passed and not confirmation else [],formal_promoted=bool(passed and confirmation),
        classification="formal_candidate_not_release_authorized" if passed else "exploration_only_not_release_authorized",
        failure_reasons=reasons,release_authorized=False,packages=0,uploads=0,fold_criteria="descriptive_only")

def bootstrap(y,bases,endpoints,spouts,reps=2000):
    y=np.asarray(y,float); diff=np.mean([np.abs(y-b)-np.abs(y-p) for b,p in zip(bases,endpoints)],axis=0)
    rng=np.random.default_rng(61001); answer={"sampling_only_not_platform_forecast":True,"test_rows":322,"reps":reps}
    spouts=np.asarray(spouts)
    for scheme in ("uniform","spout_stratified"):
        values=[]
        for _ in range(reps):
            if scheme=="uniform": ix=rng.integers(len(y),size=322)
            else:
                groups=[np.flatnonzero(spouts==s) for s in np.unique(spouts)]
                counts=np.floor(np.asarray([len(g) for g in groups])*322/len(y)).astype(int); counts[-1]+=322-counts.sum()
                ix=np.concatenate([rng.choice(g,int(n),replace=True) for g,n in zip(groups,counts)])
            values.append(50*diff[ix].sum()/y[ix].sum())
        answer[scheme]={"median":float(np.median(values)),"q025":float(np.quantile(values,.025)),"q975":float(np.quantile(values,.975))}
    return answer
