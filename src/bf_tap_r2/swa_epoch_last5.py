"""Fixed recency weighting of existing consecutive SWA states; no optimizer or fitting."""
import math
import numpy as np
from scipy.stats import t
POOL=('EPOCH_LAST5',)

def reweighted_state(witness,mode):
    import torch
    if mode not in POOL:raise ValueError('Unknown frozen candidate')
    selected=witness["selected_epoch"]
    if not isinstance(selected,int) or selected<1:raise ValueError("Invalid consecutive window epoch")
    expected=list(range(max(1,selected-9),selected+1))
    states=witness["states"]
    if witness["window"]!=10 or witness["epochs"]!=expected or len(states)!=len(expected):
        raise ValueError("Consecutive epoch window differs")
    keys=states[0].keys()
    for state in states:
        if state.keys()!=keys:raise ValueError('State window keys differ')
        for key in keys:
            v=state[key];first=states[0][key]
            if v.shape!=first.shape or v.dtype!=first.dtype:raise ValueError('State window dtype or shape differs')
            if v.is_floating_point():
                if not torch.isfinite(v).all():raise ValueError('Nonfinite state window')
            elif not torch.equal(v,first):raise ValueError('Integer buffer differs across window')
    chosen=states[-5:]
    w=torch.ones(len(chosen),dtype=torch.float64)
    w=w/w.sum();result={}
    for key in keys:
        first=chosen[0][key]
        if first.is_floating_point():
            values=torch.stack([s[key].detach().cpu().to(torch.float64) for s in chosen])
            result[key]=(values*w.reshape((-1,)+(1,)*first.ndim)).sum(0).to(first.dtype)
        else:result[key]=first.detach().cpu().clone()
    return result

def replay_candidates(model,witness,frame,column=1):
    from .tabm_swa_audit import verify_window
    verify_window(model.saved,witness)
    original={k:v.detach().clone() for k,v in model.model_.state_dict().items()};result={}
    try:
        for mode in POOL:
            model.model_.load_state_dict(reweighted_state(witness,mode));model.model_.eval()
            result[mode]=model.predict(frame)[:,column]
            if not np.isfinite(result[mode]).all():raise ValueError('Nonfinite candidate prediction')
    finally:model.model_.load_state_dict(original);model.model_.eval()
    return result

def exploration_decision(gains,controls):
    b=np.asarray(controls,float)
    if set(gains)!=set(POOL) or b.shape!=(4,) or not np.isfinite(b).all():raise ValueError('Incomplete frozen candidate pool')
    rows={};eligible=[]
    for mode in POOL:
        a=np.asarray(gains[mode],float)
        if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('Four complete seed gains required')
        mean=math.fsum(a)/4;mechanism=math.fsum(a-b)/4
        lcb=mean-float(t.ppf(.95,3))*math.sqrt(math.fsum((v-mean)**2 for v in a)/3)/2
        failures=[]
        if not all(a>0):failures.append('nonpositive_seed_gain')
        if mean<=0:failures.append('nonpositive_mean_gain')
        if mechanism<=0:failures.append('nonpositive_mechanism_mean')
        if lcb<=0:failures.append('nonpositive_descriptive_LCB95')
        rows[mode]={'seed_gains':a.tolist(),'mean_gain':mean,'mechanism_mean':mechanism,'seed_lcb95_descriptive':lcb,'failure_reasons':failures}
        if mean>0 and mechanism>0:eligible.append(mode)
    selected=sorted(eligible,key=lambda v:(-rows[v]['mean_gain'],POOL.index(v)))[:1]
    return {'candidates':rows,'control_gains':b.tolist(),'classification':'post_selection_exploration_only_not_release_authorized',
            'exploration_selected':selected,'formal_promoted':False,'release_authorized':False}