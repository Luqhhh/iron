"""Complete paired-unit coverage, reservation and isolated-column evidence.

No scheduling, reference loading or model fitting occurs in this collector.
"""
import json
from pathlib import Path

import numpy as np
import yaml

from .candidate_tiers import classify_candidates
from .data import FEATURES, TARGETS
from .modernnca_execution import audit_unit
from .modernnca_protocol import task_name
from .modernnca_model import ARMS
from .modernnca_protocol import ReservationLedger, phase_tasks, phase_limits
from .modernnca_scoring import score_seed, decide_development, decide_confirmation, wmape


def collect_audited_phase(workspace, output, phase, frame, folds, current, parent,
                          settings, unit_anchors, ledger_policy_sha256, development_records=None):
    if phase=='development':
        if development_records is not None:
            raise ValueError('Development cannot inherit selection')
        eligible=None
    elif phase=='confirmation':
        if development_records is None:
            raise ValueError('Audited development required before confirmation')
        eligible=decide_development(development_records)['eligible_targets']
    else:
        raise ValueError('Invalid phase')
    tasks=phase_tasks(phase,eligible)
    names={task_name(t) for t in tasks}
    output,workspace=Path(output),Path(workspace)
    if not tasks or set(unit_anchors) != names or {p.name for p in (output/'units').iterdir()} != names:
        raise ValueError('Incomplete/extra/empty paired phase units')
    ledger=ReservationLedger.open(output/'ledger',ledger_policy_sha256)
    limits=phase_limits(phase,eligible)
    counts=ledger.inspect()
    if (ledger.limits != limits or counts['started'] != limits or counts['completed'] != limits
            or any(counts[s][k] for s in ('failed','incomplete') for k in limits)):
        raise ValueError('Failed/incomplete/unexpected phase reservations')
    n=len(frame)
    y={t:frame[t].to_numpy(float) for t in TARGETS}
    members,audits={},[]
    expected_events=set()
    for task in tasks:
        target,seed,fold=task['target'],task['seed'],task['fold']
        name=task_name(task)
        fv=np.asarray(folds[seed])
        if fv.shape!=(n,) or not np.issubdtype(fv.dtype,np.integer) or set(fv)!=set(range(5)):
            raise ValueError('Complete five-fold identity required')
        mask=fv==fold
        train=frame.loc[~mask].reset_index(drop=True)
        query=frame.loc[mask,['sample_id','spout_no',*FEATURES]].reset_index(drop=True)
        predictions,audit=audit_unit(output/'units'/name,task,train,query,settings,
            expected_sha256=unit_anchors[name],ledger_root=ledger.root)
        if audit['ledger_policy_sha256'] != ledger_policy_sha256:
            raise ValueError('Unit belongs to a different phase ledger')
        for arm in ARMS:
            vector=members.setdefault((target,arm,seed),np.full(n,np.nan))
            if np.isfinite(vector[mask]).any():raise ValueError('Repeated OOF coverage')
            vector[mask]=predictions[arm]
        audits.append(audit)
        expected_events.add(('pair_unit',(name,)))
        for arm in ARMS:
            for stage in ('selector','refit'):
                expected_events.add(('estimator',(name,f'{arm}_{stage}')))
                if arm=='LEARNED_ENCODER':
                    expected_events.add(('optimizer',(name,f'{arm}_{stage}')))
    observed=set()
    for path in (ledger.root/'events').glob('*.started.json'):
        r=json.loads(path.read_text())
        observed.add((r['kind'],tuple(r['key'])))
    if observed != expected_events:
        raise ValueError('Unaccounted or missing phase reservation starts')
    targets=list(TARGETS) if phase=='development' else eligible
    seeds=[42,3407] if phase=='development' else [7777,12011]
    records,metrics=[],{}
    for target in targets:
        metrics[target]={route:{} for route in ('CURRENT','LEARNED_ENCODER')}
        for seed in seeds:
            values={a:members[target,a,seed] for a in ARMS}
            records.extend(score_seed(y,folds[seed],current[seed],parent[seed],values,target=target,seed=seed))
            for route,prediction in [('CURRENT',current[seed][target]),
                ('LEARNED_ENCODER',.8*current[seed][target]+.2*values['LEARNED_ENCODER'])]:
                metrics[target][route][str(seed)]=dict(wmape=wmape(y[target],prediction),
                    by_fold={str(f):wmape(y[target][folds[seed]==f],prediction[folds[seed]==f]) for f in range(5)},
                    by_spout={str(s):wmape(y[target][frame.spout_no.to_numpy()==s],
                        prediction[frame.spout_no.to_numpy()==s]) for s in sorted(frame.spout_no.unique())})
    if phase=='development':
        decision=decide_development(records)
        tier_spec=dict(split_seeds=seeds,folds=5,candidates={t:['LEARNED_ENCODER'] for t in targets},
            reference_by_target={t:'CURRENT' for t in targets},tie_preference_by_target={t:['LEARNED_ENCODER'] for t in targets})
        tiers=classify_candidates(metrics,tier_spec,yaml.safe_load((workspace/'configs/candidate_tiers.yaml').read_text()))
    else:
        decision=decide_confirmation(development_records,records)
        tiers=None
    return dict(status='passed',phase=phase,records=records,decision=decision,tiers=tiers,
        counts=counts,audited_units=audits,new_audit_fits=0,release_authorized=False)
