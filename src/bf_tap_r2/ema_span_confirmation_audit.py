"""Independent confirmation arithmetic, native budgets and training-state audit."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from .ema_reference_artifacts import sha,write_new
from .ema_span_confirmation import DEVELOPMENT,CONFIRMATION,collect,manifest_context
from .v3_4_bags import group_safe_inner_folds


def independent_gate(frame,folds,vectors,summary):
    """Use fsum and direct Student-t arithmetic, independently of evaluate()."""
    seeds=[*DEVELOPMENT,*CONFIRMATION]
    if set(vectors)!=set(seeds) or set(folds)!=set(seeds):raise ValueError('Independent four-seed coverage required')
    gains=[];maximum=0.;cells=0
    def check(actual,expected):
        nonlocal maximum,cells
        if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):raise ValueError('Nonfinite independent arithmetic')
        difference=abs(float(actual)-float(expected));maximum=max(maximum,difference);cells+=1
        if difference>1e-9:raise ValueError('Independent confirmation arithmetic mismatch')
    y=frame.tap_time_len.to_numpy(float);yi=frame.tap_iron.to_numpy(float)
    def wmape(target,prediction):
        denominator=math.fsum(abs(float(v)) for v in target)
        if denominator<=0:raise ValueError('Invalid independent WMAPE denominator')
        return math.fsum(abs(float(a)-float(b)) for a,b in zip(target,prediction))/denominator
    for seed in seeds:
        fv=folds[seed];v=vectors[seed]
        if any(p.shape!=(len(frame),) or not np.isfinite(p).all() or (p<0).any() for p in v.values()):
            raise ValueError('Independent complete OOF vectors required')
        gain=50*(wmape(y,v['q75'])-wmape(y,v['candidate']));gains.append(gain)
        check(summary['gains'][str(seed)],gain)
        for name,key in [('Q75','q75'),('SHORT_SPAN','candidate')]:
            metric=summary['metrics']['tap_time_len'][name][str(seed)]
            check(metric['wmape'],wmape(y,v[key]))
            for fold in range(5):
                mask=fv==fold
                if not mask.any():raise ValueError('Independent complete fold coverage required')
                check(metric['by_fold'][str(fold)],wmape(y[mask],v[key][mask]))
            spout=frame.spout_no.to_numpy()
            for value in sorted(set(spout)):
                mask=spout==value;check(metric['by_spout'][str(value)],wmape(y[mask],v[key][mask]))
            check(summary['package_scores'][str(seed)][name],100-50*(wmape(y,v[key])+wmape(yi,v['iron'])))
    mean=math.fsum(gains)/4;sd=math.sqrt(math.fsum((g-mean)**2 for g in gains)/3);se=sd/2
    lcb=mean-float(t.ppf(.95,3))*se
    for name,value in [('mean',mean),('sd',sd),('se',se),('lcb95',lcb)]:check(summary['paired'][name],value)
    passed=all(g>0 for g in gains) and lcb>0
    if summary['paired']['n']!=4 or summary['four_seed_gate_passed'] is not passed:
        raise ValueError('Independent formal gate mismatch')
    return dict(four_seed_gate_passed=passed,seed_gains=dict(zip(map(str,seeds),gains)),
        seed_paired_lcb95=lcb,arithmetic_cells=cells,maximum_arithmetic_difference=maximum)


def audit(workspace,out):
    from .component_regularization_audit import verify_saved
    out=Path(out);manifest,frame,folds=manifest_context(workspace,out);spec=manifest['spec']
    summary=json.loads((out/'summary.json').read_text());vectors=collect(manifest,frame,folds)
    result=independent_gate(frame,folds,vectors,summary)
    totals={k:0 for k in spec['native_budget']};states=0;units=0;maximum_rss=0.
    for seed in CONFIRMATION:
        for fold in range(5):
            unit=out/f's{seed}-f{fold}';warm=json.loads((unit/'warm-complete.json').read_text())
            cold=json.loads((unit/'cold-complete.json').read_text())
            if cold['retained_states']!=44 or cold['warm_receipt_sha256']!=sha(unit/'warm-complete.json'):
                raise ValueError('Independent warm/cold unit count changed')
            if sha(unit/'reference/cold-complete.json')!=cold['reference_cold_receipt_sha256']:
                raise ValueError('Independent reference cold receipt changed')
            reference=json.loads((unit/'reference/cold-complete.json').read_text())
            if (reference['status']!='passed' or reference['retained_states']!=40
                    or reference['warm_capture_receipt_sha256']!=warm['reference_receipt_sha256']):
                raise ValueError('Independent complete reference cold audit required')
            for k,n in reference['native_counts'].items():totals[k]+=n
            maximum_rss=max(maximum_rss,warm['peak_rss_mib'],cold['cold_peak_rss_mib'])
            if maximum_rss>spec['max_worker_rss_mib']:raise ValueError('Independent memory gate failed')
            training=frame.loc[folds[seed]!=fold].reset_index(drop=True)
            from .data import FEATURES,TARGETS
            query=frame.loc[folds[seed]==fold,['sample_id','spout_no',*FEATURES]].reset_index(drop=True)
            inner=np.asarray(group_safe_inner_folds(training,seed=spec['training']['inner_seed'])['fold'])
            fitting=training.loc[inner!=0].reset_index(drop=True);calibration=training.loc[inner==0].reset_index(drop=True)
            for name,beta in [('OLD_EMA',spec['control_beta']),('SHORT_SPAN',spec['candidate_beta'])]:
                directory=unit/name;complete=json.loads((directory/'complete.json').read_text())
                component=json.loads((directory/'cold-complete.json').read_text())
                if (sha(directory/'complete.json')!=warm['component_receipts'][name]
                        or sha(directory/'cold-complete.json')!=cold['component_cold_receipts'][name]
                        or component['status']!='passed' or component['retained_states']!=2):
                    raise ValueError('Independent component warm/cold receipt changed')
                for k,n in complete['native_counts'].items():totals[k]+=n
                mechanisms=dict(spec['mechanisms'],ema_beta=beta)
                selected=verify_saved(directory/'selection.pt',fitting,fitting[['tap_time_len']].to_numpy(),
                    'EMA',spec['training'],mechanisms,calibration)
                refit=verify_saved(directory/'refit.pt',training,training[['tap_time_len']].to_numpy(),
                    'EMA',spec['training'],mechanisms,expected_epoch=selected.saved['trace']['selected_epoch'])
                for phase,model in [('selection',selected),('refit',refit)]:
                    if model.saved['trace']!=complete['checkpoints'][phase]['trace']:
                        raise ValueError('Independent native checkpoint trace changed')
                    request=calibration.drop(columns=list(TARGETS)) if phase=='selection' else query
                    actual=model.predict(request)
                    expected=np.load(directory/(phase+'-witness')/'observed.npy',allow_pickle=False)
                    if not np.array_equal(actual,expected):
                        raise ValueError('Native checkpoint and independently cold witness disagree')
                    if phase=='refit':
                        with np.load(directory/'predictions.npz',allow_pickle=False) as saved:
                            if not np.array_equal(actual,saved['prediction']):
                                raise ValueError('Native refit checkpoint and stored component predictions disagree')
                    states+=1
            states+=40;units+=1
    if totals!=spec['native_budget'] or states!=440 or units!=10:
        raise ValueError('Independent confirmation native budget/state coverage failed')
    manifest_context(workspace,out)
    payload=dict(status='passed',**result,manifest_sha256=sha(out/'manifest.json'),summary_sha256=sha(out/'summary.json'),
        complete_units=units,new_retained_states=states,native_counts=totals,maximum_worker_rss_mib=maximum_rss,
        G0='passed_saved_state_identity_cold_inference_partition_native_budget_and_arithmetic',
        G1='four_seed_gate_passed' if result['four_seed_gate_passed'] else 'four_seed_gate_failed',
        full_data_fits=0,packages=0,desktop_writes=0,agent_uploads=0,platform_queue_additions=0)
    write_new(out/'audit.json',payload)
    return payload
