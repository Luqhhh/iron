"""Read-only final reconciliation for the frozen nested residual batch.

This is deliberately independent of the trainer and imports no training module.
It refuses to inspect prediction arrays until all actual child exits are closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def expected_tasks():
    names=[]
    for seed in (42,3407):
        for fold in range(5):
            for inner in range(5):
                for stage in ('base','base-cold'):
                    names.append(f'{len(names):03d}-{stage}-s{seed}-f{fold}-i{inner}')
            for stage in ('heads','heads-cold'):
                names.append(f'{len(names):03d}-{stage}-s{seed}-f{fold}-i-1')
    names.append('120-report-s42-f0-i-1')
    return names


def verify_events(terminal):
    if terminal.get('status')!='passed' or not terminal.get('dependencies_unchanged'):
        raise ValueError('Actual successful controller terminal required')
    events=terminal['events']
    if [e['task'] for e in events]!=expected_tasks():
        raise ValueError('Incomplete, duplicated or reordered actual child inventory')
    for event in events:
        if (event['exit_code']!=0 or not math.isfinite(event['peak_rss_mib'])
                or not 0<event['peak_rss_mib']<=1536):
            raise ValueError('Actual child exit or memory gate failed')
    times=[e['completed_ns'] for e in events]
    if times!=sorted(times):
        raise ValueError('Actual completion order differs')
    return events


def verify_optimizer_receipts(complete,fit_start,optimizer_starts,traces):
    if complete['constructors']!=['selection','refit'] or set(traces)!={'selection','refit'}:
        raise ValueError('Two distinct optimizer roles required')
    last=fit_start['time_ns']
    if fit_start['identity']!=complete['identity'] or fit_start['fit_ids']!=complete['fit_ids']:
        raise ValueError('Estimator identity differs')
    for role in ('selection','refit'):
        receipt=optimizer_starts[role]
        if receipt['identity']!=complete['identity'] or receipt['role']!=role or receipt['time_ns']<=last:
            raise ValueError('Optimizer identity/order differs')
        last=receipt['time_ns'];trace=traces[role]
        steps=complete['steps'][role]
        if steps!=trace['updates'] or steps!=sum(r['updates'] for r in trace['history']):
            raise ValueError('Actual optimizer steps disagree with saved native trajectory')
    return sum(complete['steps'].values())


def scalar_score(y,iron,time_prediction):
    wi=math.fsum(abs(float(a)-float(b)) for a,b in zip(y[:,0],iron))/math.fsum(abs(float(a)) for a in y[:,0])
    wt=math.fsum(abs(float(a)-float(b)) for a,b in zip(y[:,1],time_prediction))/math.fsum(abs(float(a)) for a in y[:,1])
    return 100.-50.*(wi+wt)


def remember_truth(truth,training,plan):
    if (training.sample_id.tolist()!=plan['training_ids']
            or set(plan['training_ids'])&set(plan['query_ids'])):
        raise ValueError('Frozen training/query identities differ')
    for identifier,a,b in training[['sample_id','tap_iron','tap_time_len']].itertuples(index=False,name=None):
        value=(float(a),float(b))
        if not all(math.isfinite(v) for v in value) or (identifier in truth and truth[identifier]!=value):
            raise ValueError('Conflicting or nonfinite frozen training labels')
        truth[identifier]=value


def run(root,output):
    root=Path(root).resolve();output=Path(output).resolve();execution=root/'execution'
    if output.exists() or output.parent!=execution:
        raise ValueError('Fresh append-only audit path in original execution directory required')
    terminal=read(execution/'terminal.json');events=verify_events(terminal)
    # All accesses to saved predictions/labels happen after the terminal gate.
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    manifest=read(root/'manifest.json');report=read(root/'report.json')
    if (sha(root/'report.json')!=terminal['report_sha256']
            or sha(root/'manifest.json')!=report['manifest_sha256']):
        raise ValueError('Controller/report/manifest binding differs')
    for name,h in manifest['files'].items():
        if sha(name)!=h:raise ValueError('Frozen dependency changed: '+name)
    for e in events:
        if read(execution/(e['task']+'-terminal.json'))!=e:
            raise ValueError('Aggregated actual exit differs from original record')
        start=read(execution/(e['task']+'-start.json'))
        if start['time_ns']>=e['completed_ns']:
            raise ValueError('Child timestamps differ')
    count=steps=base_states=heads=0;truth={}
    import pickle
    for seed in (42,3407):
        for fold in range(5):
            unit=root/f's{seed}-f{fold}'
            plan=manifest['plans'][unit.name]
            with (unit/'training.pkl').open('rb') as stream:
                remember_truth(truth,pickle.load(stream),plan)
            for inner in range(5):
                d=unit/f'inner-{inner}';c=read(d/'complete.json');cold=read(d/'cold.json')
                inner_plan=plan['inner'][str(inner)]
                if c['fit_ids']!=inner_plan['fit_ids'] or c['query_ids']!=inner_plan['held_ids']:
                    raise ValueError('Base fit/held IDs differ from frozen nested plan')
                if cold['status']!='passed' or cold['complete_sha256']!=sha(d/'complete.json'):
                    raise ValueError('Base cold binding differs')
                if c['prediction_sha256']!=sha(d/'predictions.npz'):
                    raise ValueError('Base prediction hash differs')
                base_states+=cold['states'];source=Path(c['source']);traces={}
                for role,h in c['state_hashes'].items():
                    path=source/(role+'.pt')
                    if sha(path)!=h:raise ValueError('Base state hash differs')
                    traces[role]=torch.load(path,map_location='cpu',weights_only=True)['trace']
                if inner==0:
                    if c['constructors'] or c['steps'] or source==d:
                        raise ValueError('Reused inner0 incorrectly recorded as new fit')
                else:
                    if source!=d:raise ValueError('New fit physical source differs')
                    starts={role:read(d/(role+'-optimizer-start.json')) for role in ('selection','refit')}
                    steps+=verify_optimizer_receipts(c,read(d/'estimator-start.json'),starts,traces)
                    count+=len(c['constructors'])
            c=read(unit/'heads-complete.json');cold=read(unit/'heads-cold.json');old=read(unit/'old-T-cold.json')
            if (cold['status']!='passed' or cold['complete_sha256']!=sha(unit/'heads-complete.json')
                    or old['status']!='passed' or c['prediction_sha256']!=sha(unit/'predictions.npz')
                    or c['input_sha256']!=sha(unit/'head-inputs.npz')):
                raise ValueError('Head/control bindings differ')
            base_states+=old['states'];heads+=c['head_fit_calls']
            for family,h in c['states'].items():
                if sha(unit/family/('state.npz' if family=='RIDGE' else 'state.txt'))!=h:
                    raise ValueError('Head state hash differs')
    if (count,base_states,heads)!=(80,120,20):
        raise ValueError('Native optimizer/model/head inventory differs')
    gains={k:{} for k in ('RIDGE','GBM')};maximum=0.;array_hashes={};common_ids=None
    for seed in (42,3407):
        path=root/f'oof-s{seed}.npz';array_hashes[str(seed)]=sha(path)
        with np.load(path,allow_pickle=False) as a:
            if (len(a['ids'])!=2754 or len(set(a['ids']))!=2754 or set(a['folds'])!=set(range(5))
                    or not np.isfinite(a['actual']).all() or a['actual'].shape!=(2754,2)):
                raise ValueError('Incomplete OOF identity/target array')
            if common_ids is None:common_ids=a['ids'].copy()
            else:np.testing.assert_array_equal(common_ids,a['ids'])
            if set(a['ids'])!=set(truth):raise ValueError('OOF and frozen training-label coverage differ')
            np.testing.assert_array_equal(a['actual'],np.asarray([truth[i] for i in a['ids']],float))
            for key in ('q75','iron','RIDGE','GBM'):
                if not np.isfinite(a[key]).all() or (a[key]<0).any():raise ValueError('Invalid final OOF column')
            for fold in range(5):
                unit=root/f's{seed}-f{fold}';mask=a['folds']==fold
                with np.load(unit/'predictions.npz',allow_pickle=False) as p:
                    np.testing.assert_array_equal(p['ids'],a['ids'][mask])
                    for k in ('q75','iron','RIDGE','GBM'):
                        np.testing.assert_array_equal(p[k],a[k][mask])
            base=scalar_score(a['actual'],a['iron'],a['q75'])
            for family in gains:
                if not np.isfinite(a[family]).all() or (a[family]<0).any():
                    raise ValueError('Nonfinite/negative final OOF')
                gain=scalar_score(a['actual'],a['iron'],a[family])-base
                gains[family][str(seed)]=gain
                maximum=max(maximum,abs(gain-report['gains'][family][str(seed)]))
    if maximum>1e-11:raise ValueError('Independent two-target score difference failed')
    eligible=[k for k in ('RIDGE','GBM') if min(gains[k].values())>0]
    chosen=None
    if eligible:
        means={k:math.fsum(gains[k].values())/2 for k in eligible};best=max(means.values())
        chosen=next(k for k in ('RIDGE','GBM') if k in means and means[k]>=best-1e-12)
    if chosen!=report['selected_for_confirmation'] or report['formal_promoted']:
        raise ValueError('Original two-split selection scope differs')
    value=dict(status='passed',audited_ns=time.time_ns(),auditor_sha256=sha(__file__),
        actual_child_exit_codes=[e['exit_code'] for e in events],native_optimizer_constructors=count,
        native_optimizer_steps=steps,cold_base_states=base_states,head_fits=heads,
        independent_gains=gains,independent_two_target_score_maximum_difference=maximum,
        selected_for_confirmation=chosen,formal_promoted=False,maximum_child_rss_mib=max(e['peak_rss_mib'] for e in events),
        original_terminal_sha256=sha(execution/'terminal.json'),report_sha256=sha(root/'report.json'),
        manifest_sha256=sha(root/'manifest.json'),oof_sha256=array_hashes,new_fits=0,new_predictions=0,new_packages=0)
    with output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ['actual_child_exit_codes']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,12):raise RuntimeError('Locked Python3.12 required')
    run(args.root,args.output)
