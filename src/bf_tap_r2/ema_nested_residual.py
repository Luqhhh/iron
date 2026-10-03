"""Nested EMA residual heads, with base fits isolated inside each outer T."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import numpy as np

from .data import FEATURES, TARGETS
from .v3_4_bags import group_safe_inner_folds
from .v7_periodic import digest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'local/runs/ema-nested-residual-20261003/development-r1'
OLD = ROOT/'local/runs/strong-component-regularization/development-r2'
CAL = ROOT/'local/runs/q75-error-relocation-20261001/calibration-development-r1'
PROTOCOL = ROOT/'docs/ema_nested_residual/PREREGISTRATION.md'
SEEDS = (42, 3407)
ORDER = ('RIDGE', 'GBM')
VERSIONS = {'torch':'2.14.0+cpu', 'tabm':'0.0.3', 'rtdl-num-embeddings':'0.0.12',
            'numpy':'2.2.6', 'scikit-learn':'1.8.0', 'scipy':'1.18.1', 'lightgbm':'4.6.0'}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, v):
    with Path(p).open('x') as f:
        json.dump(v, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def save_arrays(p, **arrays):
    with Path(p).open('xb') as f:
        np.savez_compressed(f, **arrays)


def verify(files):
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError('Frozen dependency changed: '+p)


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak > 1536:
        raise ValueError('RSS gate exceeded')
    return peak


def setup():
    if sys.version_info[:2] != (3, 12) or any(os.environ.get(k) != '1' for k in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise ValueError('Python3.12 and pre-import thread pinning required')
    if {k:importlib.metadata.version(k) for k in VERSIONS} != VERSIONS:
        raise ValueError('CPU dependency versions changed')
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)


def context():
    from .v5_library import load_v5_training_frame, fold_vector
    from .v5_spec import load_v5_spec
    m = read(RUN/'manifest.json'); verify(m['files'])
    frame = load_v5_training_frame(ROOT)
    folds = {s:fold_vector(ROOT, frame, s, load_v5_spec(ROOT)) for s in SEEDS}
    if {str(s):digest(v.tolist()) for s,v in folds.items()} != m['outer_folds']:
        raise ValueError('Outer assignments changed')
    return m, frame, folds


def split_training(frame, folds, fold):
    training = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold, ['sample_id','spout_no',*FEATURES]].reset_index(drop=True)
    if set(training.sample_id) & set(query.sample_id):
        raise ValueError('Outer overlap')
    return training, query


def inner_parts(training, inner_fold):
    info = group_safe_inner_folds(training, seed=27001, n_splits=5)
    mask = info['fold'] == inner_fold
    fitting = training.loc[~mask].reset_index(drop=True)
    query = training.loc[mask, ['sample_id','spout_no',*FEATURES]].reset_index(drop=True)
    if not mask.any() or mask.all() or set(fitting.sample_id) & set(query.sample_id):
        raise ValueError('Invalid inner partition')
    return fitting, query, mask, info


def features(frame, base, categories):
    if any(t in frame for t in TARGETS):
        raise ValueError('Head inputs must be label-free')
    base = np.asarray(base, float)
    if base.shape != (len(frame),) or not np.isfinite(base).all():
        raise ValueError('Misaligned/nonfinite base prediction')
    x = np.column_stack([frame[list(FEATURES)].to_numpy(float),
                         *[(frame.spout_no.to_numpy() == c).astype(float) for c in categories], base])
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite head inputs')
    return x


def fit_head(family, x, residual, directory):
    directory = Path(directory); directory.mkdir(exist_ok=False)
    write(directory/'fit-start.json', dict(family=family, rows=len(x), started_ns=time.time_ns()))
    if family == 'RIDGE':
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=10., solver='cholesky').fit(scaler.transform(x), residual)
        save_arrays(directory/'state.npz', mean=scaler.mean_, scale=scaler.scale_,
                    coef=model.coef_, intercept=np.asarray(model.intercept_))
        return lambda z:model.predict(scaler.transform(z))
    if family != 'GBM':
        raise ValueError('Unknown head')
    import lightgbm as lgb
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=.03, num_leaves=15,
        min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.6,
        reg_lambda=1., random_state=42, n_jobs=1, verbose=-1)
    model.fit(x, residual)
    with (directory/'state.txt').open('x') as f:
        f.write(model.booster_.model_to_string())
    return lambda z:model.booster_.predict(z, num_threads=1)


def predict_head(family, directory, x):
    directory = Path(directory)
    if family == 'RIDGE':
        with np.load(directory/'state.npz', allow_pickle=False) as a:
            return ((x-a['mean'])/a['scale'])@a['coef']+a['intercept']
    import lightgbm as lgb
    return lgb.Booster(model_file=str(directory/'state.txt')).predict(x, num_threads=1)


def corrected(q75, head):
    q75, head = np.asarray(q75,float), np.asarray(head,float)
    if q75.ndim != 1 or q75.shape != head.shape:
        raise ValueError('Correction alignment differs')
    result = q75+.75*.25*head
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid correction; no clipping')
    return result


def choose(gains):
    if list(gains) != list(ORDER) or any(set(v) != {'42','3407'} for v in gains.values()):
        raise ValueError('Complete fixed pool and two splits required')
    if any(not math.isfinite(v) for d in gains.values() for v in d.values()):
        raise ValueError('Nonfinite gain')
    eligible = [k for k in ORDER if min(gains[k].values()) > 0]
    if not eligible:
        return None
    means = {k:sum(gains[k].values())/2 for k in eligible}; best = max(means.values())
    return next(k for k in ORDER if k in means and means[k] >= best-1e-12)


@contextmanager
def forbid_training():
    import torch
    from .component_regularization import ComponentRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    def forbidden(*a, **kw):
        raise ValueError('Cold process attempted fitting')
    from contextlib import ExitStack
    with ExitStack() as stack:
        for cls,name in [(ComponentRegressor,'fit'),(ComponentRegressor,'_train'),
                         (ComponentRegressor,'_initialize'),(torch.optim,'AdamW'),
                         (Ridge,'fit'),(StandardScaler,'fit'),(lgb.LGBMRegressor,'fit'),(lgb,'train')]:
            stack.enter_context(patch.object(cls, name, forbidden))
        yield


def prepare(checks):
    import yaml
    from .v5_library import load_v5_training_frame, fold_vector
    from .v5_spec import load_v5_spec
    if RUN.exists():
        raise FileExistsError('Run directory already consumed')
    best = read(ROOT/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'],best['score'],best['zip_sha256']) != ('EMA_TIME_Q75',96.392,
            '41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825'):
        raise ValueError('Current reference changed')
    c = read(checks)
    if c['status'] != 'passed' or c['python'] != sys.version.split()[0] or c['source_sha256'] != sha(__file__) or sha(c['junit']) != c['junit_sha256']:
        raise ValueError('Current locked checks required')
    original = read(CAL/'manifest.json'); old = read(OLD/'manifest.json')
    for n in ['component_regularization.py','v12_joint.py','v7_periodic.py','v3_6_networks.py']:
        rel = 'src/bf_tap_r2/'+n
        if sha(ROOT/rel) != original['sources'][rel] or sha(ROOT/rel) != old['source_hashes'][rel]:
            raise ValueError('Original base science changed')
    for root in (CAL, OLD):
        a = read(root/'audit.json')
        if a['status'] != 'passed' or a['manifest_sha256'] != sha(root/'manifest.json'):
            raise ValueError('Original audit not closed')
    paths = list((ROOT/'src').rglob('*.py')) + [PROTOCOL, Path(checks), Path(c['junit']),
        ROOT/'tests/test_ema_nested_residual.py',ROOT/'uv.lock',ROOT/'pyproject.toml',
        ROOT/'configs/strong_component_regularization/SPEC.yaml', ROOT/'configs/round2_v5/SPEC.yaml',
        ROOT/'configs/data.local.yaml',ROOT/'configs/protection.yaml',ROOT/'configs/candidate_tiers.yaml',
        ROOT/'local/authorizations/optimization-standing-20261002-r1.json']
    files = {str(p.resolve()):sha(p) for p in paths}
    # Bind original data and fold files, without reopening preliminary targets.
    for rel,h in old['data_hashes'].items():
        files[str(ROOT/rel)] = h
    v5 = load_v5_spec(ROOT)
    for rel in v5.raw['reference']['frozen_folds'].values():
        files[str(ROOT/rel)] = sha(ROOT/rel)
    for base in (CAL,OLD):
        for n in ('manifest.json','audit.json'):
            files[str(base/n)] = sha(base/n)
    for seed in SEEDS:
        for fold in range(5):
            for unit in [CAL/f's{seed}-f{fold}', OLD/f'tap_time_len-EMA-s{seed}-f{fold}', OLD/f'reference-s{seed}-f{fold}']:
                receipt = read(unit/'complete.json')
                files[str(unit/'complete.json')] = sha(unit/'complete.json')
                files.update({str(unit/n):h for n,h in receipt['hashes'].items()})
    verify(files)
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN/'inputs-before-access.json', files)
    write(RUN/'round2-access.json', dict(scope='authorized_round2_only',
        preliminary_targets_read=False, input_manifest_sha256=sha(RUN/'inputs-before-access.json')))
    frame = load_v5_training_frame(ROOT)
    folds = {s:fold_vector(ROOT,frame,s,v5) for s in SEEDS}
    plans = {}
    for seed,fv in folds.items():
        for fold in range(5):
            training,query = split_training(frame,fv,fold)
            unit = RUN/f's{seed}-f{fold}';unit.mkdir(exist_ok=False)
            # Workers receive only their own T labels and label-free query.
            for name,value in [('training',training),('query',query)]:
                path=unit/(name+'.pkl')
                with path.open('xb') as stream:
                    import pickle
                    pickle.dump(value,stream,protocol=5)
                files[str(path)]=sha(path)
            plans[f's{seed}-f{fold}'] = dict(training_ids=training.sample_id.tolist(), query_ids=query.sample_id.tolist(), inner={})
            for inner in range(5):
                fitting,held,_,info = inner_parts(training,inner)
                plans[f's{seed}-f{fold}']['inner'][str(inner)] = dict(fit_ids=fitting.sample_id.tolist(),
                    held_ids=held.sample_id.tolist(), inner_hash=info['inner_fold_hash'])
    settings = yaml.safe_load((ROOT/'configs/strong_component_regularization/SPEC.yaml').read_text())
    if settings['training']['tap_time_len'] != original['spec']['training'] or settings['mechanisms'] != original['spec']['mechanisms']:
        raise ValueError('Cached model training protocol differs')
    write(RUN/'manifest.json', dict(files=files, plans=plans, training=settings['training']['tap_time_len'],
        mechanisms=settings['mechanisms'], versions=VERSIONS,
        outer_folds={str(s):digest(v.tolist()) for s,v in folds.items()}, seeds=list(SEEDS),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        candidate_order=list(ORDER),gamma=.25,component_weight=.75,reference='EMA_TIME_Q75',
        budget=dict(new_estimators=40,new_optimizers=80,new_base_states=80,reused_base_states=40,
                    head_fits=20,head_states=20,new_confirmation_seeds=0,full_fits=0,packages=0),
        monitor_seconds=600, time_budget_seconds=None, automatic_retries=False))
    print(json.dumps(dict(status='frozen',files=len(files),new_optimizers=80,head_fits=20)),flush=True)


def unit_frames(seed,fold):
    import pickle
    m=read(RUN/'manifest.json');verify(m['files'])
    unit=RUN/f's{seed}-f{fold}'
    with (unit/'training.pkl').open('rb') as f:training=pickle.load(f)
    with (unit/'query.pkl').open('rb') as f:query=pickle.load(f)
    if any(t in query for t in TARGETS) or set(training.sample_id)&set(query.sample_id):
        raise ValueError('Worker receives invalid outer inputs')
    plan = m['plans'][f's{seed}-f{fold}']
    if training.sample_id.tolist() != plan['training_ids'] or query.sample_id.tolist() != plan['query_ids']:
        raise ValueError('Outer plan changed')
    return m,training,query


def base_unit(seed,fold,inner):
    from .component_regularization import ComponentRegressor
    from .component_regularization_run import RECIPE
    import torch
    m,training,outer = unit_frames(seed,fold)
    fitting,query,_,_ = inner_parts(training,inner)
    directory = RUN/f's{seed}-f{fold}'/f'inner-{inner}'
    directory.mkdir(parents=True,exist_ok=False)
    identity = dict(source_directory=str(directory), split_seed=seed,
                    trial_id=f'EMA-nested-f{fold}-inner{inner}')
    constructors = []; steps = {}
    if inner == 0:
        source = CAL/f's{seed}-f{fold}'
        write(directory/'reuse.json',dict(source_directory=str(source),split_seed=seed,trial_id='EMA',
                                        original_complete_sha256=sha(source/'complete.json')))
    else:
        source = directory
        write(directory/'estimator-start.json',dict(identity=identity,fit_ids=fitting.sample_id.tolist(),time_ns=time.time_ns()))
        original_adam = torch.optim.AdamW
        def counted(*a,**kw):
            if len(constructors) >= 2:
                raise ValueError('Optimizer budget exceeded')
            role = ('selection','refit')[len(constructors)];constructors.append(role);steps[role]=0
            write(directory/(role+'-optimizer-start.json'),dict(identity=identity,role=role,time_ns=time.time_ns()))
            opt = original_adam(*a,**kw); real = opt.step
            def step(*a,**kw):
                value = real(*a,**kw);steps[role]+=1;return value
            opt.step=step
            return opt
        with patch.object(torch.optim,'AdamW',counted):
            model = ComponentRegressor(RECIPE,m['training'],'EMA',m['mechanisms'],directory)
            model.fit(fitting.drop(columns=list(TARGETS)), fitting[['tap_time_len']].to_numpy())
        if constructors != ['selection','refit'] or any(steps[k] != model.traces[k]['updates'] for k in constructors):
            raise ValueError('Native optimizer inventory mismatch')
    predictions = {};states={}
    with forbid_training():
        for role in ('selection','refit'):
            states[role]=sha(source/(role+'.pt'))
            predictions[role]=ComponentRegressor.load(source/(role+'.pt')).predict(query)[:,0]
    if inner == 0:
        with np.load(source/'predictions.npz',allow_pickle=False) as a:
            np.testing.assert_array_equal(a['calibration_ids'],query.sample_id.to_numpy(str))
            np.testing.assert_array_equal(a['ema_calibration'],predictions['refit'])
    save_arrays(directory/'predictions.npz',ids=query.sample_id.to_numpy(str),**predictions)
    write(directory/'complete.json',dict(identity=identity,source=str(source),state_hashes=states,
        constructors=constructors,steps=steps,fit_ids=fitting.sample_id.tolist(),query_ids=query.sample_id.tolist(),
        prediction_sha256=sha(directory/'predictions.npz'),peak_rss_mib=memory()))


def audit_models(source,fitting,query,expected,settings,mechanisms):
    from .component_regularization_audit import verify_saved
    inner = group_safe_inner_folds(fitting,seed=settings['inner_seed'])['fold']
    f = fitting.loc[inner != 0].reset_index(drop=True);v = fitting.loc[inner == 0]
    with forbid_training():
        selector = verify_saved(source/'selection.pt',f,f[['tap_time_len']].to_numpy(),'EMA',settings,mechanisms,v)
        model = verify_saved(source/'refit.pt',fitting,fitting[['tap_time_len']].to_numpy(),'EMA',settings,mechanisms,
                             expected_epoch=selector.saved['trace']['selected_epoch'])
        maximum=0.
        for role,predictor in [('selection',selector),('refit',model)]:
            if role not in expected:
                continue
            p = expected[role]
            np.testing.assert_array_equal(predictor.predict(query)[:,0],p)
            for result in [predictor.predict(query.iloc[::-1])[::-1,0],
                    np.concatenate([predictor.predict(query.iloc[i:i+37])[:,0] for i in range(0,len(query),37)])]:
                maximum=max(maximum,float(np.max(np.abs(result-p))))
        if maximum > .0005:
            raise ValueError('Cold chunk/order gate failed')
    return dict(status='passed',maximum_difference=maximum,states=2,
                selected_epoch=selector.saved['trace']['selected_epoch'],peak_rss_mib=memory())


def audit_base(seed,fold,inner):
    m,training,_ = unit_frames(seed,fold);fitting,query,_,_ = inner_parts(training,inner)
    directory = RUN/f's{seed}-f{fold}'/f'inner-{inner}';c=read(directory/'complete.json');source=Path(c['source'])
    if fitting.sample_id.tolist()!=c['fit_ids'] or query.sample_id.tolist()!=c['query_ids'] or sha(directory/'predictions.npz')!=c['prediction_sha256']:
        raise ValueError('Base receipt identity mismatch')
    for role,h in c['state_hashes'].items():
        if sha(source/(role+'.pt'))!=h:raise ValueError('Base state changed')
    with np.load(directory/'predictions.npz',allow_pickle=False) as a:
        receipt=audit_models(source,fitting,query,{k:a[k] for k in ('selection','refit')},m['training'],m['mechanisms'])
    receipt['complete_sha256']=sha(directory/'complete.json')
    write(directory/'cold.json',receipt)


def head_inputs(seed,fold):
    from .ema_average_span import old_cache
    m,training,query=unit_frames(seed,fold);base=np.full(len(training),np.nan)
    unit=RUN/f's{seed}-f{fold}'
    for inner in range(5):
        directory=unit/f'inner-{inner}';c=read(directory/'cold.json')
        if c['status']!='passed' or c['complete_sha256']!=sha(directory/'complete.json'):
            raise ValueError('Closed cold base required')
        _,held,mask,_=inner_parts(training,inner)
        with np.load(directory/'predictions.npz',allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'],held.sample_id.to_numpy(str));base[mask]=a['refit']
    if not np.isfinite(base).all():raise ValueError('Incomplete T-only OOF')
    reference=old_cache(dict(main_root=str(ROOT),old_development=str(OLD)),seed,fold,training,query)
    cats=sorted(int(x) for x in training.spout_no.unique())
    x=features(training.drop(columns=list(TARGETS)),base,cats);qx=features(query,reference['ema'],cats)
    return m,training,query,base,x,qx,reference,cats


def heads(seed,fold):
    m,training,query,base,x,qx,ref,cats=head_inputs(seed,fold);unit=RUN/f's{seed}-f{fold}'
    # This child is a fresh process relative to every base fit.
    receipt=audit_models(OLD/f'tap_time_len-EMA-s{seed}-f{fold}',training,query,
        {'refit':ref['ema']},m['training'],m['mechanisms'])
    write(unit/'old-T-cold.json',receipt)
    columns={};residual=training.tap_time_len.to_numpy(float)-base;head_hashes={}
    for family in ORDER:
        directory=unit/family
        predictor=fit_head(family,x,residual,directory)
        h=predictor(qx);columns[family+'_head']=h;columns[family]=corrected(ref['q75'],h)
        state=directory/('state.npz' if family=='RIDGE' else 'state.txt')
        head_hashes[family]=sha(state)
    save_arrays(unit/'head-inputs.npz',training_ids=training.sample_id.to_numpy(str),query_ids=query.sample_id.to_numpy(str),
        x=x,qx=qx,residual=residual,base=base,q75=ref['q75'],iron=ref['iron'],ema=ref['ema'])
    save_arrays(unit/'predictions.npz',ids=query.sample_id.to_numpy(str),q75=ref['q75'],iron=ref['iron'],**columns)
    write(unit/'heads-complete.json',dict(head_fit_calls=2,categories=cats,states=head_hashes,
        training_ids_digest=digest(training.sample_id.tolist()),query_ids_digest=digest(query.sample_id.tolist()),
        input_sha256=sha(unit/'head-inputs.npz'),prediction_sha256=sha(unit/'predictions.npz'),peak_rss_mib=memory()))


def audit_heads(seed,fold):
    _,training,query,base,x,qx,ref,cats=head_inputs(seed,fold);unit=RUN/f's{seed}-f{fold}';c=read(unit/'heads-complete.json')
    if sha(unit/'head-inputs.npz')!=c['input_sha256'] or sha(unit/'predictions.npz')!=c['prediction_sha256']:
        raise ValueError('Head input/output changed')
    with np.load(unit/'head-inputs.npz',allow_pickle=False) as a:
        for k,value in [('x',x),('qx',qx),('base',base),('residual',training.tap_time_len.to_numpy(float)-base)]:
            np.testing.assert_array_equal(a[k],value)
        residual=a['residual']
    maximum=0.
    with forbid_training(),np.load(unit/'predictions.npz',allow_pickle=False) as p:
        for family in ORDER:
            directory=unit/family;state=directory/('state.npz' if family=='RIDGE' else 'state.txt')
            if sha(state)!=c['states'][family]:raise ValueError('Head state changed')
            got=predict_head(family,directory,qx)
            maximum=max(maximum,float(np.max(np.abs(got-p[family+'_head']))))
            if family=='RIDGE':
                # Independent normal equations using only T inputs/residuals.
                mean=x.mean(0);scale=x.std(0);scale[scale==0]=1.;z=(x-mean)/scale;zm=z.mean(0)
                zz=z-zm;ym=float(residual.mean())
                coef=np.linalg.solve(zz.T@zz+10.*np.eye(x.shape[1]),zz.T@(residual-ym))
                independent=((qx-mean)/scale-zm)@coef+ym
                maximum=max(maximum,float(np.max(np.abs(got-independent))))
            maximum=max(maximum,float(np.max(np.abs(corrected(ref['q75'],got)-p[family]))))
    if maximum>1e-9:raise ValueError('Head cold/independent arithmetic mismatch')
    write(unit/'heads-cold.json',dict(status='passed',maximum_difference=maximum,
        complete_sha256=sha(unit/'heads-complete.json'),peak_rss_mib=memory()))


def report():
    from .candidate_tiers import classify_candidates
    from .v49_run import metric_detail
    import yaml
    m,frame,folds=context();gains={k:{} for k in ORDER};records={};metrics={'tap_time_len':{k:{} for k in ('Q75',*ORDER)}}
    base_count=0;state_count=0;head_count=0;maximum=0.
    for seed,fv in folds.items():
        columns={k:np.full(len(frame),np.nan) for k in ('q75','iron',*ORDER)}
        for fold in range(5):
            unit=RUN/f's{seed}-f{fold}';c=read(unit/'heads-cold.json')
            if c['status']!='passed' or c['complete_sha256']!=sha(unit/'heads-complete.json'):raise ValueError('Cold head not closed')
            head_count+=read(unit/'heads-complete.json')['head_fit_calls']
            state_count+=read(unit/'old-T-cold.json')['states']
            for inner in range(5):
                directory=unit/f'inner-{inner}';base_count+=len(read(directory/'complete.json')['constructors'])
                state_count+=read(directory/'cold.json')['states']
            with np.load(unit/'predictions.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'],frame.loc[fv==fold,'sample_id'].to_numpy(str))
                for k in columns:columns[k][fv==fold]=a[k]
        if any(not np.isfinite(v).all() or (v<0).any() for v in columns.values()):raise ValueError('Incomplete/invalid OOF')
        y=frame.tap_time_len.to_numpy(float);den=float(np.abs(y).sum());base=np.abs(y-columns['q75']).sum()/den
        independent_base=math.fsum(abs(float(a)-float(b)) for a,b in zip(y,columns['q75']))/math.fsum(abs(float(a)) for a in y)
        records[str(seed)]={}
        for family in ORDER:
            gain=float(50*(base-np.abs(y-columns[family]).sum()/den));gains[family][str(seed)]=gain
            independent=50*(independent_base-math.fsum(abs(float(a)-float(b)) for a,b in zip(y,columns[family]))/math.fsum(abs(float(a)) for a in y))
            maximum=max(maximum,abs(gain-independent));records[str(seed)][family]=dict(gain=gain,independent_gain=independent)
        for name in metrics['tap_time_len']:
            metrics['tap_time_len'][name][str(seed)]=metric_detail(y,columns['q75'] if name=='Q75' else columns[name],fv,frame.spout_no.to_numpy())
        save_arrays(RUN/f'oof-s{seed}.npz',ids=frame.sample_id.to_numpy(str),folds=fv,actual=frame[list(TARGETS)].to_numpy(),**columns)
    if (base_count,state_count,head_count)!=(80,120,20) or maximum>1e-11:raise ValueError('Count/score gate failed')
    spec=dict(split_seeds=list(SEEDS),folds=5,candidates={'tap_time_len':list(ORDER)},
        reference_by_target={'tap_time_len':'Q75'},tie_preference_by_target={'tap_time_len':list(ORDER)})
    tiers=classify_candidates(metrics,spec,yaml.safe_load((ROOT/'configs/candidate_tiers.yaml').read_text()))
    write(RUN/'report.json',dict(status='complete_development_not_formal_promotion',records=records,gains=gains,
        selected_for_confirmation=choose(gains),candidate_tiers=tiers,actual_optimizer_constructors=base_count,
        cold_base_states=state_count,head_fits=head_count,independent_score_maximum_difference=maximum,
        formal_promoted=False,manifest_sha256=sha(RUN/'manifest.json'),peak_rss_mib=memory()))
    verify(m['files']);print(json.dumps(dict(gains=gains,selected=choose(gains))),flush=True)


def controller():
    m=read(RUN/'manifest.json');verify(m['files']);launch=RUN/'execution';launch.mkdir(exist_ok=False)
    write(launch/'started.json',dict(pid=os.getpid(),started_ns=time.time_ns(),manifest_sha256=sha(RUN/'manifest.json')))
    stop=threading.Event();owned={'pid':None,'task':None};lock=threading.Lock()
    def monitor():
        import psutil
        number=0
        while not stop.wait(600):
            number+=1
            with lock:record=dict(owned)
            row=dict(number=number,time_ns=time.time_ns(),**record)
            if record['pid'] is not None:
                try:
                    p=psutil.Process(record['pid']);row.update(live=p.is_running(),rss_mib=p.memory_info().rss/1024**2)
                except psutil.NoSuchProcess:row['live']=False
            write(launch/f'observation-{number:03d}.json',row);print(json.dumps(row),flush=True)
    observer=threading.Thread(target=monitor,daemon=True);observer.start();events=[]
    tasks=[]
    for seed in SEEDS:
        for fold in range(5):
            for inner in range(5):
                tasks.extend([(s,seed,fold,inner) for s in ('base','base-cold')])
            tasks.extend([(s,seed,fold,-1) for s in ('heads','heads-cold')])
    tasks.append(('report',42,0,-1))
    try:
        for index,(stage,seed,fold,inner) in enumerate(tasks):
            key=f'{index:03d}-{stage}-s{seed}-f{fold}-i{inner}'
            with (launch/(key+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-m','bf_tap_r2.ema_nested_residual',stage,
                    '--seed',str(seed),'--fold',str(fold),'--inner',str(inner)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                with lock:owned.update(pid=child.pid,task=key)
                write(launch/(key+'-start.json'),dict(pid=child.pid,time_ns=time.time_ns()))
                _,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
            row=dict(task=key,exit_code=code,peak_rss_mib=usage.ru_maxrss/1024,completed_ns=time.time_ns())
            write(launch/(key+'-terminal.json'),row);events.append(row)
            with lock:owned.update(pid=None,task=None)
            print(json.dumps(row),flush=True)
            if code!=0 or row['peak_rss_mib']>1536:raise RuntimeError('Scientific child failed: '+key)
        verify(m['files'])
        write(launch/'terminal.json',dict(status='passed',events=events,report_sha256=sha(RUN/'report.json'),dependencies_unchanged=True))
    except BaseException as error:
        write(launch/'terminal.json',dict(status='failed',events=events,error=repr(error),automatic_retry=False));raise
    finally:
        stop.set();observer.join()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','base','base-cold','heads','heads-cold','report'])
    p.add_argument('--checks');p.add_argument('--seed',type=int,default=42);p.add_argument('--fold',type=int,default=0);p.add_argument('--inner',type=int,default=0)
    args=p.parse_args();setup()
    if args.stage=='prepare':prepare(args.checks)
    elif args.stage=='run':controller()
    elif args.stage=='report':report()
    elif args.stage in ('heads','heads-cold'):
        {'heads':heads,'heads-cold':audit_heads}[args.stage](args.seed,args.fold)
    else:{'base':base_unit,'base-cold':audit_base}[args.stage](args.seed,args.fold,args.inner)
