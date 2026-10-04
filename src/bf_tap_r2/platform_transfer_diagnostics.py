"""Frozen, retrospective transfer diagnostics. Never selects a platform winner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

MAIN = Path('/home/lux1/iron')
WORK = Path(__file__).resolve().parents[2]
SPEC = 'configs/platform_transfer_diagnostics/SPEC.json'
DEV = MAIN/'local/runs/ema-silu-20261004/development-r1'
FULL = MAIN/'local/runs/ema-silu-release-20261004/release-r1'
OLD = MAIN/'local/runs/ema-retraining-initialization-20261002/development-r1'
NESTED = MAIN/'local/runs/ema-nested-residual-20261003/development-r1'
SEEDS = (42, 3407)
INITS = (42, 1042, 2042)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    with Path(p).open('x') as f:
        json.dump(v, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def save(p, **arrays):
    with Path(p).open('xb') as f:
        np.savez_compressed(f, **arrays)


def verify(files):
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError('Frozen identity changed: '+p)


def environment():
    import importlib.metadata
    import psutil
    import torch
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Locked Python 3.12 required')
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(k) != '1':
            raise ValueError('Set numerical threads before import: '+k)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    expected = read(WORK/'configs/ema_silu/SPEC.json')['runtime_versions']
    versions = {k:importlib.metadata.version(k) for k in expected}
    if versions != expected:
        raise ValueError('Frozen CPU environment changed')
    if psutil.virtual_memory().available/2**20 < 3072:
        raise ValueError('Entry memory gate failed')
    return dict(python=sys.version.split()[0], versions=versions, pid=os.getpid())


def features(frame):
    """Only the approved inputs enter a domain diagnostic, never IDs/targets."""
    from .data import FEATURES
    numeric = frame.loc[:, list(FEATURES)].to_numpy(float)
    spout = frame.spout_no.to_numpy()
    if not np.isfinite(numeric).all() or not set(spout) <= {1, 2}:
        raise ValueError('Finite numeric inputs and original spout categories required')
    return np.column_stack([numeric, spout == 1, spout == 2])


def kernel(x, multipliers=(.5, 1., 2.)):
    from scipy.spatial.distance import pdist, squareform
    x = np.asarray(x, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Finite input matrix required')
    scale = x.std(0)
    z = (x-x.mean(0))/np.where(scale > 0, scale, 1)
    distances = pdist(z, metric='sqeuclidean')
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError('Constant pooled inputs have no positive bandwidth')
    median = float(np.median(positive))
    squared = squareform(distances)
    result = sum(np.exp(-squared/(2*median*b*b)) for b in multipliers)/len(multipliers)
    return result, median


def mmd_statistics(k, masks):
    """Unbiased unequal-sample MMD, diagonal removed for both groups."""
    k, masks = np.asarray(k, float), np.asarray(masks, float)
    if masks.ndim == 1:
        masks = masks[:, None]
    if (k.shape != (len(masks), len(masks)) or not np.isfinite(k).all()
            or not np.isin(masks, [0., 1.]).all()):
        raise ValueError('Aligned finite kernel and binary masks required')
    n, m = masks.sum(0), len(masks)-masks.sum(0)
    if (n < 2).any() or (m < 2).any():
        raise ValueError('Both samples need at least two rows')
    xx = (masks*(k@masks)).sum(0)
    x_all = masks.T@k.sum(1)
    xy = x_all-xx
    yy = k.sum()-2*x_all+xx
    dx = masks.T@np.diag(k)
    dy = np.trace(k)-dx
    return (xx-dx)/(n*(n-1))+(yy-dy)/(m*(m-1))-2*xy/(n*m)


def permutation_mmd(x, n_train, spec):
    k, bandwidth = kernel(x, spec['mmd_bandwidth_multipliers'])
    labels = np.arange(len(x)) >= n_train
    observed = float(mmd_statistics(k, labels)[0])
    rng = np.random.default_rng(spec['mmd_seed'])
    null, masks_saved = [], []
    for start in range(0, spec['mmd_permutations'], 32):
        masks = np.stack([rng.permutation(labels) for _ in
            range(min(32, spec['mmd_permutations']-start))], axis=1)
        null.extend(mmd_statistics(k, masks).tolist())
        if not masks_saved:
            masks_saved = masks[:, :3].T.tolist()
    null = np.asarray(null)
    return dict(mmd2=observed, p_value=float((1+np.sum(null >= observed))/(1+len(null))),
        permutations=len(null), squared_distance_median=bandwidth,
        null_quantiles={str(q):float(np.quantile(null,q)) for q in (.025,.5,.975)},
        interpretation='fixed_kernel_joint_input_test_not_conditional_target_test'), null, k, np.asarray(masks_saved)


def domain_crossfit(x, n_train, spec):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    labels = (np.arange(len(x)) >= n_train).astype(int)
    splitter = StratifiedKFold(spec['domain_folds'], shuffle=True, random_state=spec['domain_seed'])
    probability = {name:np.full(len(x), np.nan) for name in ('logistic','histogram')}
    ratio = {name:np.full(n_train, np.nan) for name in probability}
    assignment = np.full(len(x), -1, int)
    partitions = []
    models = []
    for fold, (fit, held) in enumerate(splitter.split(x, labels)):
        assignment[held] = fold
        prior = float(np.sum(labels[fit] == 0)/np.sum(labels[fit] == 1))
        partitions.append(dict(fold=fold, fit_indices=fit.tolist(), held_indices=held.tolist(), prior_ratio=prior))
        for name in probability:
            model = (make_pipeline(StandardScaler(), LogisticRegression(**spec['logistic']))
                if name == 'logistic' else HistGradientBoostingClassifier(**spec['histogram']))
            model.fit(x[fit], labels[fit])
            p = model.predict_proba(x[held])[:, 1]
            if not np.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
                raise ValueError('Domain probability outside open unit interval')
            probability[name][held] = p
            train_mask = held < n_train
            ratio[name][held[train_mask]] = p[train_mask]/(1-p[train_mask])*prior
            models.append((name,fold,model))
    details, arrays = {}, dict(domain_labels=labels, domain_folds=assignment)
    for name, p in probability.items():
        raw = ratio[name]
        clipped = np.clip(raw, *spec['diagnostic_weight_clip'])
        details[name] = dict(auc=float(roc_auc_score(labels,p)), log_loss=float(log_loss(labels,p)),
            raw_weight_min=float(raw.min()),raw_weight_max=float(raw.max()),
            clipped_fraction=float(np.mean(raw != clipped)),
            effective_sample_size=float(clipped.sum()**2/np.square(clipped).sum()),
            raw_effective_sample_size=float(raw.sum()**2/np.square(raw).sum()),
            classifier_fits=spec['domain_folds'])
        arrays[name+'_probability'] = p
        arrays[name+'_raw_weight'] = raw
        arrays[name+'_weight'] = clipped
    return details, arrays, partitions, models


def weighted_gain(y, ref, pred, weights):
    y, ref, pred, weights = [np.asarray(a, float) for a in (y, ref, pred, weights)]
    if (y.ndim != 1 or any(a.shape != y.shape or not np.isfinite(a).all() for a in (y,ref,pred,weights))
            or (weights <= 0).any() or (y < 0).any() or np.dot(weights,y) <= 0):
        raise ValueError('Positive aligned weights and target denominator required')
    return float(50*np.dot(weights,np.abs(y-ref)-np.abs(y-pred))/np.dot(weights,y))


def movement_detail(cv, full):
    cv, full = np.asarray(cv,float), np.asarray(full,float)
    if cv.ndim != 1 or cv.shape != full.shape or not np.isfinite([cv,full]).all():
        raise ValueError('Same finite query coordinates required')
    norm = np.linalg.norm(cv)*np.linalg.norm(full)
    return dict(rows=len(cv), cv_mean=float(cv.mean()), full_mean=float(full.mean()),
        cv_rms=float(np.sqrt(np.mean(cv**2))), full_rms=float(np.sqrt(np.mean(full**2))),
        difference_rms=float(np.sqrt(np.mean((full-cv)**2))),
        cosine=float(np.dot(cv,full)/norm) if norm else None,
        sign_agreement=float(np.mean(np.sign(cv) == np.sign(full))))


def inventory():
    """Resolve original cache identities; a reused trial is not a new fit."""
    records = []
    for seed in SEEDS:
        for fold in range(5):
            for init in INITS:
                d = OLD/f's{seed}-f{fold}-init{init}-EMA'
                w = read(d/'warm-complete.json')
                records.append(dict(arm='old',seed=seed,fold=fold,init=init,
                    checkpoint=w['paths']['refit'],checkpoint_sha256=w['checkpoint_hashes'][w['paths']['refit']],
                    evidence=str(d/'warm-complete.json')))
                d = DEV/f's{seed}-f{fold}-init{init}'
                w = read(d/'complete.json')
                records.append(dict(arm='new',seed=seed,fold=fold,init=init,
                    checkpoint=str(d/'refit.pt'),checkpoint_sha256=w['state_hashes']['refit'],
                    evidence=str(d/'complete.json')))
    manifest = read(FULL/'manifest.json')
    for init in INITS:
        for arm, d in [('old',Path(manifest['old_model_directories'][str(init)])),
                       ('new',FULL/f's-1-f-1-init{init}')]:
            checkpoint = str(d/'refit.pt')
            h = (manifest['files'][checkpoint] if arm == 'old'
                 else read(d/'complete.json')['state_hashes']['refit'])
            records.append(dict(arm=arm,seed=-1,fold=-1,init=init,
                checkpoint=checkpoint,checkpoint_sha256=h,evidence=str(FULL/'manifest.json')))
    if len(records) != 66 or len({r['checkpoint'] for r in records}) != 66:
        raise ValueError('66 distinct original refit states required')
    return records


def prepare(run, spec, checks):
    if run.exists():
        raise FileExistsError('Append-only run already exists')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():
        raise ValueError('Committed clean diagnostic source required')
    files = {}
    for base in (DEV,FULL):
        for name,h in read(base/'manifest.json')['files'].items():
            path = str((MAIN/name).resolve())
            if path in files and files[path] != h:
                raise ValueError('Conflicting immutable input identities')
            files[path] = h
    records = inventory()
    source_names = ['src/bf_tap_r2/platform_transfer_diagnostics.py',
        'tests/test_platform_transfer_diagnostics.py', SPEC,
        'docs/platform_transfer_diagnostics/PREREGISTRATION.md','uv.lock','pyproject.toml']
    paths = [WORK/p for p in source_names] + list((WORK/'src').rglob('*.py'))
    c = read(checks)
    if c['status'] != 'passed' or c['source_hashes'] != {p:sha(WORK/p) for p in source_names}:
        raise ValueError('Exact-source test receipt required')
    if c['actual_exit_code'] != 0 or sha(c['junit']) != c['junit_sha256']:
        raise ValueError('Tests did not close successfully')
    paths += [Path(checks),Path(c['junit']),DEV/'manifest.json',FULL/'manifest.json',
        DEV/'report.json',DEV/'execution/final-reconciliation.json',
        FULL/'delivery-terminal-reconciliation.json',FULL/'release.json',FULL/'original-members.npz']
    for p in (DEV/'execution/final-reconciliation.json',FULL/'delivery-terminal-reconciliation.json'):
        if read(p)['status'] != 'passed':
            raise ValueError('Original run not closed')
    for r in records:
        if sha(r['checkpoint']) != r['checkpoint_sha256']:
            raise ValueError('Original model checkpoint differs')
        paths += [Path(r['checkpoint']),Path(r['evidence'])]
    for seed in SEEDS:
        paths.append(DEV/f'oof-s{seed}.npz')
        if sha(paths[-1]) != read(DEV/'report.json')['oof_sha256'][str(seed)]:
            raise ValueError('OOF identity differs')
        for fold in range(5):
            paths += [NESTED/f's{seed}-f{fold}'/n for n in ('training.pkl','query.pkl')]
    paths += [FULL/n for n in ('training.pkl','query.pkl')]
    for init in INITS:
        paths += [FULL/f's-1-f-1-init{init}'/n for n in ('complete.json','cold.json','predictions.npz')]
    state = read(MAIN/'EVIDENCE_STATUS.json')
    best = state['round2_current_platform_best']
    if any(best[k] != v for k,v in spec['reference'].items()):
        raise ValueError('Incumbent changed before freeze')
    paths += [MAIN/best['package'],MAIN/best['platform_feedback_record'],
        MAIN/'local/runs/ema-silu-release-20261004/platform-feedback-r1/feedback.json']
    release_zip = Path(read(FULL/'release.json')['zip'])
    paths.append(release_zip)
    if sha(release_zip) != read(FULL/'delivery-terminal-reconciliation.json')['zip_sha256']:
        raise ValueError('Original submission package differs')
    for p in paths:
        name,h = str(p.resolve()),sha(p)
        if name in files and files[name] != h:
            raise ValueError('Frozen source/input drift: '+name)
        files[name] = h
    # These exact loaded model implementations must still equal the trained versions.
    for name in ('component_regularization.py','ema_silu_model.py','v7_periodic.py','v3_6_networks.py'):
        p = WORK/'src/bf_tap_r2'/name
        original = Path(read(DEV/'manifest.json')['source_directory'])/'src/bf_tap_r2'/name
        if sha(p) != sha(original):
            raise ValueError('Inference source differs from original training')
        files[str(p)] = sha(p)
    verify(files)
    run.mkdir(parents=True,exist_ok=False)
    write(run/'reference-snapshot.json',dict(best=best,feedback=state['ema_silu_platform_feedback_20261004']))
    files[str(run/'reference-snapshot.json')] = sha(run/'reference-snapshot.json')
    write(run/'manifest.json',dict(spec=spec, files=files, models=records,
        source_directory=str(WORK), source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        environment=environment(),created_ns=time.time_ns(),original_zip=str(release_zip),parent_zip=str(MAIN/best['package'])))
    print(json.dumps(dict(status='frozen',files=len(files),models=len(records))),flush=True)


def oof_read(seed, training):
    with np.load(DEV/f'oof-s{seed}.npz',allow_pickle=False) as p:
        a = {k:p[k].copy() for k in p.files}
    np.testing.assert_array_equal(a['ids'],training.sample_id.to_numpy(str))
    np.testing.assert_array_equal(a['actual'],training[['tap_iron','tap_time_len']].to_numpy(float))
    np.testing.assert_array_equal(a['spouts'],training.spout_no.to_numpy())
    if len(a['ids']) != 2754 or len(set(a['ids'])) != 2754 or set(a['folds']) != set(range(5)):
        raise ValueError('Complete unique five-fold OOF required')
    np.testing.assert_allclose(a['SILU_A100']-a['reference'],a['new_members'].mean(0)-a['old_members'].mean(0),rtol=0,atol=1e-10)
    return a


def replay(run, manifest, training, query):
    import gc
    import pandas as pd
    from .component_regularization import ComponentRegressor
    from .ema_silu_model import SiLUEMARegressor
    from .ema_nested_residual import forbid_training
    from .v7_periodic import digest
    arrays, traces = dict(ids=query.sample_id.to_numpy(str)), []
    original_members = np.load(FULL/'original-members.npz',allow_pickle=False)
    np.testing.assert_array_equal(original_members['ids'],arrays['ids'])
    max_difference = 0.
    for r in manifest['models']:
        seed,fold,init,arm = (r[k] for k in ('seed','fold','init','arm'))
        if seed == -1:
            fitting, held = training, query
            if arm == 'old':
                expected = original_members['members'][INITS.index(init)].copy()
            else:
                with np.load(FULL/f's-1-f-1-init{init}/predictions.npz',allow_pickle=False) as p:
                    np.testing.assert_array_equal(p['ids'],arrays['ids']);expected=p['prediction'].copy()
        else:
            unit = NESTED/f's{seed}-f{fold}'
            fitting, held = pd.read_pickle(unit/'training.pkl'),pd.read_pickle(unit/'query.pkl')
            a = oof_read(seed,training); mask = a['folds'] == fold
            np.testing.assert_array_equal(held.sample_id.to_numpy(str),a['ids'][mask])
            np.testing.assert_array_equal(fitting.sample_id.to_numpy(str),a['ids'][~mask])
            expected = a[arm+'_members'][INITS.index(init),mask]
        if any(c in held for c in ('tap_iron','tap_time_len')) or set(fitting.sample_id)&set(held.sample_id):
            raise ValueError('Model fit/query isolation failed')
        with forbid_training():
            model = (ComponentRegressor if arm == 'old' else SiLUEMARegressor).load(r['checkpoint'])
            trace = model.saved['trace']
            if trace['fit_rows'] != len(fitting) or trace['fit_ids_digest'] != digest(fitting.sample_id.tolist()):
                raise ValueError('Saved native training partition differs')
            got = model.predict(held)[:,0]
            np.testing.assert_array_equal(got,expected)
            test = model.predict(query)[:,0]
            reverse = model.predict(query.iloc[::-1])[::-1,0]
            chunk = np.concatenate([model.predict(query.iloc[i:i+37])[:,0] for i in range(0,len(query),37)])
            difference = float(max(np.max(abs(test-reverse)),np.max(abs(test-chunk))))
            if not np.isfinite(test).all() or difference > manifest['spec']['cold_order_chunk_atol']:
                raise ValueError('New-query cold inference failed')
            max_difference = max(max_difference,difference)
            key = f'{arm}_s{seed}_f{fold}_i{init}'
            arrays[key] = test
            traces.append(dict(**r,fit_rows=len(fitting),selected_epoch=trace['selected_epoch'],
                updates=trace['updates'],original_prediction_max_difference=0.,new_query_order_chunk_difference=difference))
            del model
            gc.collect()
    original_members.close()
    save(run/'test-replay.npz',**arrays)
    write(run/'replay.json',dict(status='passed',models=traces,maximum_order_chunk_difference=max_difference,
        new_target_fits=0,new_target_optimizers=0,predictions_sha256=sha(run/'test-replay.npz')))
    return arrays


def summarize_movements(arrays):
    full = np.mean([arrays[f'new_s-1_f-1_i{i}']-arrays[f'old_s-1_f-1_i{i}'] for i in INITS],axis=0)
    summary = {}
    for seed in SEEDS:
        folds = np.stack([np.mean([arrays[f'new_s{seed}_f{f}_i{i}']-arrays[f'old_s{seed}_f{f}_i{i}']
            for i in INITS],axis=0) for f in range(5)])
        summary[str(seed)] = dict(folds={str(f):movement_detail(folds[f],full) for f in range(5)},
            within_seed_fold_mean=movement_detail(folds.mean(0),full),
            individual_members={str(i):movement_detail(np.mean([arrays[f'new_s{seed}_f{f}_i{i}']-
                arrays[f'old_s{seed}_f{f}_i{i}'] for f in range(5)],axis=0),
                arrays[f'new_s-1_f-1_i{i}']-arrays[f'old_s-1_f-1_i{i}']) for i in INITS})
    return summary,full


def execute(run):
    import pickle
    import pandas as pd
    import resource
    from .ema_evaluation_diagnostics import sample_indices, simulated_gains, distribution, reduction_detail
    manifest = read(run/'manifest.json');spec = manifest['spec']
    environment();verify(manifest['files'])
    write(run/'started.json',dict(pid=os.getpid(),created_ns=time.time_ns(),manifest_sha256=sha(run/'manifest.json')))
    training,query = pd.read_pickle(FULL/'training.pkl'),pd.read_pickle(FULL/'query.pkl')
    if len(training)!=2754 or len(query)!=322 or any(t in query for t in ('tap_iron','tap_time_len')):
        raise ValueError('Original official train/query shape differs')
    x = np.vstack([features(training),features(query)])
    mmd,null,k,witness_masks = permutation_mmd(x,len(training),spec)
    save(run/'mmd.npz',null=null,witness_masks=witness_masks,witness_values=mmd_statistics(k,witness_masks.T))
    del k
    domain,weights,partitions,models = domain_crossfit(x,len(training),spec)
    if len(models) != spec['domain_classifier_fits']:
        raise ValueError('Domain fitting inventory differs')
    with (run/'domain-models.pkl').open('xb') as f:
        pickle.dump(models,f)
    save(run/'domain.npz',**weights)
    write(run/'domain-partitions.json',partitions)
    del models
    print(json.dumps(dict(stage='domain_complete',target_fits=0,domain_fits=10)),flush=True)
    oof = {}
    quotas = {int(k):int(v) for k,v in query.spout_no.value_counts().items()}
    for seed in SEEDS:
        a=oof_read(seed,training);y=a['actual'][:,1]
        draws = {'uniform':sample_indices(len(y),spec['draw_rows'],spec['draws'],spec['draw_seed']+seed),
            'spout':sample_indices(len(y),spec['draw_rows'],spec['draws'],spec['draw_seed']+seed,
                groups=a['spouts'],quotas=quotas)}
        save(run/f'draws-s{seed}.npz',**draws)
        oof[str(seed)] = {}
        for candidate in spec['candidates']:
            weighted = {name:weighted_gain(y,a['reference'],a[candidate],weights[name+'_weight']) for name in domain}
            raw_weighted = {name:weighted_gain(y,a['reference'],a[candidate],weights[name+'_raw_weight']) for name in domain}
            simulations = {}
            for name,d in draws.items():
                values,_ = simulated_gains(y,a['reference'],a[candidate],d)
                simulations[name]=distribution(values)
            value = dict(unweighted_gain=weighted_gain(y,a['reference'],a[candidate],np.ones(len(y))),
                weighted_gain=weighted,unclipped_weighted_gain=raw_weighted,
                draws=simulations,reduction=reduction_detail(y,a['reference'],a[candidate]))
            if abs(value['unweighted_gain']-read(DEV/'report.json')['gains'][candidate][str(seed)])>1e-10:
                raise ValueError('Original development score not reproduced')
            oof[str(seed)][candidate] = value
    print(json.dumps(dict(stage='paired_oof_complete',splits=list(SEEDS))),flush=True)
    arrays = replay(run,manifest,training,query)
    movements,full = summarize_movements(arrays)
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    if peak > spec['max_rss_mib']:
        raise ValueError('Memory gate failed')
    verify(manifest['files'])
    output = dict(status='completed_retrospective_diagnostic',G1='not_a_platform_predictor_no_promotion',
        reference=spec['reference'],mmd=mmd,domain=domain,oof=oof,movements=movements,
        platform_gain_user_reported=-.0247,full_movement_rms=float(np.sqrt(np.mean(full**2))),
        spout_quotas=quotas,new_target_fits=0,new_target_optimizers=0,domain_classifier_fits=10,
        models_replayed=66,packages=0,desktop_writes=0,agent_uploads=0,formal_promoted=False,
        peak_rss_mib=peak,manifest_sha256=sha(run/'manifest.json'))
    write(run/'report.json',output)
    write(run/'complete.json',dict(status='passed',pid=os.getpid(),report_sha256=sha(run/'report.json'),
        artifacts={p.name:sha(p) for p in run.iterdir() if p.is_file() and p.suffix in ('.npz','.pkl','.json')},
        finished_ns=time.time_ns()))
    print(json.dumps(dict(status='complete',mmd_p=mmd['p_value'],peak_rss_mib=peak)),flush=True)


def audit(run):
    """Separate process, scalar arithmetic and independent native replay; no fits."""
    import pickle
    import pandas as pd
    from unittest.mock import patch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from .ema_nested_residual import forbid_training
    manifest=read(run/'manifest.json');spec=manifest['spec'];environment();verify(manifest['files'])
    complete=read(run/'complete.json');verify({str(run/k):v for k,v in complete['artifacts'].items()})
    report=read(run/'report.json');differences=[]
    training,query=pd.read_pickle(FULL/'training.pkl'),pd.read_pickle(FULL/'query.pkl')
    x=np.vstack([features(training),features(query)])
    with (run/'domain-models.pkl').open('rb') as f:
        models=pickle.load(f)
    partitions=read(run/'domain-partitions.json')
    def forbidden(*a,**kw):
        raise ValueError('Audit attempted fitting')
    with np.load(run/'domain.npz',allow_pickle=False) as w, forbid_training(), \
            patch.object(LogisticRegression,'fit',forbidden),patch.object(HistGradientBoostingClassifier,'fit',forbidden), \
            patch.object(StandardScaler,'fit',forbidden):
        for name,fold,model in models:
            part=partitions[fold];fit=part['fit_indices'];held=part['held_indices']
            if set(fit)&set(held) or set(fit)|set(held)!=set(range(len(x))):
                raise ValueError('Domain outer partition invalid')
            np.testing.assert_array_equal(model.predict_proba(x[held])[:,1],w[name+'_probability'][held])
        for seed in SEEDS:
            a=oof_read(seed,training);y=a['actual'][:,1]
            for candidate in spec['candidates']:
                for name,weight in [('unweighted',np.ones(len(y))),*[(n,w[n+'_weight']) for n in ('logistic','histogram')]]:
                    numerator=math.fsum(float(v)*(abs(float(t)-float(b))-abs(float(t)-float(c)))
                        for v,t,b,c in zip(weight,y,a['reference'],a[candidate]))
                    value=50*numerator/math.fsum(float(v)*float(t) for v,t in zip(weight,y))
                    expected=(report['oof'][str(seed)][candidate]['unweighted_gain'] if name=='unweighted'
                        else report['oof'][str(seed)][candidate]['weighted_gain'][name])
                    differences.append(abs(value-expected))
            with np.load(run/f'draws-s{seed}.npz',allow_pickle=False) as draws:
                for name in draws.files:
                    d=draws[name]
                    if d.shape!=(10000,322) or (np.diff(np.sort(d,axis=1),axis=1)==0).any():
                        raise ValueError('Invalid paired resampling')
                    if name=='spout':
                        for g,count in report['spout_quotas'].items():
                            if not ((a['spouts'][d]==int(g)).sum(1)==count).all():
                                raise ValueError('Wrong spout quotas')
                    for candidate in spec['candidates']:
                        values=[]
                        for row in d:
                            values.append(50*math.fsum(abs(float(y[i])-float(a['reference'][i]))-
                                abs(float(y[i])-float(a[candidate][i])) for i in row)/math.fsum(float(y[i]) for i in row))
                        got=report['oof'][str(seed)][candidate]['draws'][name]
                        differences.extend([abs(float(np.mean(values))-got['mean']),
                            abs(float(np.mean(np.asarray(values)<0))-got['negative_fraction'])])
                        differences += [abs(float(np.quantile(values,float(q)))-v) for q,v in got['quantiles'].items()]
    k,_=kernel(x,spec['mmd_bandwidth_multipliers'])
    with np.load(run/'mmd.npz',allow_pickle=False) as a:
        for mask,expected in [(np.arange(len(x))>=len(training),report['mmd']['mmd2']),*zip(a['witness_masks'],a['witness_values'])]:
            left=np.flatnonzero(mask);right=np.flatnonzero(~mask)
            xx=k[np.ix_(left,left)];yy=k[np.ix_(right,right)];xy=k[np.ix_(left,right)]
            value=((math.fsum(xx.ravel())-math.fsum(np.diag(xx)))/(len(left)*(len(left)-1))
                +(math.fsum(yy.ravel())-math.fsum(np.diag(yy)))/(len(right)*(len(right)-1))
                -2*math.fsum(xy.ravel())/(len(left)*len(right)))
            differences.append(abs(value-float(expected)))
        p=(1+int(np.sum(a['null']>=report['mmd']['mmd2'])))/(1+len(a['null']))
        differences.append(abs(p-report['mmd']['p_value']))
    with np.load(run/'test-replay.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'],query.sample_id.to_numpy(str))
        movements,full=summarize_movements(a)
        if movements!=report['movements']:
            raise ValueError('Saved movement summary differs')
        rows=[]
        for p in [manifest['parent_zip'],manifest['original_zip']]:
            with zipfile.ZipFile(p) as z:
                if z.namelist()!=['result.csv'] or z.testzip() is not None:
                    raise ValueError('Original ZIP structure/CRC differs')
                rows.append(list(csv.DictReader(io.StringIO(z.read('result.csv').decode('utf-8-sig')))))
        for group in rows:
            if [r['sample_id'] for r in group]!=query.sample_id.tolist():
                raise ValueError('Original submission order differs')
        for i,(base,candidate) in enumerate(zip(*rows)):
            if base['pred_tap_iron']!=candidate['pred_tap_iron']:
                raise ValueError('Original unchanged iron strings differ')
            differences.append(abs(float(candidate['pred_tap_time_len'])-float(base['pred_tap_time_len'])-full[i]))
    if max(differences)>spec['scalar_atol']:
        raise ValueError('Independent arithmetic mismatch')
    verify(manifest['files'])
    write(run/'independent-audit.json',dict(status='passed',maximum_scalar_difference=max(differences),
        scalar_checks=len(differences),domain_models_cold=10,target_models_original_cold=66,
        report_sha256=sha(run/'report.json'),new_fits=0,pid=os.getpid()))
    print(json.dumps(dict(status='passed',maximum_scalar_difference=max(differences))),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','audit'])
    parser.add_argument('--checks')
    args=parser.parse_args();spec=read(WORK/SPEC);run=Path(spec['run_directory'])
    if args.action=='prepare':
        prepare(run,spec,args.checks)
    else:
        try:
            (execute if args.action=='run' else audit)(run)
        except BaseException as exc:
            failure=run/(args.action+'-failure.json')
            if not failure.exists():
                write(failure,dict(type=type(exc).__name__,message=str(exc),pid=os.getpid(),time_ns=time.time_ns()))
            raise


if __name__=='__main__':
    main()
