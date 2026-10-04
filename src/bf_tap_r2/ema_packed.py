"""Frozen native TabM-packed EMA development against DE3 + mean3 Q100."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import numpy as np

from .data import TARGETS
from .ema_nested_residual import (audit_models as audit_original_models, forbid_training, memory, read,
    save_arrays, setup, sha, verify, write)
from .v7_periodic import digest
from .ema_packed_audit import audit_models

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path('/home/lux1/iron')
RUN = MAIN/'local/runs/ema-packed-20261004/development-r1'
Q100 = MAIN/'local/runs/ema-mean3-q100-20261004/release-r1'
DE3 = MAIN/'local/runs/independent-ensemble-checkpoints-20260929/development-DE3'
NESTED = MAIN/'local/runs/ema-nested-residual-20261003/development-r1'
MEAN = MAIN/'local/runs/ema-median3-20261003/development-r1'
SOURCE = MAIN/'local/runs/ema-retraining-initialization-20261002/development-r1'
MINI = MAIN/'local/runs/ema-mini-20261004/development-r1'
PROTOCOL = ROOT/'docs/ema_packed/PREREGISTRATION.md'
SEEDS = (42, 3407)
OLD_INITS = (42, 1042, 2042)
NEW_INITS = OLD_INITS
CANDIDATES = {'PACKED_A100':1., 'PACKED_A20':.2}
REFERENCE = 'DE3_EMA_MEAN3_Q100'
MODEL = ROOT/'src/bf_tap_r2/ema_packed_model.py'


def columns(v32, v7, old_members, new_members):
    arrays = [np.asarray(v, float) for v in (v32, v7, old_members, new_members)]
    v32, v7, old, new = arrays
    if (v32.ndim != 1 or v7.shape != v32.shape or old.shape != (3, len(v32))
            or new.shape != old.shape or not all(np.isfinite(v).all() for v in arrays)):
        raise ValueError('Exactly three aligned finite old and new members required')
    base = v32 + (old.mean(0)-v7)
    result = {k:base+a*(new.mean(0)-old.mean(0)) for k,a in CANDIDATES.items()}
    if any(not np.isfinite(v).all() or (v<0).any() for v in [base,*result.values()]):
        raise ValueError('Invalid affine prediction; no clipping')
    return base, result


def choose(gains):
    if list(gains) != list(CANDIDATES):
        raise ValueError('Complete fixed candidate pool required')
    passing = [k for k,v in gains.items() if eligible(v)]
    return max(passing, key=lambda k:sum(gains[k].values())) if passing else None


def merge_legacy_files(files, incoming, root):
    """Old manifests resolve relative identities at their original repository."""
    for path, value in incoming.items():
        name = str((Path(root)/path).resolve())
        if name in files and files[name] != value:
            raise ValueError('Conflicting frozen inputs')
        files[name] = value


def eligible(gains):
    if set(gains) != {'42', '3407'} or not all(math.isfinite(v) for v in gains.values()):
        raise ValueError('Two complete finite split gains required')
    return all(v > 0 for v in gains.values())


def validate_spec(spec, training, mechanisms):
    expected_budget = dict(estimators=30, optimizers=60, new_states=60, reused_states=60,
        new_confirmation_seeds=0, full_fits=0, packages=0)
    if (spec['architecture'] != 'tabm-packed' or spec['training'] != training
            or spec['mechanisms'] != mechanisms or spec['weights'] != CANDIDATES
            or spec['candidate_order'] != list(CANDIDATES)
            or spec['training_seeds'] != list(NEW_INITS) or spec['split_seeds'] != list(SEEDS)
            or spec['folds'] != 5 or spec['scientific_budget'] != expected_budget
            or spec['workers'] != 1 or spec['numerical_threads'] != 1
            or spec['torch_interop_threads'] != 1 or spec['max_rss_mib'] != 1536
            or spec['monitor_seconds'] != 600 or spec['time_budget_seconds'] is not None
            or spec['automatic_retries'] or spec['desktop_writes'] or spec['agent_uploads']):
        raise ValueError('Frozen packed recipe, scope or resources differ')


def require_previous():
    e = NESTED/'execution'
    t, a, r = [read(e/name) for name in ('terminal.json', 'independent-terminal-audit.json', 'final-reconciliation.json')]
    if (any(v['status'] != 'passed' for v in (t, a, r))
            or r['actual_supervisor_exec_exit_code'] != 0 or r['actual_independent_audit_exit_code'] != 0
            or not r['owned_processes_absent'] or len(t['events']) != 121
            or any(v['exit_code'] != 0 for v in t['events'])
            or r['independent_audit_sha256'] != sha(e/'independent-terminal-audit.json')
            or a['original_terminal_sha256'] != sha(e/'terminal.json')
            or a['report_sha256'] != sha(NESTED/'report.json')):
        raise ValueError('Original nested batch terminal not reconciled')
    t = read(MEAN/'terminal-reconciliation.json')
    if (t['status'] != 'passed' or any(t[k] != 0 for k in
            ('actual_prepare_exit_code', 'actual_evaluation_exit_code', 'actual_independent_audit_exit_code'))
            or t['manifest_sha256'] != sha(MEAN/'manifest.json')
            or t['report_sha256'] != sha(MEAN/'report.json')
            or t['independent_audit_sha256'] != sha(MEAN/'independent-audit.json')):
        raise ValueError('Mean3 input source not reconciled')


def require_mini_closed():
    """Bind the preceding stage's actual exits, without reconsidering its quality."""
    execution = MINI/'execution'
    paths = [execution/name for name in
        ('terminal.json', 'independent-terminal-audit.json', 'final-reconciliation.json')]
    terminal, audit, receipt = map(read, paths)
    bindings = {'original_terminal_sha256': paths[0],
        'independent_terminal_audit_sha256': paths[1],
        'report_sha256': MINI/'report.json',
        'arithmetic_audit_sha256': MINI/'independent-audit.json',
        'manifest_sha256': MINI/'manifest.json'}
    expected = [f'{i:03d}-{stage}-s{seed}-f{fold}-i{init}'
        for i, (stage, seed, fold, init) in enumerate(tasks())]
    if (any(v.get('status') != 'passed' for v in (terminal, audit, receipt))
            or [e['task'] for e in terminal['events']] != expected
            or any(e['exit_code'] != 0 for e in terminal['events'])
            or audit['actual_child_exit_codes'] != [0]*72
            or any(receipt[k] != 0 for k in ('actual_supervisor_exec_exit_code',
                'actual_followup_exec_exit_code', 'actual_independent_audit_exit_code'))
            or not receipt['owned_processes_absent']
            or not audit['controller_and_owned_children_absent']
            or any(receipt[key] != sha(path) for key, path in bindings.items())
            or any(audit[key] != receipt[key] for key in
                ('original_terminal_sha256', 'report_sha256', 'arithmetic_audit_sha256', 'manifest_sha256'))):
        raise ValueError('Preceding mini stage actual terminal not reconciled')
    for pid in receipt['owned_pids']:
        path = Path('/proc')/str(pid)/'cmdline'
        try:
            command = path.read_bytes()
        except FileNotFoundError:
            continue
        if b'ema_mini' in command or b'ema-mini' in command:
            raise ValueError('Preceding mini stage process still active')
    return list(dict.fromkeys([*paths, *bindings.values()]))


def prepare(checks):
    if RUN.exists():
        raise FileExistsError('Private run already consumed')
    require_previous()
    previous_paths = require_mini_closed()
    c = read(checks)
    if (c['status'] != 'passed' or c['python'] != sys.version.split()[0]
            or c['source_sha256'] != sha(__file__) or c['model_sha256'] != sha(MODEL)
            or c['test_sha256'] != sha(ROOT/'tests/test_ema_packed.py')
            or c['junit_sha256'] != sha(c['junit'])
            or c['synthetic_estimators'] != 1 or c['optimizer_constructors'] != 2
            or c['scientific_estimators'] != 0
            or c['independent_cold']['status'] != 'passed'
            or c['independent_cold']['states'] != 2
            or c['independent_cold']['cold_pid'] == c['independent_cold']['warm_pid']
            or not 0 <= c['independent_cold']['maximum_difference'] <= .0005):
        raise ValueError('Matching locked engineering checks required')
    verify(c['files'])
    best = read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'], best['score'], best['zip_sha256']) != ('DE3_IRON_EMA_MEAN3_Q100', 96.3979,
            '528b8bf91102bea7ce120a71c560f6b021382132fb425a8f95cbfe79f9e3712c'):
        raise ValueError('Latest incumbent changed before freeze')
    n = read(NESTED/'manifest.json')
    files = {}
    for directory in (NESTED, MEAN, Q100):
        merge_legacy_files(files, read(directory/'manifest.json')['files'], MAIN)
    qa = read(Q100/'independent-audit.json')
    da = read(DE3/'audit.json')
    if (qa['status'] != 'passed' or qa['manifest_sha256'] != sha(Q100/'manifest.json')
            or da['status'] != 'passed' or da['manifest_sha256'] != sha(DE3/'manifest.json')
            or da['summary_sha256'] != sha(DE3/'summary.json')):
        raise ValueError('Original Q100/DE3 evidence not closed')
    paths = previous_paths + list((ROOT/'src').rglob('*.py')) + [PROTOCOL, ROOT/'configs/ema_packed/SPEC.json',
        Path(checks),Path(c['junit']),ROOT/'tests/test_ema_packed.py',ROOT/'uv.lock',ROOT/'pyproject.toml',
        MAIN/best['package'],MAIN/best['platform_feedback_record'],
        Path(importlib.util.find_spec('tabm').origin),
        ROOT/'src/bf_tap_r2/ema_packed_audit.py',ROOT/'scripts/audit_ema_packed_terminal.py']
    merge_legacy_files(files, c['files'], MAIN)
    for directory,names in [(NESTED,['manifest.json','report.json','execution/terminal.json',
            'execution/independent-terminal-audit.json','execution/final-reconciliation.json']),
            (MEAN,['manifest.json','report.json','independent-audit.json','terminal-reconciliation.json']),
            (Q100,['manifest.json','local-report.json','independent-audit.json']),
            (DE3,['manifest.json','summary.json','audit.json','arithmetic-audit.json'])]:
        paths += [directory/name for name in names]
    for seed in SEEDS:
        paths += [MEAN/f'oof-s{seed}.npz', Q100/f'oof-s{seed}.npz']
        for fold in range(5):
            for unit in (DE3/f'reference-s{seed}-f{fold}', DE3/f'tap_iron-DE3-s{seed}-f{fold}'):
                completion = read(unit/'complete.json')
                paths.append(unit/'complete.json')
                for name,h in completion['hashes'].items():
                    files[str(unit/name)] = h
    for p in paths:
        name,h = str(p.resolve()),sha(p)
        if name in files and files[name] != h:
            raise ValueError('Previous frozen science changed: '+name)
        files[name] = h
    verify(files)
    spec = read(ROOT/'configs/ema_packed/SPEC.json')
    validate_spec(spec, n['training'], n['mechanisms'])
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN/'manifest.json', dict(files=files, plans=n['plans'], training=n['training'], mechanisms=n['mechanisms'],
        reference=best, seeds=list(SEEDS), training_seeds=list(NEW_INITS), candidates=CANDIDATES,
        budget=dict(new_estimators=30,new_optimizers=60,new_states=60,reused_states=60,cold_states=120,
                    new_confirmation_seeds=0,full_fits=0,packages=0),monitor_seconds=600,
        time_budget_seconds=None,automatic_retries=False,created_ns=time.time_ns(),
        source_directory=str(ROOT),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    print(json.dumps(dict(status='frozen',files=len(files),new_optimizers=60)),flush=True)


def unit_frames(seed, fold):
    if seed not in SEEDS or fold not in range(5):
        raise ValueError('Undeclared outer unit')
    m = read(RUN/'manifest.json'); verify(m['files'])
    u = NESTED/f's{seed}-f{fold}'
    with (u/'training.pkl').open('rb') as f:
        training = pickle.load(f)
    with (u/'query.pkl').open('rb') as f:
        query = pickle.load(f)
    validate_frames(training, query, m['plans'][u.name])
    return m, training, query


def validate_frames(training, query, plan):
    if (any(t in query for t in TARGETS) or set(training.sample_id) & set(query.sample_id)
            or training.sample_id.tolist() != plan['training_ids'] or query.sample_id.tolist() != plan['query_ids']):
        raise ValueError('Worker training/query isolation or identity differs')


@contextmanager
def optimizer_ledger(directory, identity):
    import torch
    original = torch.optim.AdamW
    counts = dict(constructors=[], steps={})
    def counted(*args, **kwargs):
        if len(counts['constructors']) >= 2:
            raise ValueError('Optimizer budget exceeded before construction')
        role = ('selection', 'refit')[len(counts['constructors'])]
        write(directory/(role+'-optimizer-start.json'), dict(identity=identity, role=role, time_ns=time.time_ns()))
        counts['constructors'].append(role); counts['steps'][role] = 0
        opt = original(*args, **kwargs); step = opt.step
        def counted_step(*a, **kw):
            value = step(*a, **kw); counts['steps'][role] += 1; return value
        opt.step = counted_step
        return opt
    with patch.object(torch.optim, 'AdamW', counted):
        yield counts


def verify_counts(counts, traces):
    if (counts['constructors'] != ['selection', 'refit'] or set(counts['steps']) != {'selection', 'refit'}
            or any(counts['steps'][r] != traces[r]['updates'] or counts['steps'][r] < 1 for r in counts['steps'])):
        raise ValueError('Native optimizer inventory differs')


def worker(seed, fold, init):
    from .ema_packed_model import PackedEMARegressor
    from .component_regularization_run import RECIPE
    if init not in NEW_INITS:
        raise ValueError('Undeclared new training seed')
    m, training, query = unit_frames(seed, fold)
    d = RUN/f's{seed}-f{fold}-init{init}'; d.mkdir(exist_ok=False)
    identity = dict(source_directory=str(d), split_seed=seed, fold=fold, trial_id=f'PACKED_EMA_INIT{init}')
    write(d/'estimator-start.json', dict(identity=identity, pid=os.getpid(), time_ns=time.time_ns(),
        training_ids=training.sample_id.tolist(), query_ids=query.sample_id.tolist()))
    try:
        settings = dict(m['training'], random_seed=init, arch_type="tabm-packed")
        with optimizer_ledger(d, identity) as counts:
            model = PackedEMARegressor(RECIPE, settings, 'EMA', m['mechanisms'], d)
            model.fit(training.drop(columns=list(TARGETS)), training[['tap_time_len']].to_numpy())
        verify_counts(counts, model.traces)
        prediction = model.predict(query)[:, 0]
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite component')
        save_arrays(d/'predictions.npz', ids=query.sample_id.to_numpy(str), prediction=prediction)
        write(d/'complete.json', dict(identity=identity, pid=os.getpid(), settings=settings,
            manifest_sha256=sha(RUN/'manifest.json'), state_hashes={r:sha(d/(r+'.pt')) for r in ('selection','refit')},
            training_ids=training.sample_id.tolist(), query_ids=query.sample_id.tolist(),
            prediction_sha256=sha(d/'predictions.npz'), **counts, peak_rss_mib=memory()))
    except BaseException as error:
        write(d/'failure.json', dict(error=repr(error), automatic_retry=False)); raise


def cold(seed, fold, init):
    import torch
    m, training, query = unit_frames(seed, fold)
    d = RUN/f's{seed}-f{fold}-init{init}'; c = read(d/'complete.json')
    identity = dict(source_directory=str(d), split_seed=seed, fold=fold, trial_id=f'PACKED_EMA_INIT{init}')
    settings = dict(m['training'], random_seed=init, arch_type="tabm-packed")
    if (init not in NEW_INITS or c['pid'] == os.getpid() or c['identity'] != identity or c['settings'] != settings
            or c['manifest_sha256'] != sha(RUN/'manifest.json') or (d/'failure.json').exists()
            or c['training_ids'] != training.sample_id.tolist() or c['query_ids'] != query.sample_id.tolist()
            or c['prediction_sha256'] != sha(d/'predictions.npz')):
        raise ValueError('New model identity mismatch')
    traces = {}
    start = read(d/'estimator-start.json')
    for role, h in c['state_hashes'].items():
        p = d/(role+'.pt')
        if sha(p) != h:
            raise ValueError('Saved state changed')
        s = read(d/(role+'-optimizer-start.json'))
        if s['identity'] != identity or s['role'] != role or s['time_ns'] < start['time_ns']:
            raise ValueError('Optimizer pre-construction identity differs')
        traces[role] = torch.load(p, map_location='cpu', weights_only=True)['trace']
    verify_counts(c, traces)
    with np.load(d/'predictions.npz', allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'], query.sample_id.to_numpy(str))
        receipt = audit_models(d, training, query, {'refit':a['prediction']}, settings, m['mechanisms'])
    write(d/'cold.json', dict(**receipt, pid=os.getpid(), complete_sha256=sha(d/'complete.json'),
        native_optimizer_constructors=2, native_steps=sum(c['steps'].values())))


def reuse(seed, fold):
    m, training, query = unit_frames(seed, fold)
    receipts = {}; members = []
    for init in OLD_INITS:
        d = SOURCE/f's{seed}-f{fold}-init{init}-EMA'
        w, c = read(d/'warm-complete.json'), read(d/'cold-complete.json')
        if ((w['seed'], w['fold'], w['training_seed'], w['arm']) != (seed, fold, init, 'EMA')
                or c['status'] != 'passed' or c['warm_sha256'] != sha(d/'warm-complete.json')
                or w['partitions']['training'] != digest(training.sample_id.tolist())
                or w['partitions']['query'] != digest(query.sample_id.tolist())
                or w['predictions_sha256'] != sha(d/'predictions.npz')):
            raise ValueError('Original EMA identity differs')
        verify(w['checkpoint_hashes'])
        source = Path(w['paths']['refit']).parent
        if Path(w['paths']['selection']).parent != source:
            raise ValueError('Selector and refit physical source mismatch')
        with np.load(d/'predictions.npz', allow_pickle=False) as a:
            np.testing.assert_array_equal(a['query_ids'], query.sample_id.to_numpy(str))
            receipts[str(init)] = audit_original_models(source, training, query, {'refit':a['prediction']},
                dict(m['training'], random_seed=init), m['mechanisms'])
            members.append(a['prediction'].copy())
    with np.load(MEAN/f'oof-s{seed}.npz', allow_pickle=False) as a:
        mask = a['folds'] == fold
        np.testing.assert_array_equal(a['ids'][mask], query.sample_id.to_numpy(str))
        np.testing.assert_array_equal(np.stack(members), a['members'][:,mask])
    write(RUN/f'reuse-s{seed}-f{fold}.json', dict(status='passed', receipts=receipts, cold_states=6,
        manifest_sha256=sha(RUN/'manifest.json'), new_optimizers=0, peak_rss_mib=memory()))


def tasks():
    result = []
    for seed in SEEDS:
        for fold in range(5):
            result.append(('reuse', seed, fold, -1))
            for init in NEW_INITS:
                result.extend((stage, seed, fold, init) for stage in ('worker', 'cold'))
    return result + [('report', 42, 0, -1), ('audit', 42, 0, -1)]


def verify_model_events():
    for index, (stage, seed, fold, init) in enumerate(tasks()[:-2]):
        key = f'{index:03d}-{stage}-s{seed}-f{fold}-i{init}'
        e = read(RUN/'execution'/(key+'-terminal.json'))
        if e['task'] != key or e['exit_code'] != 0 or e['peak_rss_mib'] > 1536:
            raise ValueError('Complete successful model execution required before quality reads')


def scalar_score(y, iron, pred):
    return 100-50*math.fsum(math.fsum(abs(float(a)-float(b)) for a,b in zip(y[:,j],p)) /
        math.fsum(abs(float(a)) for a in y[:,j]) for j,p in enumerate((iron,pred)))


def reference_arrays(seed):
    with np.load(MEAN/f'oof-s{seed}.npz',allow_pickle=False) as a:
        data = {k:a[k].copy() for k in a.files}
    with np.load(Q100/f'oof-s{seed}.npz',allow_pickle=False) as q:
        for k in ('ids','folds','actual','members'):
            np.testing.assert_array_equal(data[k],q[k])
        data.update(v32=q['v32'].copy(),v7=q['v7'].copy(),current=q['candidate'].copy())
    iron = np.full(len(data['ids']),np.nan)
    for fold in range(5):
        mask = data['folds']==fold
        with np.load(DE3/f'reference-s{seed}-f{fold}/predictions.npz',allow_pickle=False) as r, \
                np.load(DE3/f'tap_iron-DE3-s{seed}-f{fold}/predictions.npz',allow_pickle=False) as p:
            np.testing.assert_array_equal(p['query_ids'],data['ids'][mask])
            np.testing.assert_array_equal(r['query_ids'],data['ids'][mask])
            np.testing.assert_allclose(r['tap_iron'],data['iron'][mask],rtol=0,atol=1e-10)
            np.testing.assert_array_equal(p['seed_42'],r['v12_iron'])
            expected = (p['seed_42']+p['seed_104729']+p['seed_130363'])/3
            np.testing.assert_array_equal(p['DE3'],expected)
            iron[mask] = r['tap_iron']+.5*(p['DE3']-r['v12_iron'])
    data['iron'] = iron
    return data


def report():
    import yaml
    from .candidate_tiers import classify_candidates
    from .component_regularization_run import metric_detail
    verify_model_events()
    m = read(RUN/'manifest.json'); verify(m['files'])
    gains = {k:{} for k in CANDIDATES}; hashes = {}; states = optimizers = steps = 0
    metrics = {'tap_time_len':{k:{} for k in [REFERENCE,*CANDIDATES]}}
    for seed in SEEDS:
        a = reference_arrays(seed); ids,fv,y,iron = (a[k] for k in ('ids','folds','actual','iron'))
        if len(ids)!=2754 or len(set(ids))!=2754 or set(fv)!=set(range(5)):
            raise ValueError('Incomplete within-seed OOF')
        members = np.full((3,len(ids)),np.nan)
        for fold in range(5):
            old = read(RUN/f'reuse-s{seed}-f{fold}.json')
            if old['status']!='passed' or old['manifest_sha256']!=sha(RUN/'manifest.json'):
                raise ValueError('Original cold coverage incomplete')
            states += old['cold_states']; mask=fv==fold
            for j,init in enumerate(NEW_INITS):
                d=RUN/f's{seed}-f{fold}-init{init}'; c=read(d/'cold.json'); w=read(d/'complete.json')
                if c['status']!='passed' or c['complete_sha256']!=sha(d/'complete.json') or w['prediction_sha256']!=sha(d/'predictions.npz'):
                    raise ValueError('New cold coverage incomplete')
                states+=c['states'];optimizers+=c['native_optimizer_constructors'];steps+=c['native_steps']
                with np.load(d/'predictions.npz',allow_pickle=False) as p:
                    np.testing.assert_array_equal(p['ids'],ids[mask]);members[j,mask]=p['prediction']
        base,candidates = columns(a['v32'],a['v7'],a['members'],members)
        np.testing.assert_array_equal(base,a['current'])
        metrics['tap_time_len'][REFERENCE][str(seed)] = metric_detail(y[:,1],base,fv,a['spouts'])
        for name,p in candidates.items():
            gains[name][str(seed)] = float(50*np.sum(np.abs(y[:,1]-base)-np.abs(y[:,1]-p))/np.abs(y[:,1]).sum())
            metrics['tap_time_len'][name][str(seed)] = metric_detail(y[:,1],p,fv,a['spouts'])
        path=RUN/f'oof-s{seed}.npz'
        save_arrays(path,ids=ids,folds=fv,actual=y,iron=iron,reference=base,v32=a['v32'],v7=a['v7'],
            old_members=a['members'],new_members=members,spouts=a['spouts'],**candidates)
        hashes[str(seed)]=sha(path)
    if (states,optimizers)!=(120,60):
        raise ValueError('Cold/optimizer inventory failed')
    spec=dict(split_seeds=list(SEEDS),folds=5,candidates={'tap_time_len':list(CANDIDATES)},
        reference_by_target={'tap_time_len':REFERENCE},tie_preference_by_target={'tap_time_len':list(CANDIDATES)})
    value=dict(status='completed_two_split_development',gains=gains,selected_for_confirmation=choose(gains),
        formal_promoted=False,metrics=metrics,candidate_tiers=classify_candidates(metrics,spec,
        yaml.safe_load((ROOT/'configs/candidate_tiers.yaml').read_text())),new_optimizer_constructors=optimizers,
        native_steps=steps,cold_base_states=states,manifest_sha256=sha(RUN/'manifest.json'),oof_sha256=hashes,
        new_confirmation_seeds=0,full_fits=0,packages=0)
    verify(m['files']);write(RUN/'report.json',value)
    print(json.dumps({k:value[k] for k in ['status','gains','selected_for_confirmation']}),flush=True)


def audit():
    verify_model_events()
    m,r=read(RUN/'manifest.json'),read(RUN/'report.json');verify(m['files'])
    if r['manifest_sha256']!=sha(RUN/'manifest.json'):
        raise ValueError('Report identity differs')
    truth={}; maximum=0.; gains={k:{} for k in CANDIDATES}; steps=0
    for seed in SEEDS:
        for fold in range(5):
            _,training,_=unit_frames(seed,fold)
            for row in training[['sample_id',*TARGETS]].itertuples(index=False,name=None):
                if row[0] in truth and truth[row[0]]!=row[1:]:raise ValueError('Target binding differs')
                truth[row[0]]=row[1:]
        with np.load(RUN/f'oof-s{seed}.npz',allow_pickle=False) as a:
            if sha(RUN/f'oof-s{seed}.npz')!=r['oof_sha256'][str(seed)]:raise ValueError('OOF identity changed')
            np.testing.assert_array_equal(a['actual'],np.asarray([truth[k] for k in a['ids']]))
            source=reference_arrays(seed)
            np.testing.assert_array_equal(a['iron'],source['iron'])
            np.testing.assert_array_equal(a['old_members'],source['members'])
            for fold in range(5):
                mask=a['folds']==fold
                for j,init in enumerate(NEW_INITS):
                    d=RUN/f's{seed}-f{fold}-init{init}'; w=read(d/'complete.json');c=read(d/'cold.json')
                    if c['complete_sha256']!=sha(d/'complete.json') or w['prediction_sha256']!=sha(d/'predictions.npz'):
                        raise ValueError('Model prediction evidence changed')
                    for role,h in w['state_hashes'].items():
                        if sha(d/(role+'.pt'))!=h:raise ValueError('Native model identity changed')
                    steps+=sum(w['steps'].values())
                    with np.load(d/'predictions.npz',allow_pickle=False) as p:
                        np.testing.assert_array_equal(p['ids'],a['ids'][mask])
                        np.testing.assert_array_equal(p['prediction'],a['new_members'][j,mask])
            base=[float(b)+math.fsum(map(float,old))/3-float(v) for b,v,old in
                  zip(a['v32'],a['v7'],a['old_members'].T)]
            maximum=max(maximum,max(abs(x-float(y)) for x,y in zip(base,a['reference'])))
            for name,weight in CANDIDATES.items():
                expected=[b+weight*(math.fsum(map(float,new))/3-math.fsum(map(float,old))/3)
                          for b,new,old in zip(base,a['new_members'].T,a['old_members'].T)]
                maximum=max(maximum,max(abs(x-float(y)) for x,y in zip(expected,a[name])))
                gain=scalar_score(a['actual'],a['iron'],expected)-scalar_score(a['actual'],a['iron'],base)
                gains[name][str(seed)]=gain;maximum=max(maximum,abs(gain-r['gains'][name][str(seed)]))
    if maximum>1e-10 or choose(gains)!=r['selected_for_confirmation'] or r['formal_promoted'] or steps!=r['native_steps']:
        raise ValueError('Independent arithmetic/decision/inventory differs')
    write(RUN/'independent-audit.json',dict(status='passed',gains=gains,maximum_difference=maximum,
        manifest_sha256=sha(RUN/'manifest.json'),report_sha256=sha(RUN/'report.json'),new_fits=0,packages=0))
    print(json.dumps(dict(status='passed',gains=gains,maximum_difference=maximum)),flush=True)


def controller():
    m = read(RUN/'manifest.json'); verify(m['files']); launch = RUN/'execution'; launch.mkdir(exist_ok=False)
    write(launch/'started.json',dict(pid=os.getpid(),started_ns=time.time_ns(),manifest_sha256=sha(RUN/'manifest.json')))
    stop = threading.Event(); lock = threading.Lock(); owned = {'pid':None,'task':None}; events = []
    def monitor():
        import psutil
        number = 0
        while not stop.wait(600):
            number += 1
            with lock:
                row = dict(number=number,time_ns=time.time_ns(),**owned)
            if row['pid'] is not None:
                try:
                    p=psutil.Process(row['pid']);row.update(live=p.is_running(),rss_mib=p.memory_info().rss/1024**2)
                except psutil.NoSuchProcess:
                    row['live']=False
            write(launch/f'observation-{number:03d}.json',row);print(json.dumps(row),flush=True)
    observer = threading.Thread(target=monitor,daemon=True); observer.start()
    try:
        for index,(stage,seed,fold,init) in enumerate(tasks()):
            key=f'{index:03d}-{stage}-s{seed}-f{fold}-i{init}'
            with (launch/(key+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-m','bf_tap_r2.ema_packed',stage,'--seed',str(seed),
                    '--fold',str(fold),'--init',str(init)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                with lock:
                    owned.update(pid=child.pid,task=key)
                write(launch/(key+'-start.json'),dict(pid=child.pid,time_ns=time.time_ns()))
                _,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
            e=dict(task=key,exit_code=code,peak_rss_mib=usage.ru_maxrss/1024,completed_ns=time.time_ns())
            write(launch/(key+'-terminal.json'),e);events.append(e)
            with lock:
                owned.update(pid=None,task=None)
            print(json.dumps(e),flush=True)
            if code!=0 or e['peak_rss_mib']>1536:
                raise RuntimeError('Scientific child failed: '+key)
        verify(m['files'])
        write(launch/'terminal.json',dict(status='passed',events=events,report_sha256=sha(RUN/'report.json'),
            independent_audit_sha256=sha(RUN/'independent-audit.json'),dependencies_unchanged=True))
    except BaseException as error:
        write(launch/'terminal.json',dict(status='failed',events=events,error=repr(error),automatic_retry=False));raise
    finally:
        stop.set();observer.join()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','reuse','worker','cold','report','audit'])
    p.add_argument('--checks');p.add_argument('--seed',type=int,default=42);p.add_argument('--fold',type=int,default=0)
    p.add_argument('--init',type=int,default=-1);a=p.parse_args();setup()
    if a.stage=='prepare':prepare(a.checks)
    elif a.stage=='run':controller()
    elif a.stage=='reuse':reuse(a.seed,a.fold)
    elif a.stage in ('worker','cold'):{'worker':worker,'cold':cold}[a.stage](a.seed,a.fold,a.init)
    else:{'report':report,'audit':audit}[a.stage]()
