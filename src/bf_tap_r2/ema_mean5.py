"""Fixed five-member EMA extension; original trainers and runs stay immutable."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from .ema_nested_residual import (audit_models, forbid_training, memory, read,
    save_arrays, setup, sha, verify, write)
from .v7_periodic import digest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'local/runs/ema-mean5-20261003/development-r1'
NESTED = ROOT/'local/runs/ema-nested-residual-20261003/development-r1'
MEAN = ROOT/'local/runs/ema-median3-20261003/development-r1'
SOURCE = ROOT/'local/runs/ema-retraining-initialization-20261002/development-r1'
PROTOCOL = ROOT/'docs/ema_mean5/PREREGISTRATION.md'
SEEDS = (42, 3407)
OLD_INITS = (42, 1042, 2042)
NEW_INITS = (3042, 4042)
CANDIDATE = 'EMA_MEAN5'


def columns(q75, members):
    q75, members = np.asarray(q75, float), np.asarray(members, float)
    if q75.ndim != 1 or members.shape != (5, len(q75)) or not np.isfinite(members).all():
        raise ValueError('Exactly five finite aligned members required')
    mean3 = q75 + .75*(members[:3].mean(0)-members[0])
    mean5 = q75 + .75*(members.mean(0)-members[0])
    if any(not np.isfinite(v).all() or (v < 0).any() for v in (mean3, mean5)):
        raise ValueError('Invalid prediction; no clipping')
    return mean3, mean5


def eligible(gains):
    if set(gains) != {'42', '3407'} or not all(math.isfinite(v) for v in gains.values()):
        raise ValueError('Two complete finite split gains required')
    return all(v > 0 for v in gains.values())


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


def prepare(checks):
    if RUN.exists():
        raise FileExistsError('Private run already consumed')
    require_previous()
    c = read(checks)
    if (c['status'] != 'passed' or c['python'] != sys.version.split()[0]
            or c['source_sha256'] != sha(__file__) or c['test_sha256'] != sha(ROOT/'tests/test_ema_mean5.py')
            or c['junit_sha256'] != sha(c['junit'])):
        raise ValueError('Matching locked checks required')
    best = read(ROOT/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'], best['score'], best['zip_sha256']) != ('EMA_MEAN3_FULL_Q75', 96.3954,
            '016e7e9cb3f750c74509dbb51961c61fea205d0296bb233056da66005f69a396'):
        raise ValueError('Latest incumbent changed before freeze')
    n, m = read(NESTED/'manifest.json'), read(MEAN/'manifest.json')
    files = dict(n['files'])
    for p, h in m['files'].items():
        if p in files and files[p] != h:
            raise ValueError('Incompatible source inputs')
        files[p] = h
    paths = list((ROOT/'src').rglob('*.py')) + [PROTOCOL, Path(checks), Path(c['junit']),
        ROOT/'tests/test_ema_mean5.py', ROOT/best['package'], ROOT/best['platform_feedback_record'],
        NESTED/'manifest.json', NESTED/'report.json', NESTED/'execution/terminal.json',
        NESTED/'execution/independent-terminal-audit.json', NESTED/'execution/final-reconciliation.json',
        MEAN/'manifest.json', MEAN/'report.json', MEAN/'independent-audit.json', MEAN/'terminal-reconciliation.json']
    paths += [MEAN/f'oof-s{s}.npz' for s in SEEDS]
    for p in paths:
        name, h = str(p.resolve()), sha(p)
        if name in files and files[name] != h:
            raise ValueError('Previous frozen science changed: '+name)
        files[name] = h
    verify(files)
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN/'manifest.json', dict(files=files, plans=n['plans'], training=n['training'], mechanisms=n['mechanisms'],
        reference=best['candidate'], reference_score_user_reported=best['score'], reference_zip_sha256=best['zip_sha256'],
        seeds=list(SEEDS), training_seeds=list(OLD_INITS+NEW_INITS), candidate_order=[CANDIDATE], component_weight=.75,
        budget=dict(new_estimators=20, new_optimizers=40, new_states=40, reused_states=60, cold_states=100,
                    new_confirmation_seeds=0, full_fits=0, packages=0), monitor_seconds=600,
        time_budget_seconds=None, automatic_retries=False, created_ns=time.time_ns(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    print(json.dumps(dict(status='frozen', files=len(files), new_optimizers=40)), flush=True)


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
    from .component_regularization import ComponentRegressor
    from .component_regularization_run import RECIPE
    if init not in NEW_INITS:
        raise ValueError('Undeclared new training seed')
    m, training, query = unit_frames(seed, fold)
    d = RUN/f's{seed}-f{fold}-init{init}'; d.mkdir(exist_ok=False)
    identity = dict(source_directory=str(d), split_seed=seed, fold=fold, trial_id=f'EMA_INIT{init}')
    write(d/'estimator-start.json', dict(identity=identity, pid=os.getpid(), time_ns=time.time_ns(),
        training_ids=training.sample_id.tolist(), query_ids=query.sample_id.tolist()))
    try:
        settings = dict(m['training'], random_seed=init)
        with optimizer_ledger(d, identity) as counts:
            model = ComponentRegressor(RECIPE, settings, 'EMA', m['mechanisms'], d)
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
    identity = dict(source_directory=str(d), split_seed=seed, fold=fold, trial_id=f'EMA_INIT{init}')
    settings = dict(m['training'], random_seed=init)
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
            receipts[str(init)] = audit_models(source, training, query, {'refit':a['prediction']},
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
    for index, (stage, seed, fold, init) in enumerate(tasks()[:50]):
        key = f'{index:03d}-{stage}-s{seed}-f{fold}-i{init}'
        e = read(RUN/'execution'/(key+'-terminal.json'))
        if e['task'] != key or e['exit_code'] != 0 or e['peak_rss_mib'] > 1536:
            raise ValueError('Complete successful model execution required before quality reads')


def scalar_score(y, iron, pred):
    return 100-50*math.fsum(math.fsum(abs(float(a)-float(b)) for a,b in zip(y[:,j],p)) /
        math.fsum(abs(float(a)) for a in y[:,j]) for j,p in enumerate((iron,pred)))


def report():
    import yaml
    from .candidate_tiers import classify_candidates
    from .component_regularization_run import metric_detail
    verify_model_events()
    m = read(RUN/'manifest.json'); verify(m['files'])
    gains = {}; hashes = {}; states = optimizers = steps = 0
    metrics = {'tap_time_len':{'EMA_MEAN3':{}, CANDIDATE:{}}}
    for seed in SEEDS:
        with np.load(MEAN/f'oof-s{seed}.npz', allow_pickle=False) as a:
            ids, fv, y, iron = (a[k] for k in ('ids', 'folds', 'actual', 'iron'))
            if len(ids) != 2754 or len(set(ids)) != 2754 or set(fv) != set(range(5)):
                raise ValueError('Incomplete within-seed OOF')
            members = np.full((5, len(ids)), np.nan); members[:3] = a['members']
            for fold in range(5):
                old = read(RUN/f'reuse-s{seed}-f{fold}.json')
                if old['status'] != 'passed' or old['manifest_sha256'] != sha(RUN/'manifest.json'):
                    raise ValueError('Original cold coverage incomplete')
                states += old['cold_states']; mask = fv == fold
                for j, init in enumerate(NEW_INITS, 3):
                    d = RUN/f's{seed}-f{fold}-init{init}'; c = read(d/'cold.json'); w = read(d/'complete.json')
                    if c['status'] != 'passed' or c['complete_sha256'] != sha(d/'complete.json') or w['prediction_sha256'] != sha(d/'predictions.npz'):
                        raise ValueError('New cold coverage incomplete')
                    states += c['states']; optimizers += c['native_optimizer_constructors']; steps += c['native_steps']
                    with np.load(d/'predictions.npz', allow_pickle=False) as p:
                        np.testing.assert_array_equal(p['ids'], ids[mask]); members[j,mask] = p['prediction']
            base, candidate = columns(a['q75'], members)
            np.testing.assert_array_equal(base, a['mean3'])
            gains[str(seed)] = float(50*(np.abs(y[:,1]-base).sum()-np.abs(y[:,1]-candidate).sum())/np.abs(y[:,1]).sum())
            for key,p in [('EMA_MEAN3',base),(CANDIDATE,candidate)]:
                metrics['tap_time_len'][key][str(seed)] = metric_detail(y[:,1],p,fv,a['spouts'])
            path = RUN/f'oof-s{seed}.npz'
            save_arrays(path, ids=ids, folds=fv, actual=y, iron=iron, q75=a['q75'], mean3=base,
                mean5=candidate, members=members, spouts=a['spouts'])
            hashes[str(seed)] = sha(path)
    if (states, optimizers) != (100, 40):
        raise ValueError('Cold/optimizer inventory failed')
    spec = dict(split_seeds=list(SEEDS), folds=5, candidates={'tap_time_len':[CANDIDATE]},
        reference_by_target={'tap_time_len':'EMA_MEAN3'}, tie_preference_by_target={'tap_time_len':[CANDIDATE]})
    value = dict(status='completed_two_split_development', gains=gains, confirmation_eligible=eligible(gains),
        formal_promoted=False, metrics=metrics, candidate_tiers=classify_candidates(metrics,spec,
            yaml.safe_load((ROOT/'configs/candidate_tiers.yaml').read_text())), new_optimizer_constructors=optimizers,
        native_steps=steps, cold_base_states=states, manifest_sha256=sha(RUN/'manifest.json'), oof_sha256=hashes,
        new_confirmation_seeds=0, full_fits=0, packages=0)
    verify(m['files']); write(RUN/'report.json', value)
    print(json.dumps({k:value[k] for k in ['status','gains','confirmation_eligible']}), flush=True)


def audit():
    from .component_regularization import ComponentRegressor
    verify_model_events()
    m, r = read(RUN/'manifest.json'), read(RUN/'report.json'); verify(m['files'])
    if r['manifest_sha256'] != sha(RUN/'manifest.json'):
        raise ValueError('Report identity differs')
    gains = {}; maximum = 0.; truth = {}; model_steps = 0
    # Label-to-frozen-training binding, then independent saved-member arithmetic.
    for seed in SEEDS:
        for fold in range(5):
            _, training, query = unit_frames(seed, fold)
            for row in training[['sample_id',*TARGETS]].itertuples(index=False,name=None):
                if row[0] in truth and truth[row[0]] != row[1:]:
                    raise ValueError('Inconsistent original training labels')
                truth[row[0]] = row[1:]
            for init in NEW_INITS:
                d = RUN/f's{seed}-f{fold}-init{init}'; c = read(d/'complete.json')
                traces = {}
                with forbid_training():
                    for role, h in c['state_hashes'].items():
                        if sha(d/(role+'.pt')) != h:
                            raise ValueError('Post-cold state mutation')
                        traces[role] = ComponentRegressor.load(d/(role+'.pt')).saved['trace']
                verify_counts(c, traces); model_steps += sum(c['steps'].values())
    if model_steps != r['native_steps']:
        raise ValueError('Reported native steps differ')
    for seed in SEEDS:
        path = RUN/f'oof-s{seed}.npz'
        if sha(path) != r['oof_sha256'][str(seed)]:
            raise ValueError('OOF hash differs')
        with np.load(path, allow_pickle=False) as a, np.load(MEAN/f'oof-s{seed}.npz',allow_pickle=False) as old:
            for k in ('ids','folds','actual','iron','q75','mean3','spouts'):
                np.testing.assert_array_equal(a[k], old[k])
            if set(a['ids']) != set(truth) or len(a['ids']) != 2754:
                raise ValueError('Incomplete target identity')
            np.testing.assert_array_equal(a['actual'], np.asarray([truth[i] for i in a['ids']]))
            np.testing.assert_array_equal(a['members'][:3],old['members'])
            for fold in range(5):
                mask = a['folds'] == fold
                for j,init in enumerate(NEW_INITS,3):
                    d = RUN/f's{seed}-f{fold}-init{init}'
                    with np.load(d/'predictions.npz',allow_pickle=False) as p:
                        np.testing.assert_array_equal(p['ids'],a['ids'][mask])
                        np.testing.assert_array_equal(p['prediction'],a['members'][j,mask])
            expected = [float(q)+.75*(math.fsum(float(x) for x in values)/5-float(values[0]))
                for q,values in zip(a['q75'],a['members'].T)]
            maximum = max(maximum,max(abs(x-float(y)) for x,y in zip(expected,a['mean5'])))
            gain = scalar_score(a['actual'],a['iron'],expected)-scalar_score(a['actual'],a['iron'],a['mean3'])
            gains[str(seed)] = gain; maximum = max(maximum,abs(gain-r['gains'][str(seed)]))
    if maximum > 1e-11 or eligible(gains) != r['confirmation_eligible'] or r['formal_promoted']:
        raise ValueError('Independent score or selection mismatch')
    write(RUN/'independent-audit.json', dict(status='passed', gains=gains, maximum_difference=maximum,
        manifest_sha256=sha(RUN/'manifest.json'), report_sha256=sha(RUN/'report.json'), native_steps=model_steps,
        new_fits=0, new_packages=0))
    print(json.dumps(dict(status='passed',gains=gains,maximum_difference=maximum)), flush=True)


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
                child=subprocess.Popen([sys.executable,'-m','bf_tap_r2.ema_mean5',stage,'--seed',str(seed),
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
