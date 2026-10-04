"""Fixed SiLU A100 full-data release using the unchanged scientific worker."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import zipfile

import numpy as np

from . import ema_silu as science
from .data import FEATURES, TARGETS, SUBMISSION_COLUMNS
from .ema_nested_residual import setup, read, sha, write, verify, memory, forbid_training, save_arrays
from .ema_span_confirmation import partition
from .submission import ZIP_NAME, package, validate_result, deny_training_reads
from .v7_periodic import digest

WORK = Path(__file__).resolve().parents[2]
MAIN = Path('/home/lux1/iron')
RUN = MAIN/'local/runs/ema-silu-release-20261004/release-r1'
SPEC = 'configs/ema_silu_release/SPEC.json'
INITS = (42, 1042, 2042)


def source_files():
    paths = list((WORK/'src').rglob('*.py')) + [WORK/p for p in
        (SPEC, 'configs/ema_silu/SPEC.json', 'docs/ema_silu_release/PREREGISTRATION.md',
         'tests/test_ema_silu_release.py', 'uv.lock', 'pyproject.toml')]
    return {str(p.relative_to(WORK)):sha(p) for p in paths}


def validate_scope(spec):
    expected = dict(candidate='SILU_A100', training_seeds=list(INITS), full_training_procedures=3,
        optimizer_runs=6, new_native_states=6, reused_native_states=6, new_CV=0,
        new_confirmation_seeds=0, packages=1, desktop_copies=1, agent_uploads=0,
        replacement_weight=1., formal_promoted=False, workers=1, numerical_threads=1,
        torch_interop_threads=1, monitor_seconds=600, max_rss_mib=1536,
        time_budget_seconds=None, automatic_retries=False, cold_full_batch_atol=0,
        cold_order_chunk_atol=.0005, scalar_atol=1e-10, pause_after_delivery=True, engineering_new_fits=0)
    if any(spec.get(k) != v or type(spec.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError('Frozen release scope differs')
    original = read(WORK/'configs/ema_silu/SPEC.json')
    if any(spec[k] != original[k] for k in ('training','mechanisms','runtime_versions','training_seeds')):
        raise ValueError('Completed SiLU scientific recipe differs')


def authority(spec):
    grant = Path(spec['explicit_grant'])
    if sha(grant) != spec['explicit_grant_sha256']:
        raise ValueError('Explicit delivery grant changed')
    value = read(grant)
    if (value['source'] != 'explicit_user_task' or not value['user_authorized']
            or value['user_request'] != '生成**SiLU A100**提交包，写桌面'
            or any(spec.get(k) != v for k,v in value['scope'].items())):
        raise ValueError('Exact explicit release and desktop scope required')


def official_frames():
    import pandas as pd
    from .v3_run import load_training_frame
    from .v2_release import load_v2
    training = load_training_frame(MAIN)
    query = load_v2(MAIN/'复赛_test','test',322)[['sample_id','spout_no',*FEATURES]].copy()
    ids = pd.read_csv(MAIN/'复赛_test/result_template.csv', dtype={'sample_id':str}).sample_id.tolist()
    if len(training) != 2754 or len(query) != 322 or query.sample_id.tolist() != ids:
        raise ValueError('Official full-data or template order differs')
    return training, query


def package_rows(path, ids):
    with zipfile.ZipFile(path) as z:
        if z.namelist() != ['result.csv'] or z.testzip() is not None:
            raise ValueError('ZIP structure or CRC differs')
        data = z.read('result.csv')
    return validate_result(data, ids), data


def prediction(parent_time, old, new):
    parent_time, old, new = [np.asarray(v,float) for v in (parent_time,old,new)]
    if (parent_time.ndim != 1 or old.shape != (3,len(parent_time)) or new.shape != old.shape
            or not all(np.isfinite(a).all() for a in (parent_time,old,new))):
        raise ValueError('Exactly three finite aligned members required')
    result = parent_time+(new.mean(0)-old.mean(0))
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid affine prediction; no clipping')
    return result


def payload(parent, values):
    if len(parent) != len(values):
        raise ValueError('Prediction count differs')
    stream = io.StringIO(newline=''); writer = csv.writer(stream,lineterminator='\n')
    writer.writerow(SUBMISSION_COLUMNS)
    for row,value in zip(parent,values):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError('Finite nonnegative prediction required')
        writer.writerow([row['sample_id'], row['pred_tap_iron'], format(float(value),'.17g')])
    return stream.getvalue().encode()


def prepare(checks):
    import psutil
    spec = read(WORK/SPEC); validate_scope(spec); authority(spec)
    if RUN.exists() or str(RUN) != spec['run_directory']:
        raise ValueError('Fresh preregistered private run required')
    best = read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best.get(k) != v for k,v in spec['platform_reference'].items()):
        raise ValueError('Current platform reference changed')
    if Path(spec['desktop_directory']).exists():
        raise FileExistsError('Desktop destination already consumed')
    c = read(checks); sources = source_files()
    if (c['status'] != 'passed' or c['sources'] != sources or c['python'] != sys.version.split()[0]
            or c['actual_exit_code'] != 0 or c['new_fits'] != 0 or c['junit_sha256'] != sha(c['junit'])):
        raise ValueError('Exact-source locked no-fit release checks required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():
        raise ValueError('Clean committed source snapshot required')
    dev = Path(spec['development_directory']); terminal = dev/'execution/final-reconciliation.json'
    f, report = read(terminal), read(dev/'report.json')
    if (sha(terminal) != spec['development_terminal_sha256'] or f['status'] != 'passed'
            or not f['owned_processes_absent'] or report['selected_for_confirmation'] != 'SILU_A100'
            or report['formal_promoted'] or not all(v > 0 for v in report['gains']['SILU_A100'].values())
            or f['report_sha256'] != sha(dev/'report.json')):
        raise ValueError('Complete development evidence required')
    files = {str(WORK/p):h for p,h in sources.items()}
    roots = [dev, Path(spec['old_intake_directory']), Path(spec['old_q100_directory'])]
    for directory in roots:
        science.merge_legacy_files(files, read(directory/'manifest.json')['files'], MAIN)
    old_intake = read(roots[1]/'manifest.json')
    for name in ('ema_silu.py','ema_silu_model.py','ema_silu_audit.py','component_regularization.py'):
        old = Path(read(dev/'manifest.json')['source_directory'])/'src/bf_tap_r2'/name
        if sha(WORK/'src/bf_tap_r2'/name) != sha(old):
            raise ValueError('Original scientific code changed: '+name)
    qterminal = read(roots[2]/'terminal-reconciliation.json')
    if (qterminal['status'] != 'passed' or any(qterminal['actual_exit_codes'].values())
            or qterminal['independent_audit_sha256'] != sha(roots[2]/'independent-audit.json')):
        raise ValueError('Old full Q100 evidence not closed')
    paths = [terminal,dev/'report.json',dev/'independent-audit.json',dev/'manifest.json',
        Path(checks),Path(c['junit']),Path(spec['explicit_grant']),Path(spec['parent_zip']),
        MAIN/best['platform_feedback_record'],MAIN/'configs/protection.yaml']
    for directory in roots[1:]:
        paths += [directory/n for n in ('manifest.json','terminal-reconciliation.json')]
    paths += [roots[1]/'cold-audit.json', roots[2]/'release.json',roots[2]/'independent-audit.json']
    old_q100 = read(roots[2]/'release.json');paths.append(Path(old_q100['zip']))
    for stage in ('train','test'):
        paths += [MAIN/f'复赛_{stage}/{stage}_{kind}.csv' for kind in ('samples','features')]
    paths.append(MAIN/'复赛_test/result_template.csv')
    for p in paths:
        science.merge_legacy_files(files,{str(p):sha(p)},MAIN)
    verify(files)
    training,query = official_frames();part = partition(training,query,spec['training'])
    if part != old_intake['original_model_partition']:
        raise ValueError('Old/new full training and inner selection partitions differ')
    parent,_ = package_rows(spec['parent_zip'],query.sample_id.tolist())
    old_rows,_ = package_rows(old_q100['zip'],query.sample_id.tolist())
    if sha(spec['parent_zip']) != spec['platform_reference']['zip_sha256'] or any(
            a['pred_tap_time_len'] != b['pred_tap_time_len'] for a,b in zip(parent,old_rows)):
        raise ValueError('Parent full EMA time column differs from audited Q100')
    available = psutil.virtual_memory().available/2**20
    if available < 3072:
        raise ValueError('Insufficient available memory')
    RUN.mkdir(parents=True,exist_ok=False)
    for name,frame in [('training.pkl',training),('query.pkl',query)]:
        with (RUN/name).open('xb') as stream:frame.to_pickle(stream)
        files[str(RUN/name)] = sha(RUN/name)
    write(RUN/'manifest.json',dict(spec=spec, files=files, sources=sources, partition=part,
        training=spec['training'],mechanisms=spec['mechanisms'],
        old_model_directories=old_intake['model_directories'], source_directory=str(WORK),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        available_memory_mib=available, created_ns=time.time_ns(), formal_promoted=False))
    print(json.dumps(dict(status='frozen',files=len(files),full_estimators=3,optimizers=6)),flush=True)


def context():
    m = read(RUN/'manifest.json');verify(m['files']);validate_scope(m['spec']);authority(m['spec'])
    if m['source_directory'] != str(WORK) or m['spec'] != read(WORK/SPEC) or (RUN/'failure.json').exists():
        raise ValueError('Bound release source differs or prior failure retained')
    return m


def frames():
    import pandas as pd
    m = context();training=pd.read_pickle(RUN/'training.pkl');query=pd.read_pickle(RUN/'query.pkl')
    if partition(training,query,m['training']) != m['partition']:
        raise ValueError('Frozen full-data partition differs')
    return m,training,query


def full_unit(seed, fold):
    if (seed,fold) != (-1,-1):
        raise ValueError('Only the frozen full-training identity is allowed')
    return frames()


def native(stage, init):
    if stage not in ('worker','cold') or init not in INITS:
        raise ValueError('Undeclared full-data task')
    with patch.object(science,'RUN',RUN), patch.object(science,'unit_frames',full_unit):
        getattr(science,stage)(-1,-1,init)


def member(init):
    return RUN/f's-1-f-1-init{init}'


def reuse():
    from .ema_nested_residual import audit_models
    m,training,query=frames();members=[];receipts={}
    for init in INITS:
        d=Path(m['old_model_directories'][str(init)])
        if init == 42:
            expected=np.load(d/'cold.npy',allow_pickle=False)[:,0]
        else:
            with np.load(d/'predictions.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['query_ids'],query.sample_id.to_numpy(str));expected=a['prediction'][:,0]
        receipts[str(init)]=audit_models(d,training,query,{'refit':expected},
            dict(m['training'],random_seed=init),m['mechanisms'])
        members.append(expected)
    save_arrays(RUN/'original-members.npz',ids=query.sample_id.to_numpy(str),members=np.asarray(members))
    write(RUN/'reuse.json',dict(status='passed',receipts=receipts,states=6,pid=os.getpid(),
        prediction_sha256=sha(RUN/'original-members.npz'),manifest_sha256=sha(RUN/'manifest.json'),new_fits=0))


def tasks():
    return [('reuse',-1)]+[(stage,i) for i in INITS for stage in ('worker','cold')]+[('build',-1),('verify',-1)]


def require_events(count):
    events=[]
    for index,(stage,init) in enumerate(tasks()[:count]):
        key=f'{index:03d}-{stage}-{init}';e=read(RUN/'execution'/(key+'-terminal.json'))
        if e['task'] != key or e['exit_code'] != 0 or not 0 < e['peak_rss_mib'] <= 1536:
            raise ValueError('Successful actual child exit required')
        events.append(e)
    return events


def label_free_query():
    import pandas as pd
    m=context()
    def deny(event,args):
        deny_training_reads(event,args)
        if event == 'open' and isinstance(args[0],(str,bytes)) and 'training.pkl' in str(args[0]):
            raise RuntimeError('Frozen training frame unavailable during inference')
    sys.addaudithook(deny)
    query=pd.read_pickle(RUN/'query.pkl')
    if any(t in query for t in TARGETS) or len(query) != 322:
        raise ValueError('322 label-free query rows required')
    return m,query


def build():
    require_events(7);m,query=label_free_query();ids=query.sample_id.tolist()
    parent,_=package_rows(m['spec']['parent_zip'],ids)
    reuse_receipt=read(RUN/'reuse.json')
    if reuse_receipt['status'] != 'passed' or reuse_receipt['prediction_sha256'] != sha(RUN/'original-members.npz'):
        raise ValueError('Original member identity differs')
    with np.load(RUN/'original-members.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ids'],np.asarray(ids));old=a['members'].copy()
    new=[]
    for init in INITS:
        d=member(init);c=read(d/'complete.json');cold=read(d/'cold.json')
        if cold['status'] != 'passed' or cold['complete_sha256'] != sha(d/'complete.json') or c['prediction_sha256'] != sha(d/'predictions.npz'):
            raise ValueError('New member cold identity differs')
        with np.load(d/'predictions.npz',allow_pickle=False) as a:
            np.testing.assert_array_equal(a['ids'],np.asarray(ids));new.append(a['prediction'])
    values=prediction([float(r['pred_tap_time_len']) for r in parent],old,new)
    directory=RUN/'package';directory.mkdir(exist_ok=False);package(directory,payload(parent,values),ids)
    write(RUN/'release.json',dict(candidate='SILU_A100',zip=str(directory/ZIP_NAME),
        zip_sha256=sha(directory/ZIP_NAME),csv_sha256=sha(directory/'result.csv'),
        manifest_sha256=sha(RUN/'manifest.json'),pid=os.getpid(),formal_promoted=False,platform_score=None))


def verify_package():
    from .component_regularization import ComponentRegressor
    from .ema_silu_model import SiLUEMARegressor
    require_events(8);m,query=label_free_query();r=read(RUN/'release.json');ids=query.sample_id.tolist()
    parent,_=package_rows(m['spec']['parent_zip'],ids);rows,data=package_rows(r['zip'],ids)
    if r['pid'] == os.getpid() or sha(r['zip']) != r['zip_sha256'] or sha(Path(r['zip']).with_name('result.csv')) != r['csv_sha256']:
        raise ValueError('Independent package process or bytes differ')
    old=[];new=[]
    with forbid_training(),patch.object(SiLUEMARegressor,'_initialize',side_effect=RuntimeError('No training')):
        for init in INITS:
            old.append(ComponentRegressor.load(Path(m['old_model_directories'][str(init)])/'refit.pt').predict(query)[:,0])
            d=member(init);c=read(d/'complete.json')
            if c['state_hashes']['refit'] != sha(d/'refit.pt'):
                raise ValueError('New full state changed')
            value=SiLUEMARegressor.load(d/'refit.pt').predict(query)[:,0]
            with np.load(d/'predictions.npz',allow_pickle=False) as a:np.testing.assert_array_equal(value,a['prediction'])
            new.append(value)
    with np.load(RUN/'original-members.npz',allow_pickle=False) as a:np.testing.assert_array_equal(old,a['members'])
    expected=[float(p['pred_tap_time_len'])+(math.fsum(float(v[i]) for v in new)/3-math.fsum(float(v[i]) for v in old)/3) for i,p in enumerate(parent)]
    difference=max(abs(float(row['pred_tap_time_len'])-v) for row,v in zip(rows,expected))
    if difference > m['spec']['scalar_atol'] or data != Path(r['zip']).with_name('result.csv').read_bytes():
        raise ValueError('Independent scalar fusion or package payload differs')
    if [row['pred_tap_iron'] for row in rows] != [row['pred_tap_iron'] for row in parent]:
        raise ValueError('Original parent iron strings changed')
    write(RUN/'package-audit.json',dict(status='passed',pid=os.getpid(),rows=322,
        release_sha256=sha(RUN/'release.json'),zip_sha256=sha(r['zip']),iron_string_mismatches=0,
        maximum_scalar_difference=difference,new_fits=0,training_inputs_read_after_boundary=0,peak_rss_mib=memory()))


def controller():
    m=context();launch=RUN/'execution';launch.mkdir(exist_ok=False)
    write(launch/'started.json',dict(pid=os.getpid(),started_ns=time.time_ns(),manifest_sha256=sha(RUN/'manifest.json')))
    stop=threading.Event();lock=threading.Lock();owned={'pid':None,'task':None};events=[]
    def monitor():
        import psutil
        number=0
        while not stop.wait(600):
            number+=1
            with lock:row=dict(number=number,time_ns=time.time_ns(),**owned)
            if row['pid'] is not None:
                try:row['rss_mib']=psutil.Process(row['pid']).memory_info().rss/2**20
                except psutil.NoSuchProcess:row['live']=False
            write(launch/f'observation-{number:03d}.json',row);print(json.dumps(row),flush=True)
    observer=threading.Thread(target=monitor,daemon=True);observer.start()
    try:
        for index,(stage,init) in enumerate(tasks()):
            key=f'{index:03d}-{stage}-{init}'
            with (launch/(key+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-m','bf_tap_r2.ema_silu_release',stage,'--init',str(init)],
                    cwd=WORK,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                with lock:owned.update(pid=child.pid,task=key)
                write(launch/(key+'-start.json'),dict(pid=child.pid,time_ns=time.time_ns()))
                _,status,usage=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
            e=dict(task=key,exit_code=code,peak_rss_mib=usage.ru_maxrss/1024,completed_ns=time.time_ns())
            write(launch/(key+'-terminal.json'),e);events.append(e)
            with lock:owned.update(pid=None,task=None)
            print(json.dumps(e),flush=True)
            if code != 0 or e['peak_rss_mib'] > 1536:raise RuntimeError('Release child failed: '+key)
        verify(m['files'])
        write(launch/'terminal.json',dict(status='passed',events=events,package_audit_sha256=sha(RUN/'package-audit.json')))
    except BaseException as error:
        write(launch/'terminal.json',dict(status='failed',events=events,error=repr(error),automatic_retry=False));raise
    finally:stop.set();observer.join()


def close(actual_code):
    import torch
    m=context();e=RUN/'execution';t=read(e/'terminal.json');events=require_events(9)
    if actual_code != 0 or t['status'] != 'passed' or t['events'] != events:
        raise ValueError('Actual successful controller exit required')
    pids=[read(e/'started.json')['pid']]+[read(e/(v['task']+'-start.json'))['pid'] for v in events]
    if len(set(pids)) != 10 or any((Path('/proc')/str(p)).exists() for p in pids):
        raise ValueError('Owned controller and children must be absent')
    steps=0;epochs={};cold_maximum=0.
    for init in INITS:
        d=member(init);c=read(d/'complete.json');a=read(d/'cold.json')
        if (c['identity'] != dict(source_directory=str(d),split_seed=-1,fold=-1,trial_id=f'SILU_EMA_INIT{init}')
                or c['manifest_sha256'] != sha(RUN/'manifest.json') or a['status'] != 'passed'
                or a['complete_sha256'] != sha(d/'complete.json') or a['pid'] == c['pid']
                or a['states'] != 2 or a['native_optimizer_constructors'] != 2
                or a['native_steps'] != sum(c['steps'].values())):
            raise ValueError('Full member inventory differs')
        traces={}
        for role,h in c['state_hashes'].items():
            if sha(d/(role+'.pt')) != h:raise ValueError('Native checkpoint changed')
            traces[role]=torch.load(d/(role+'.pt'),map_location='cpu',weights_only=True)['trace']
        science.verify_counts(c,traces);steps+=a['native_steps'];epochs[str(init)]=a['selected_epoch']
        cold_maximum=max(cold_maximum,a['maximum_difference'])
    r=read(RUN/'release.json');a=read(RUN/'package-audit.json')
    if (a['status'] != 'passed' or a['release_sha256'] != sha(RUN/'release.json')
            or t['package_audit_sha256'] != sha(RUN/'package-audit.json') or a['zip_sha256'] != sha(r['zip'])):
        raise ValueError('Package terminal binding differs')
    write(e/'final-reconciliation.json',dict(status='passed',actual_controller_exec_exit_code=actual_code,
        actual_child_exit_codes=[v['exit_code'] for v in events],owned_pids=pids,owned_processes_absent=True,
        new_estimators=3,new_optimizer_constructors=6,native_steps=steps,new_states=6,reused_states=6,
        selected_epochs=epochs,maximum_new_cold_difference=cold_maximum,
        maximum_child_rss_mib=max(v['peak_rss_mib'] for v in events),
        manifest_sha256=sha(RUN/'manifest.json'),terminal_sha256=sha(e/'terminal.json'),
        release_sha256=sha(RUN/'release.json'),package_audit_sha256=sha(RUN/'package-audit.json'),
        zip_sha256=sha(r['zip']),new_confirmation_seeds=0,packages=1,desktop_copies=0,agent_uploads=0,
        formal_promoted=False,completed_ns=time.time_ns()))
    print(json.dumps(dict(status='passed',selected_epochs=epochs,native_steps=steps,zip_sha256=sha(r['zip']))),flush=True)


def deliver():
    m=context();f=read(RUN/'execution/final-reconciliation.json');r=read(RUN/'release.json')
    if f['status'] != 'passed' or f['zip_sha256'] != sha(r['zip']) or not f['owned_processes_absent']:
        raise ValueError('Complete successful terminal required before desktop')
    target=Path(m['spec']['desktop_directory']);target.mkdir(exist_ok=False)
    with (target/ZIP_NAME).open('xb') as stream:stream.write(Path(r['zip']).read_bytes())
    write(RUN/'desktop-copy.json',dict(desktop=str(target/ZIP_NAME),zip_sha256=sha(target/ZIP_NAME),
        source_zip=r['zip'],final_reconciliation_sha256=sha(RUN/'execution/final-reconciliation.json'),
        pid=os.getpid(),desktop_copies=1,agent_uploads=0))


def desktop_audit():
    m,query=label_free_query();c=read(RUN/'desktop-copy.json');r=read(RUN/'release.json')
    if c['pid'] == os.getpid() or c['zip_sha256'] != r['zip_sha256'] or sha(c['desktop']) != r['zip_sha256']:
        raise ValueError('Independent desktop copy identity differs')
    rows,data=package_rows(c['desktop'],query.sample_id.tolist());parent,_=package_rows(m['spec']['parent_zip'],query.sample_id.tolist())
    if (Path(c['desktop']).read_bytes() != Path(r['zip']).read_bytes()
            or data != Path(r['zip']).with_name('result.csv').read_bytes()
            or [v['pred_tap_iron'] for v in rows] != [v['pred_tap_iron'] for v in parent]):
        raise ValueError('Desktop bytes or unchanged iron differs')
    write(RUN/'desktop-audit.json',dict(status='passed',pid=os.getpid(),desktop=c['desktop'],
        zip_sha256=r['zip_sha256'],rows=322,iron_string_mismatches=0,bytes_equal=True,
        desktop_copy_sha256=sha(RUN/'desktop-copy.json'),new_fits=0,agent_uploads=0))
    print(json.dumps(dict(status='passed',desktop=c['desktop'],zip_sha256=r['zip_sha256'])),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','reuse','worker','cold','build','verify','close','deliver','desktop-audit'])
    p.add_argument('--checks');p.add_argument('--init',type=int,default=-1);p.add_argument('--actual-controller-exit-code',type=int)
    a=p.parse_args();setup()
    try:
        if a.stage == 'prepare':prepare(a.checks)
        elif a.stage in ('worker','cold'):native(a.stage,a.init)
        elif a.stage == 'close':close(a.actual_controller_exit_code)
        else:{'run':controller,'reuse':reuse,'build':build,'verify':verify_package,'deliver':deliver,'desktop-audit':desktop_audit}[a.stage]()
    except BaseException as error:
        if RUN.exists():
            destination=RUN/(a.stage+'-failure-'+str(time.time_ns())+'.json')
            write(destination,dict(error=repr(error),automatic_retry=False))
        raise
