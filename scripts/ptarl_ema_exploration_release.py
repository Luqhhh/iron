"""Two explicitly requested exploration releases using unchanged audited trainers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
import types
import zipfile

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = Path('/home/lux1/iron')
SPEC = WORKSPACE / 'configs/ptarl_ema_exploration_release/RELEASE.json'
NAMES = ('PTARL_TIME_Q20', 'EMA_IRON_EMA_TIME')
ZIP = 'Luqhhh_bf_tap_predict_round2.zip'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def config():
    c = json.loads(SPEC.read_text())
    if (tuple(c['candidates']) != NAMES or c['agent_uploads'] or c['desktop_writes']
            or c['execution']['maximum_runtime_seconds'] is not None
            or c['execution']['automatic_retries'] or c['execution']['workers'] != 1):
        raise ValueError('Frozen two-release scope required')
    return c


def activate(c):
    source = Path(c['scientific_source'])
    sys.path.insert(0, str(source / 'src'))
    import torch
    from bf_tap_r2.rfm_freeze import runtime_snapshot
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(k) != '1':
            raise ValueError('Single numerical thread required')
    torch.set_num_threads(1)
    runtime_snapshot()
    import bf_tap_r2.ptarl_execution as implementation
    if Path(implementation.__file__).resolve() != source/'src/bf_tap_r2/ptarl_execution.py':
        raise ValueError('Original scientific worktree must supply trainers')
    return source


def best_identity():
    b = json.loads((ROOT/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    return {k:b[k] for k in ('candidate','score','zip_sha256')}


def checked(out):
    m = json.loads((out/'manifest.json').read_text())
    from bf_tap_r2.rfm_freeze import runtime_snapshot
    if runtime_snapshot()!=m['scientific_runtime']:
        raise ValueError('Original scientific runtime changed')
    if best_identity() != m['best']:
        raise ValueError('Platform incumbent changed; do not silently move release reference')
    for name, expected in m['files'].items():
        if sha(name) != expected:
            raise ValueError('Frozen input/source/evidence changed: '+name)
    return m


def data():
    from bf_tap_r2.v5_library import load_v5_training_frame
    from bf_tap_r2.v2_release import load_v2
    from bf_tap_r2.data import FEATURES
    train = load_v5_training_frame(ROOT)
    test = load_v2(ROOT/'复赛_test','test',322)
    return train, test[['sample_id','spout_no',*FEATURES]].copy()


def prepare(c, out):
    source = activate(c)
    from bf_tap_r2.ptarl_unbudgeted_freeze import verify_manifest
    from bf_tap_r2.ptarl_unbudgeted_run import development_records
    from bf_tap_r2.rfm_execution import validate_partition
    original = source/'local/ptarl-unbudgeted-entry-retry-r1'
    verify_manifest(original/'manifest.json',c['ptarl_manifest_sha256'])
    development_records(original,c['ptarl_manifest_sha256'],c['ptarl_audit_sha256'],c['ptarl_arithmetic_sha256'])
    if sha(original/'terminal-verification.json') != c['ptarl_terminal_sha256']:
        raise ValueError('PTaRL terminal proof changed')
    prior = json.loads((ROOT/'configs/local_platform_diagnostic_release/RELEASE.json').read_text())
    reg = ROOT/'local/runs/strong-component-regularization/development-r2'
    rm = json.loads((reg/'manifest.json').read_text())
    for rel in ('src/bf_tap_r2/component_regularization.py','src/bf_tap_r2/component_regularization_audit.py',
                'src/bf_tap_r2/v12_joint.py','src/bf_tap_r2/v3_6_networks.py','src/bf_tap_r2/v7_periodic.py'):
        if sha(source/rel) != rm['source_hashes'][rel]:
            raise ValueError('EMA trainer differs from audited development source')
    best = best_identity()
    if best['candidate'] != c['comparison']['candidate'] or best['score'] != c['comparison']['score']:
        raise ValueError('Expected current EMA platform incumbent')
    artifacts = {key:prior['artifacts'][key] for key in ('parent','native_iron_model','native_iron_prediction','reg_manifest','reg_summary','reg_audit')}
    if best['zip_sha256']!=c['comparison']['zip_sha256']:
        raise ValueError('Current scored package identity differs from release selection')
    artifacts['incumbent'] = dict(path=c['comparison']['package'],sha256=best['zip_sha256'])
    files = {str(SPEC):sha(SPEC),str(Path(__file__).resolve()):sha(__file__)}
    for key, item in artifacts.items():
        p=ROOT/item['path']
        if sha(p)!=item['sha256']:raise ValueError('Original package/model/evidence changed: '+key)
        files[str(p)]=item['sha256']
    for base in (source/'src',source/'configs',source/'scripts',source/'tests'):
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml'):files[str(p)]=sha(p)
    for base in (original,reg):
        for p in base.rglob('*'):
            if p.is_file():files[str(p)]=sha(p)
    for stage in ('train','test'):
        for kind in ('samples','features'):
            p=ROOT/f'复赛_{stage}/{stage}_{kind}.csv';files[str(p)]=sha(p)
    for p in (ROOT/'复赛_test/result_template.csv',source/'uv.lock',source/'pyproject.toml',
              source/'configs/strong_component_regularization/SPEC.yaml'):
        files[str(p)]=sha(p)
    train,query=data();validate_partition(train,query)
    out.mkdir(parents=True,exist_ok=False)
    query.to_pickle(out/'query.pkl')
    write(out/'manifest.json',dict(config=c,files=files,artifacts=artifacts,best=best,
        original_ptarl_settings=json.loads((original/'manifest.json').read_text())['settings'],
        query_sha256=sha(out/'query.pkl'),training_rows=len(train),query_rows=len(query),
        scientific_runtime=json.loads((original/'manifest.json').read_text())['runtime'],
        historical_decisions_unchanged=True,four_seed_promoted=False))
    print(json.dumps(dict(prepared=str(out),frozen_files=len(files))),flush=True)


def memory(c):
    import psutil
    source=Path(c['scientific_source'])
    # Original resource evidence lives in the separate resource worktree.
    formal=json.loads((source/'local/ptarl-unbudgeted-entry-retry-r1/manifest.json').read_text())
    report=json.loads((Path(formal['resource_workspace'])/'local/ptarl-space-calibration-r1/preflight/admission.json').read_text())
    if psutil.virtual_memory().available/1024**2 < report['required_available_mib']:
        raise ValueError('Original non-time memory admission required')
    return report['required_available_mib']


def load_native(path):
    from bf_tap_r2.v12_release import IronView
    class Loader(pickle.Unpickler):
        def find_class(self,module,name):
            return IronView if (module,name)==('__main__','IronView') else super().find_class(module,name)
    with Path(path).open('rb') as f:return Loader(f).load()


def reserve_ema_training(model, ledger, directory):
    """Only wrap the unchanged numerical method with exclusive reservations."""
    from bf_tap_r2.component_regularization import ComponentRegressor
    def recorded(self,frame,y,epochs,validation=None):
        role='selection' if validation is not None else 'refit'
        with ledger.event('optimizer',('EMA_IRON',role),dict(rows=len(frame),maximum_epochs=epochs)) as event:
            selected=ComponentRegressor._train(self,frame,y,epochs,validation)
            event.update(selected_epoch=selected,model_sha256=sha(directory/(role+'.pt')))
        return selected
    model._train=types.MethodType(recorded,model)


def fit(c,out,name):
    source=activate(c);m=checked(out);memory(c)
    from bf_tap_r2.ptarl_protocol import ReservationLedger
    from bf_tap_r2.ptarl_execution import execute_unit
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.component_regularization_run import RECIPE
    import numpy as np
    import yaml
    train,query=data();work=out/name;work.mkdir(exist_ok=False)
    if name==NAMES[0]:
        ledger=ReservationLedger.create(work/'ledger',dict(pair_unit=1,optimizer=6,kmeans=2))
        task=dict(target='tap_time_len',seed=42,fold=0)
        result=execute_unit(task,train,query,m['original_ptarl_settings'],work/'unit',ledger.root,ledger.policy_sha256)
        with np.load(work/'unit/predictions.npz') as a:prediction=a['PTARL_AUX'].copy()
        receipt=dict(result=result,task=task,counts=ledger.inspect(),unit_sha256=sha(work/'unit/complete.json'))
    else:
        spec=yaml.safe_load((source/'configs/strong_component_regularization/SPEC.yaml').read_text())
        settings=spec['training']['tap_iron']
        ledger=ReservationLedger.create(work/'ledger',dict(pair_unit=1,optimizer=2,kmeans=0))
        native=load_native(ROOT/m['artifacts']['native_iron_model']['path'])
        from bf_tap_r2.v7_periodic import digest
        if (native.joint.settings!=settings or native.joint.recipe!=RECIPE
                or native.joint.metadata_['fit_ids_digest']!=digest(train.sample_id.tolist())):
            raise ValueError('Native full-data recipe/rows changed')
        replay=native.predict(query)
        np.testing.assert_array_equal(replay,np.load(ROOT/m['artifacts']['native_iron_prediction']['path']))
        with ledger.event('pair_unit',('EMA_IRON',),dict(rows=len(train))) as event:
            model=ComponentRegressor(RECIPE,settings,'EMA',spec['mechanisms'],work)
            reserve_ema_training(model,ledger,work)
            model.fit(train,train[['tap_iron','tap_time_len']].to_numpy())
            prediction=model.predict(query)[:,0]
            event.update(selection_sha256=sha(work/'selection.pt'),refit_sha256=sha(work/'refit.pt'))
        receipt=dict(metadata=model.metadata_,counts=ledger.inspect(),selection_sha256=sha(work/'selection.pt'),refit_sha256=sha(work/'refit.pt'))
    with (work/'warm.npy').open('xb') as f:np.save(f,prediction)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>1536:raise ValueError('Original per-worker memory bound exceeded')
    checked(out);memory(c)
    write(work/'fit.json',dict(**receipt,warm_sha256=sha(work/'warm.npy'),peak_rss_mib=peak,query_sha256=m['query_sha256'],new_CV_fits=0))
    print(json.dumps(dict(fit_complete=name,counts=receipt['counts'])),flush=True)


def audit(c,out,name):
    source=activate(c);m=checked(out);work=out/name;f=json.loads((work/'fit.json').read_text())
    import numpy as np
    import yaml
    train,query=data()
    if sha(work/'warm.npy')!=f['warm_sha256']:raise ValueError('Warm prediction changed')
    if name==NAMES[0]:
        from bf_tap_r2.ptarl_execution import audit_unit
        p,report=audit_unit(work/'unit',f['task'],train,query,m['original_ptarl_settings'],expected_sha256=f['unit_sha256'],ledger_root=work/'ledger')
        np.testing.assert_array_equal(p['PTARL_AUX'],np.load(work/'warm.npy'))
    else:
        from bf_tap_r2.component_regularization_audit import verify_saved
        from bf_tap_r2.v3_4_bags import group_safe_inner_folds
        spec=yaml.safe_load((source/'configs/strong_component_regularization/SPEC.yaml').read_text());settings=spec['training']['tap_iron']
        y=train[['tap_iron','tap_time_len']].to_numpy();mask=np.asarray(group_safe_inner_folds(train,seed=settings['inner_seed'])['fold'])!=0
        if sha(work/'selection.pt')!=f['selection_sha256'] or sha(work/'refit.pt')!=f['refit_sha256']:raise ValueError('EMA state changed')
        selected=verify_saved(work/'selection.pt',train.loc[mask].reset_index(drop=True),y[mask],'EMA',settings,spec['mechanisms'],train.loc[~mask])
        full=verify_saved(work/'refit.pt',train,y,'EMA',settings,spec['mechanisms'],expected_epoch=selected.saved['trace']['selected_epoch'])
        np.testing.assert_array_equal(full.predict(query)[:,0],np.load(work/'warm.npy'))
        report=dict(status='passed',models_checked=2,new_optimizer_calls=0,selected_epoch=selected.saved['trace']['selected_epoch'])
    from bf_tap_r2.ptarl_protocol import ReservationLedger
    ledger=ReservationLedger.open(work/'ledger',sha(work/'ledger/policy.json'));counts=ledger.inspect()
    if counts!=f['counts'] or any(counts['failed'].values()) or any(counts['incomplete'].values()):raise ValueError('Full-fit ledger not closed')
    write(work/'audit.json',dict(G0='passed',report=report,fit_sha256=sha(work/'fit.json'),counts=counts,new_fits=0))
    checked(out)


def cold(c,out,name):
    activate(c);m=checked(out);work=out/name
    import numpy as np
    import pandas as pd
    from bf_tap_r2.submission import deny_training_reads
    import torch
    def no_fit(*a,**k):raise RuntimeError('Cold audit forbids training')
    torch.optim.AdamW.step=no_fit
    sys.addaudithook(deny_training_reads)
    if sha(out/'query.pkl')!=m['query_sha256']:raise ValueError('Query changed')
    query=pd.read_pickle(out/'query.pkl')
    if any(t in query for t in ('tap_iron','tap_time_len')):raise ValueError('Cold query has targets')
    f=json.loads((work/'fit.json').read_text())
    if name==NAMES[0]:
        from bf_tap_r2.ptarl_model import PrototypeRegressor
        from bf_tap_r2.ptarl_verify import saved,oracle
        from bf_tap_r2.v49_gradients import GradientRegressor
        PrototypeRegressor.train=no_fit;GradientRegressor.train=no_fit
        complete=json.loads((work/'unit/complete.json').read_text());anchor=complete['artifacts']['PTARL_AUX_refit']
        model=PrototypeRegressor.load(work/'unit/PTARL_AUX_refit.pt',anchor)
        independent,_=oracle(saved(work/'unit/PTARL_AUX_refit.pt',anchor),query)
        predict=model.predict;atol=1e-8
    else:
        from bf_tap_r2.component_regularization import ComponentRegressor
        ComponentRegressor.fit=no_fit;ComponentRegressor._train=no_fit
        model=ComponentRegressor.load(work/'refit.pt');predict=lambda q:model.predict(q)[:,0];atol=5e-4
        independent=predict(query)
        native=load_native(ROOT/m['artifacts']['native_iron_model']['path'])
        np.testing.assert_array_equal(native.predict(query),np.load(ROOT/m['artifacts']['native_iron_prediction']['path']))
    prediction=predict(query);warm=np.load(work/'warm.npy')
    np.testing.assert_array_equal(prediction,warm)
    variants=[independent,predict(query.iloc[::-1])[::-1],np.concatenate([predict(query.iloc[i:i+37]) for i in range(0,len(query),37)]),predict(query.iloc[:1])]
    differences=[float(np.max(np.abs(v-prediction[:1] if i==3 else v-prediction))) for i,v in enumerate(variants)]
    if max(differences)>atol:raise ValueError('Original cold numerical tolerance exceeded')
    with (work/'cold.npy').open('xb') as handle:np.save(handle,prediction)
    write(work/'cold.json',dict(G0='passed',training_reads_prohibited=True,optimizer_calls=0,new_fits=0,maximum_differences=differences,tolerance=atol,prediction_sha256=sha(work/'cold.npy'),fit_sha256=sha(work/'fit.json')))


def payload(name, incumbent, v32, ids, member, native=None):
    import numpy as np
    from bf_tap_r2.submission import validate_result
    from bf_tap_r2.v5_package import payload_with_parent_other_column
    current=validate_result(incumbent,ids);parent=validate_result(v32,ids)
    member=np.asarray(member,float)
    if member.shape!=(len(ids),) or not np.isfinite(member).all():raise ValueError('Invalid member')
    if name==NAMES[0]:
        target='tap_time_len';values=.8*np.array([float(r['pred_tap_time_len']) for r in parent])+.2*member
        other='pred_tap_iron'
    elif name==NAMES[1]:
        native=np.asarray(native,float)
        if native.shape!=member.shape or not np.isfinite(native).all():raise ValueError('Invalid native component')
        target='tap_iron';values=np.array([float(r['pred_tap_iron']) for r in current])+.5*(member-native)
        other='pred_tap_time_len'
    else:raise ValueError('Unknown isolated release')
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid endpoint; no clipping allowed')
    result=payload_with_parent_other_column(ids,target,values,[r[other] for r in current])
    actual=validate_result(result,ids)
    if [r[other] for r in actual]!=[r[other] for r in current]:raise ValueError('Unchanged field strings drifted')
    np.testing.assert_array_equal([float(r['pred_'+target]) for r in actual],values)
    return result


def package_inputs(out,m):
    import pandas as pd
    if sha(out/'query.pkl')!=m['query_sha256']:raise ValueError('Query changed')
    ids=pd.read_pickle(out/'query.pkl').sample_id.tolist()
    blobs={}
    for key in ('incumbent','parent'):
        path=ROOT/m['artifacts'][key]['path']
        if sha(path)!=m['artifacts'][key]['sha256']:raise ValueError('Parent ZIP changed')
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None or z.namelist()!=['result.csv']:raise ValueError('Original ZIP invalid')
            blobs[key]=z.read('result.csv')
    return ids,blobs


def build(c,out):
    activate(c);m=checked(out)
    import numpy as np
    from bf_tap_r2.submission import package
    ids,b=package_inputs(out,m);reports=[]
    for name in NAMES:
        work=out/name;f=json.loads((work/'fit.json').read_text());a=json.loads((work/'audit.json').read_text());cold=json.loads((work/'cold.json').read_text())
        if (a['G0']!='passed' or cold['G0']!='passed' or a['fit_sha256']!=sha(work/'fit.json')
                or cold['fit_sha256']!=sha(work/'fit.json') or cold['prediction_sha256']!=sha(work/'cold.npy')):raise ValueError('Independent fit/cold admission required')
        prediction=np.load(work/'cold.npy');native=np.load(ROOT/m['artifacts']['native_iron_prediction']['path']) if name==NAMES[1] else None
        result=payload(name,b['incumbent'],b['parent'],ids,prediction,native)
        dest=work/'package';dest.mkdir(exist_ok=False);package(dest,result,ids)
        reports.append(dict(candidate=name,path=str(dest/ZIP),zip_sha256=sha(dest/ZIP),csv_sha256=sha(dest/'result.csv'),audit_sha256=sha(work/'audit.json'),cold_sha256=sha(work/'cold.json'),platform_score=None,G1='explicit_user_requested_exploration_not_four_seed_promoted'))
    write(out/'release-summary.json',dict(packages=reports,new_full_training_procedures=2,optimizer_runs=8,kmeans_calls=2,new_CV_fits=0,agent_uploads=0,desktop_writes=0,comparison=m['best']))
    checked(out)


def verify(c,out,name):
    activate(c);m=checked(out)
    import numpy as np
    from bf_tap_r2.submission import deny_training_reads
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.ptarl_model import PrototypeRegressor
    def no_fit(*a,**k):raise RuntimeError('Package replay forbids fitting')
    ComponentRegressor.fit=no_fit;PrototypeRegressor.train=no_fit
    sys.addaudithook(deny_training_reads)
    ids,b=package_inputs(out,m)
    import pandas as pd
    query=pd.read_pickle(out/'query.pkl');work=out/name
    if name==NAMES[0]:
        complete=json.loads((work/'unit/complete.json').read_text());model=PrototypeRegressor.load(work/'unit/PTARL_AUX_refit.pt',complete['artifacts']['PTARL_AUX_refit']);prediction=model.predict(query);native=None
    else:
        model=ComponentRegressor.load(work/'refit.pt');prediction=model.predict(query)[:,0];native=load_native(ROOT/m['artifacts']['native_iron_model']['path']).predict(query)
    result=payload(name,b['incumbent'],b['parent'],ids,prediction,native)
    record=next(r for r in json.loads((out/'release-summary.json').read_text())['packages'] if r['candidate']==name)
    dest=work/'package'
    with zipfile.ZipFile(dest/ZIP) as z:
        if z.testzip() is not None or z.namelist()!=['result.csv'] or z.read('result.csv')!=result:raise ValueError('Independent ZIP replay mismatch')
    if (dest/'result.csv').read_bytes()!=result or sha(dest/ZIP)!=record['zip_sha256']:raise ValueError('Release identity mismatch')
    write(work/'independent-package-audit.json',dict(G0='passed',cold_csv_byte_identical=True,rows=len(ids),unchanged_field_mismatches=0,new_fits=0,training_reads_prohibited=True,zip_sha256=record['zip_sha256']))


def supervise(c,out):
    write(out/'supervisor-started.json',dict(time_ns=time.time_ns(),runtime_limit=None,restart=False))
    try:
        for name in NAMES:
            for mode in ('fit','audit','cold'):
                subprocess.run([sys.executable,__file__,mode,'--candidate',name],check=True,cwd=WORKSPACE)
        subprocess.run([sys.executable,__file__,'build'],check=True,cwd=WORKSPACE)
        for name in NAMES:subprocess.run([sys.executable,__file__,'verify','--candidate',name],check=True,cwd=WORKSPACE)
        write(out/'completion-event.json',dict(status='completed',time_ns=time.time_ns(),packages=2,uploads=0,new_CV_fits=0))
    except BaseException as exc:
        write(out/'completion-event.json',dict(status='failed',time_ns=time.time_ns(),type=type(exc).__name__,message=str(exc),automatic_retry=False))
        raise


def monitor(out):
    """Scheduled600s observations only; no model launch or restart."""
    result=dict(time_ns=time.time_ns(),terminal=(out/'completion-event.json').exists(),candidates={})
    for name in NAMES:
        work=out/name
        result['candidates'][name]={n:(work/n).exists() for n in ('fit.json','audit.json','cold.json','independent-package-audit.json')}
        events=work/'ledger/events'
        result['candidates'][name]['optimizer_started']=len(list(events.glob('optimizer-*.started.json')))
        result['candidates'][name]['optimizer_completed']=len(list(events.glob('optimizer-*.complete.json')))
    with (out/'scheduled-monitor.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','fit','audit','cold','build','verify','supervise','monitor'));p.add_argument('--candidate',choices=NAMES)
    args=p.parse_args();c=config();out=Path(c['output'])
    if args.mode in ('fit','audit','cold','verify'):
        if args.candidate is None:p.error('--candidate required')
        globals()[args.mode](c,out,args.candidate)
    elif args.mode=='monitor':monitor(out)
    else:globals()[args.mode](c,out)


if __name__=='__main__':main()
