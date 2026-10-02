"""One standing-authorized manual SAM-EMA exploration, with native accounting."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'src'))
sys.path.insert(0,str(WORK/'scripts'))
import numpy as np
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.ema_reference_artifacts import cold_only, sha, write_new
from bf_tap_r2.ema_reference_ledger import binding_sources, reference_bindings
from bf_tap_r2.ema_span_confirmation import partition
from bf_tap_r2.ema_training_scale import require_memory
from bf_tap_r2.sam_ema_development import runtime
from bf_tap_r2.v7_periodic import digest
from ema_short_release import release_payload, zip_payload
from observe_ema_dropout_development import wait_actual

SPEC = 'configs/sam_ema_release/SPEC.json'
read = lambda p:json.loads(Path(p).read_text())


def validate_scope(spec, science):
    expected = dict(candidate='SAM_EMA_TIME_Q75_EXPLORATION',target='tap_time_len',replacement_weight=.75,
        full_training_procedures=1,torch_optimizer=2,new_CV=0,new_confirmation_seeds=0,packages=1,
        desktop_writes=0,agent_uploads=0,workers=1,numerical_threads=1,torch_interop_threads=1,monitor_seconds=600,
        max_worker_rss_mib=1536,cold_predict_atol=.0005,maximum_runtime_seconds=None,
        automatic_scientific_retries=False,formal_promotion=False,full_scope_identity_seed=-1,full_scope_identity_fold=-1,
        authorization_mode='standing_user_optimization_grant_manual_platform_interaction_exploration')
    if any(spec.get(k)!=v or type(spec.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError('Unregistered manual exploration scope')
    if any(spec[k]!=science[k] for k in ('training','mechanisms','runtime_versions')):
        raise ValueError('Scientific SAM-EMA recipe changed')


def sources(work):
    work=Path(work);paths=list((work/'src').rglob('*.py'))+[work/p for p in (
        SPEC,'configs/sam_ema_time/SPEC.json','docs/sam_ema_release/PREREGISTRATION.md',
        'scripts/sam_ema_release.py','scripts/ema_short_release.py','scripts/observe_ema_dropout_development.py',
        'tests/test_sam_ema_release.py','tests/test_sam_ema.py','tests/test_sam_ema_development.py',
        'tests/test_component_regularization.py','tests/test_ema_short_release.py','uv.lock','pyproject.toml')]
    return {str(p.resolve()):sha(p) for p in paths}


def verify_files(files):
    for path,h in files.items():
        if sha(path)!=h:raise ValueError(f'Frozen identity changed: {path}')


def current_reference(spec):
    best=read(Path(spec['main_root'])/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best.get(k)!=v for k,v in spec['platform_reference'].items()):raise ValueError('Current platform reference changed')


def official_frames(spec):
    import pandas as pd
    from bf_tap_r2.v3_run import load_training_frame
    from bf_tap_r2.v2_release import load_v2
    main=Path(spec['main_root']);training=load_training_frame(main)
    query=load_v2(main/'复赛_test','test',322)[['sample_id','spout_no',*FEATURES]].copy()
    ids=pd.read_csv(main/'复赛_test/result_template.csv',dtype={'sample_id':str}).sample_id.tolist()
    if len(training)!=2754 or len(query)!=322 or query.sample_id.tolist()!=ids:
        raise ValueError('Official training/query/template identity differs')
    return training,query


def development_evidence(spec):
    verify_files(spec['scientific_evidence']);dev=Path(spec['scientific_development'])
    report=read(dev/'report.json');audit=read(dev/'independent-score.json');terminal=read(dev/'terminal-verification.json');actual=read(dev/'actual-main-exit.json')
    if (audit['status']!='passed' or terminal['status']!='passed' or actual['status']!='passed'
            or terminal['actual_exit_codes']!=[0,0] or actual['actual_supervisor_tool_exit_code']!=0
            or actual['terminal_sha256']!=sha(dev/'terminal-verification.json')
            or audit['report_sha256']!=sha(dev/'report.json') or audit['manifest_sha256']!=sha(dev/'manifest.json')
            or terminal['report_sha256']!=sha(dev/'report.json') or terminal['independent_score_sha256']!=sha(dev/'independent-score.json')
            or terminal['manifest_sha256']!=sha(dev/'manifest.json')
            or audit['actual_optimizer_runs']!=20 or audit['new_saved_states']!=20 or audit['old_saved_states']!=40
            or report['formal_promotion'] or report['confirmation_eligible']
            or set(report['records'])!={'42','3407'}
            or not all(r['gains']['SAM_EMA_TIME']<0 and r['matched_sam_gain']>0 for r in report['records'].values())):
        raise ValueError('Frozen complete development contrasts and actual G0 required')
    return read(dev/'manifest.json')['files']


def prepare(work,out,tests):
    work=Path(work).resolve();out=Path(out).resolve();spec=read(work/SPEC);science=read(work/spec['scientific_config'])
    validate_scope(spec,science);current_reference(spec)
    if out.exists() or not out.is_relative_to(Path(spec['main_root'])/'local/runs'):raise ValueError('Fresh private run required')
    if sha(work/spec['scientific_config'])!=spec['scientific_config_sha256']:raise ValueError('Science configuration changed')
    source=sources(work);receipt=read(tests)
    if receipt['status']!='passed' or receipt['sources']!=source or receipt['skipped']!=0:
        raise ValueError('Exact-source locked release checks required')
    if subprocess.check_output(['git','status','--porcelain','--',*source],cwd=work,text=True).strip():raise ValueError('Commit tested source before freeze')
    grant=Path(spec['main_root'])/spec['standing_grant']
    if sha(grant)!=spec['standing_grant_sha256']:raise ValueError('Standing authorization changed')
    inputs=development_evidence(spec)|spec['scientific_evidence']|spec['old_inputs']|{str(grant):sha(grant)}
    main=Path(spec['main_root']);inputs[str(main/'configs/protection.yaml')]=sha(main/'configs/protection.yaml')
    for stage in ('train','test'):
        for kind in ('samples','features'):
            p=main/f'复赛_{stage}/{stage}_{kind}.csv';inputs[str(p)]=sha(p)
    p=main/'复赛_test/result_template.csv';inputs[str(p)]=sha(p);verify_files(inputs)
    training,query=official_frames(spec);part=partition(training,query,spec['training']);old=Path(spec['old_ema_directory'])
    if read(old/'fit.json')['metadata']['fit_ids_digest']!=part['training']:raise ValueError('Old full EMA training partition differs')
    import pandas as pd
    if digest(pd.read_pickle(old/'query.pkl').to_dict(orient='list'))!=part['query_frame']:raise ValueError('Old full query identity differs')
    zip_payload(spec['parent_zip'],spec['platform_reference']['zip_sha256'],query.sample_id.tolist())
    versions=runtime(spec);available=require_memory(spec);models=source|binding_sources(reference_bindings());verify_files(models)
    out.mkdir(parents=True,exist_ok=False)
    with (out/'query.pkl').open('xb') as f:query.to_pickle(f)
    write_new(out/'manifest.json',dict(spec=spec,workspace=str(work),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip(),
        sources=source,inputs=inputs,model_sources=models,partition=part,query_sha256=sha(out/'query.pkl'),tests_receipt_sha256=sha(tests),
        versions=versions,available_memory_mib=available,created_ns=time.time_ns(),formal_promotion=False))


def context(out,cold=False):
    out=Path(out).resolve();m=read(out/'manifest.json');spec=m['spec'];work=Path(m['workspace'])
    if not out.is_relative_to(Path(spec['main_root'])/'local/runs') or spec!=read(work/SPEC):raise ValueError('Bound release context differs')
    verify_files(m['sources']);verify_files(m['model_sources']);verify_files({p:h for p,h in m['inputs'].items() if not cold or not p.lower().endswith('.csv')})
    validate_scope(spec,read(work/spec['scientific_config']));runtime(spec)
    if sha(out/'query.pkl')!=m['query_sha256']:raise ValueError('Frozen query changed')
    if (out/'failure.json').exists():raise ValueError('Failed release retained; no retry')
    return m


def peak(spec):
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if value>spec['max_worker_rss_mib']:raise ValueError('Memory gate failed')
    return value


def warm(out):
    from bf_tap_r2.sam_ema_release_models import fit_component
    m=context(out);s=m['spec'];training,query=official_frames(s)
    if partition(training,query,s['training'])!=m['partition']:raise ValueError('Frozen partitions changed')
    identity=dict(source_directory=str(Path(out).resolve()),split_seed=-1,fold=-1,trial_id='SAM_EMA_FULL')
    _,receipt=fit_component(Path(out)/'SAM_EMA_FULL',training,query,identity=identity,settings=s['training'],mechanisms=s['mechanisms'],source_hashes=m['model_sources'])
    if receipt['native_counts']['torch_optimizer']!=2 or any(v for k,v in receipt['native_counts'].items() if k!='torch_optimizer'):raise ValueError('Native budget changed')
    write_new(Path(out)/'warm-complete.json',dict(status='passed',pid=os.getpid(),manifest_sha256=sha(Path(out)/'manifest.json'),component_sha256=sha(Path(out)/'SAM_EMA_FULL/complete.json'),peak_rss_mib=peak(s)))


def cold(out,receipt):
    from bf_tap_r2.sam_ema_release_models import audit_component
    m=context(out,cold=True);out=Path(out)
    if sha(out/'warm-complete.json')!=receipt:raise ValueError('External warm receipt changed')
    w=read(out/'warm-complete.json')
    if w['pid']==os.getpid():raise ValueError('Independent cold process required')
    with cold_only():audit=audit_component(out/'SAM_EMA_FULL',w['component_sha256'])
    if audit['retained_states']!=2:raise ValueError('Two selected/refit states required')
    write_new(out/'cold-complete.json',dict(status='passed',warm_sha256=receipt,component_sha256=sha(out/'SAM_EMA_FULL/cold-complete.json'),peak_rss_mib=peak(m['spec']),training_csv_reads=0,new_fits=0))


def audit(out):
    from bf_tap_r2.sam_ema_audit import verify_sam_ema_saved
    from bf_tap_r2.component_regularization_audit import verify_saved
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    out=Path(out);m=context(out);s=m['spec'];training,query=official_frames(s);inner=np.asarray(group_safe_inner_folds(training,seed=42)['fold'])
    fitting=training.loc[inner!=0].reset_index(drop=True);validation=training.loc[inner==0]
    if partition(training,query,s['training'])!=m['partition']:raise ValueError('Audit partition differs')
    models=[]
    for directory,arm,verify in [(out/'SAM_EMA_FULL','SAM_EMA',verify_sam_ema_saved),(Path(s['old_ema_directory']),'EMA',verify_saved)]:
        selector=verify(directory/'selection.pt',fitting,fitting[['tap_time_len']].to_numpy(),arm,s['training'],s['mechanisms'],validation)
        model=verify(directory/'refit.pt',training,training[['tap_time_len']].to_numpy(),arm,s['training'],s['mechanisms'],expected_epoch=selector.saved['trace']['selected_epoch']);models.append(model)
    component=read(out/'SAM_EMA_FULL/complete.json');metadata=component['model_metadata']
    if metadata['fit_ids_digest']!=m['partition']['training'] or metadata['selected_epoch']!=models[0].saved['trace']['selected_epoch']:raise ValueError('Full native metadata differs')
    with np.load(out/'SAM_EMA_FULL/predictions.npz',allow_pickle=False) as values:
        np.testing.assert_array_equal(values['query_ids'],query.sample_id.to_numpy(str));np.testing.assert_array_equal(models[0].predict(query),values['prediction'])
    np.testing.assert_array_equal(models[1].predict(query),np.load(Path(s['old_ema_directory'])/'cold.npy',allow_pickle=False))
    write_new(out/'native-audit.json',dict(status='passed',new_native_states=2,old_native_states=2,optimizer_runs=2,selected_epoch=metadata['selected_epoch'],cold_sha256=sha(out/'cold-complete.json'),peak_rss_mib=peak(s),new_fits=0,formal_promotion=False))


def build(out):
    out=Path(out);m=context(out,cold=True);s=m['spec'];import pandas as pd
    query=pd.read_pickle(out/'query.pkl');ids=query.sample_id.tolist();parent=zip_payload(s['parent_zip'],s['platform_reference']['zip_sha256'],ids)
    with np.load(out/'SAM_EMA_FULL/predictions.npz',allow_pickle=False) as z:new=z['prediction'][:,0]
    old=np.load(Path(s['old_ema_directory'])/'cold.npy',allow_pickle=False)[:,0]
    payload=release_payload(parent,ids,old,new);directory=out/'package';directory.mkdir(exist_ok=False)
    with (directory/'result.csv').open('xb') as f:f.write(payload)
    path=directory/'Luqhhh_bf_tap_predict_round2.zip'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:archive.writestr('result.csv',payload)
    write_new(out/'release.json',dict(candidate=s['candidate'],zip=str(path),zip_sha256=sha(path),result_sha256=sha(directory/'result.csv'),parent_sha256=s['platform_reference']['zip_sha256'],formal_promotion=False,platform_score=None))


def verify(out,receipt):
    import pandas as pd
    from bf_tap_r2.sam_ema import SAMEMARegressor
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.submission import validate_result
    out=Path(out);m=context(out,cold=True);s=m['spec'];query=pd.read_pickle(out/'query.pkl');ids=query.sample_id.tolist()
    if sha(out/'release.json')!=receipt:raise ValueError('External package receipt changed')
    release=read(out/'release.json');payload=zip_payload(release['zip'],release['zip_sha256'],ids);rows=validate_result(payload,ids)
    parent=validate_result(zip_payload(s['parent_zip'],s['platform_reference']['zip_sha256'],ids),ids)
    with cold_only():
        new=SAMEMARegressor.load(out/'SAM_EMA_FULL/refit.pt').predict(query)[:,0]
        old=ComponentRegressor.load(Path(s['old_ema_directory'])/'refit.pt').predict(query)[:,0]
    expected=[float(row['pred_tap_time_len'])+.75*(float(a)-float(b)) for row,a,b in zip(parent,new,old)]
    np.testing.assert_array_equal([float(row['pred_tap_time_len']) for row in rows],expected)
    if [r['pred_tap_iron'] for r in rows]!=[r['pred_tap_iron'] for r in parent]:raise ValueError('Parent iron field strings changed')
    if payload!=(out/'package/result.csv').read_bytes():raise ValueError('ZIP/result payload differs')
    write_new(out/'package-audit.json',dict(status='passed',rows=322,zip_sha256=release['zip_sha256'],release_sha256=receipt,unmodified_iron_field_string_mismatches=0,arithmetic_difference=0,new_fits=0,training_csv_reads=0,peak_rss_mib=peak(s)))


def execute(out):
    out=Path(out);m=context(out);current_reference(m['spec']);require_memory(m['spec'])
    write_new(out/'activation.json',dict(pid=os.getpid(),started_ns=time.time_ns(),manifest_sha256=sha(out/'manifest.json')))
    try:
        for mode in ('warm','cold','audit','build','verify'):
            command=[sys.executable,str(Path(m['workspace'])/'scripts/sam_ema_release.py'),mode,'--output',str(out)]
            if mode in ('cold','verify'):command+=['--receipt',sha(out/('warm-complete.json' if mode=='cold' else 'release.json'))]
            subprocess.run(command,cwd=m['workspace'],check=True)
        verify_files(m['sources']);verify_files(m['inputs'])
        write_new(out/'completion-event.json',dict(status='passed',optimizer_runs=2,full_training_procedures=1,new_CV=0,new_confirmation_seeds=0,packages=1,formal_promotion=False,release_sha256=sha(out/'release.json'),native_audit_sha256=sha(out/'native-audit.json'),package_audit_sha256=sha(out/'package-audit.json')))
    except BaseException as error:
        write_new(out/'failure.json',dict(error=repr(error),automatic_retries=0));raise


def observe(out):
    out=Path(out);m=context(out);env=dict(os.environ,PYTHONPATH=str(WORK/'src'))
    with (out/'controller.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__)),'execute','--output',str(out)],cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        launch=dict(supervisor_pid=os.getpid(),controller_pid=child.pid,started_ns=time.time_ns(),monitor_seconds=600);write_new(out/'process-launch.json',launch);print(json.dumps(launch),flush=True)
        terminal=wait_actual(child,out,'release')
    result=dict(status='passed' if terminal['exit_code']==0 else 'failed',actual_main=terminal,manifest_sha256=sha(out/'manifest.json'),automatic_retries=0)
    if terminal['exit_code']==0:
        complete=read(out/'completion-event.json')
        if complete['status']!='passed' or complete['optimizer_runs']!=2:raise ValueError('Full release completion budget differs')
        result.update(completion_sha256=sha(out/'completion-event.json'),release_sha256=sha(out/'release.json'),package_audit_sha256=sha(out/'package-audit.json'),optimizer_runs=2,packages=1)
    write_new(out/'terminal-verification.json',result);print(json.dumps(result),flush=True);return terminal['exit_code']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','observe','execute','warm','cold','audit','build','verify']);parser.add_argument('--output',required=True);parser.add_argument('--tests');parser.add_argument('--receipt');args=parser.parse_args()
    if args.mode=='prepare':prepare(WORK,args.output,args.tests)
    elif args.mode=='observe':raise SystemExit(observe(args.output))
    elif args.mode in ('cold','verify'):globals()[args.mode](args.output,args.receipt)
    else:globals()[args.mode](args.output)
