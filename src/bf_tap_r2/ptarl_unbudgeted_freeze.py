"""User-authorized non-time admission; preserve the complete original recipe."""
from pathlib import Path
import json
import os
import subprocess
import sys

import psutil
import yaml

from .ptarl_freeze import SPEC,validate_spec,reload_references
from .rfm_freeze import source_snapshot,runtime_snapshot,private_path,load_anchored,require_threads
from .ptarl_preflight import resource_decision
from .ptarl_protocol import file_hash,write_new

AUTHORITY='configs/ptarl_space_calibration/RUNTIME_AUTHORITY.yaml'
ORIGINAL_MANIFEST_SHA='5722fd914c6239c4dfa9228a58f9b376e4d1593dcee40ab105107217dd93aa3c'
ORIGINAL_ADMISSION_SHA='e0b26300ceee5e3b20946ec55756ddcdb5ea9afc33ef3d45d51d2b9f0c1e8380'
NON_TIME=('available_memory','cold_inference','synthetic_quality','worker_peak')
REQUIRED=(AUTHORITY,'src/bf_tap_r2/ptarl_unbudgeted_freeze.py','src/bf_tap_r2/ptarl_unbudgeted_run.py','src/bf_tap_r2/ptarl_unbudgeted_controller.py',
    'src/bf_tap_r2/ptarl_unbudgeted_arithmetic.py','scripts/run_ptarl_unbudgeted_frozen.py',
    'scripts/monitor_ptarl_unbudgeted_once.py','tests/test_ptarl_unbudgeted.py')
DANET_ROOT='/home/lux1/iron/local/worktrees/danet-abstract-development/local/danet-abstract-formal-r1'
DANET_MANIFEST_SHA='fa17063826dea7ac9c39b8fd24aabe6ead96d8144eaca91d0cc8c222516e205b'


def require_serial_ready(manifest):
    expected=dict(root=DANET_ROOT,manifest_sha256=DANET_MANIFEST_SHA,service='iron-danet-formal-r1.service')
    if manifest['serial_dependency']!=expected:raise ValueError('Frozen serial dependency differs')
    root=Path(DANET_ROOT)
    if not (root/'supervisor-terminal.json').exists():raise ValueError('DANet terminal event required before PTaRL fitting')
    terminal=json.loads((root/'supervisor-terminal.json').read_text())
    if terminal['manifest_sha256']!=DANET_MANIFEST_SHA:raise ValueError('Foreign DANet terminal event')
    load_anchored(root/'manifest.json',DANET_MANIFEST_SHA)
    status=subprocess.check_output(['systemctl','--user','show',expected['service'],
        '--property=MainPID,ActiveState,SubState'],text=True)
    values=dict(line.split('=',1) for line in status.splitlines())
    if values['MainPID']!='0' or values['ActiveState'] not in ('inactive','failed'):
        raise ValueError('DANet actual process must be terminal before serial PTaRL fitting')
    return terminal


def validate_authority(workspace):
    authority=yaml.safe_load((Path(workspace)/AUTHORITY).read_text())
    expected=dict(version='ptarl-no-time-budget-v1',user_instruction='不要再设置时间预算',maximum_runtime_seconds=None,
        reject_by_projected_seconds=False,timing='descriptive_only',original_probe='preserve_cost_only_refusal_no_refit',
        model_and_quality_gates='unchanged',formal_training='serial_after_current_DANet_terminal',automatic_packages=False,agent_uploads=0)
    if authority!=expected:raise ValueError('Explicit non-time authority or unchanged quality/release rules differ')
    return authority


def non_time_decision(report):
    decision=resource_decision(report['measurement'],report['available_mib'])
    if (any(report[k]!=v for k,v in decision.items()) or report['official_fits']!=0 or report['release_authorized']
            or set(report['checks'])!=set(NON_TIME)|{'development_cost'}
            or any(report['checks'][k] is not True for k in NON_TIME)):
        raise ValueError('Original numerical/learning/memory requirements remain binding')
    limits=dict(pair_unit=1,optimizer=6,kmeans=2);counts=report['counts']
    if (counts['started']!=limits or counts['completed']!=limits
            or any(v for status in ('failed','incomplete') for v in counts[status].values())):
        raise ValueError('All six optimizers and both clustering calls must be complete')
    if report['status']=='failed' and report['checks']['development_cost'] is not False:
        raise ValueError('Only original cost-only refusal may continue')
    return dict(status='passed_non_time_requirements',maximum_runtime_seconds=None,new_fits=0)


def verify_resource(manifest,expected_sha256,*,cold=False):
    old=Path(manifest['resource_workspace']).resolve();original_path=old/'local/ptarl-space-calibration-r1/manifest.json'
    original=load_anchored(original_path,ORIGINAL_MANIFEST_SHA)
    if original['workspace']!=str(old) or original['runtime']!=runtime_snapshot():raise ValueError('Original source workspace/runtime changed')
    for name,digest in original['source_hashes'].items():
        if file_hash(old/name)!=digest or file_hash(Path(manifest['workspace'])/name)!=digest:
            raise ValueError('Original model/source transfer changed: '+name)
    if expected_sha256!=ORIGINAL_ADMISSION_SHA or manifest['resource_admission_sha256']!=expected_sha256:
        raise ValueError('Externally anchored original resource report required')
    root=original_path.parent/'preflight';path=root/'admission.json';report=load_anchored(path,expected_sha256)
    if report['manifest_sha256']!=ORIGINAL_MANIFEST_SHA:raise ValueError('Original resource identity differs')
    non_time_decision(report)
    extra={'admission.json'}
    if report['status']=='failed':
        failed=json.loads((root/'failed.json').read_text())
        if failed!=dict(type='ValueError',message='Frozen PTaRL synthetic resource admission failed'):
            raise ValueError('Failure beyond original cost-only refusal')
        extra.add('failed.json')
    if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!=set(report['artifact_hashes'])|extra:
        raise ValueError('Original closed resource artifacts differ')
    for name,digest in report['artifact_hashes'].items():
        item=root/name
        if item.resolve()!=item or not item.is_relative_to(root) or file_hash(item)!=digest:
            raise ValueError('Original resource artifact changed or escaped')
    if cold:
        code='''from pathlib import Path
import sys,json,numpy as np
from bf_tap_r2.ptarl_freeze import verify_manifest,load_anchored
from bf_tap_r2.ptarl_preflight import synthetic_data,synthetic_settings,TASK,_costs,resource_decision
from bf_tap_r2.ptarl_execution import audit_unit
from bf_tap_r2.ptarl_model import PrototypeRegressor
from bf_tap_r2.v49_gradients import GradientRegressor
from sklearn.cluster import KMeans
manifest=verify_manifest(Path(sys.argv[1]),sys.argv[2]);root=Path(sys.argv[1]).parent/'preflight'
report=load_anchored(root/'admission.json',sys.argv[3]);measured=report['measurement']
training,query,truth=synthetic_data()
def no_fit(*args,**kwargs):raise AssertionError('Independent audit attempted fit')
PrototypeRegressor.train=no_fit
GradientRegressor.train=no_fit
KMeans.fit=no_fit
predictions,audit=audit_unit(root/'unit',TASK,training,query,synthetic_settings(manifest),
    expected_sha256=measured['unit_complete_sha256'],ledger_root=root/'ledger')
if audit['models_checked']!=measured['models_checked'] or audit['maximum_difference']!=measured['cold_difference']:
    raise ValueError('Fresh independent six-model audit differs')
for arm,p in predictions.items():
    if float(np.abs(p-truth).mean())!=measured['mae'][arm]:raise ValueError('Synthetic arithmetic differs')
if float(np.abs(truth-np.median(training.tap_time_len)).mean())!=measured['median_mae']:
    raise ValueError('Median-control arithmetic differs')
if _costs(root,report['policy_sha256'])!=report['counts']:raise ValueError('Fitting reservations differ')
decision=resource_decision(measured,report['available_mib'])
if any(report[k]!=v for k,v in decision.items()):raise ValueError('Original descriptive timing arithmetic differs')
if any(v is not True for k,v in report['checks'].items() if k!='development_cost'):
    raise ValueError('Non-time requirement failed')
print(json.dumps({'status':'passed_non_time_requirements','new_fits':0}))
'''
        completed=subprocess.run([sys.executable,'-c',code,str(original_path),ORIGINAL_MANIFEST_SHA,expected_sha256],
            cwd=old,env=dict(os.environ,PYTHONPATH=str(old/'src')),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        if json.loads(completed.stdout.splitlines()[-1])!=dict(status='passed_non_time_requirements',new_fits=0):
            raise ValueError('Fresh original-worktree non-time audit refused')
    if psutil.virtual_memory().available/1024**2<report['required_available_mib']:
        raise ValueError('Current memory below original admitted requirement')
    return report


def checked_receipt(workspace,path,sha256,sources):
    path=private_path(workspace,path);receipt=load_anchored(path,sha256)
    if (receipt.get('status')!='passed' or receipt.get('exit_code')!=0 or receipt.get('full_suite') is not True
            or receipt.get('source_changed') is not False or receipt.get('source_hashes')!=sources
            or receipt.get('runtime')!=runtime_snapshot() or not set(REQUIRED).issubset(sources)):
        raise ValueError('Passed exact-source full locked suite required')
    log=private_path(workspace,receipt['log'])
    if file_hash(log)!=receipt['log_sha256'] or receipt['summary'] not in log.read_text():raise ValueError('Checked full-suite log changed')
    return receipt


def prepare_manifest(workspace,output,tests_path,tests_sha256,resource_workspace):
    workspace=Path(workspace).resolve();output=private_path(workspace,output);sources=source_snapshot(workspace)
    checked=checked_receipt(workspace,tests_path,tests_sha256,sources)
    if subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=workspace,text=True).strip():
        raise ValueError('Commit checked complete wrapper before freeze')
    original_path=Path(resource_workspace).resolve()/'local/ptarl-space-calibration-r1/manifest.json'
    original=load_anchored(original_path,ORIGINAL_MANIFEST_SHA)
    spec=yaml.safe_load((workspace/SPEC).read_text());settings=validate_spec(spec);authority=validate_authority(workspace)
    manifest=dict(original,experiment='ptarl_unbudgeted_v1',workspace=str(workspace),source_hashes=sources,runtime=runtime_snapshot(),
        execution_python=sys.executable,spec=spec,settings=settings,runtime_authority=authority,resource_workspace=str(Path(resource_workspace).resolve()),
        serial_dependency=dict(root=DANET_ROOT,manifest_sha256=DANET_MANIFEST_SHA,service='iron-danet-formal-r1.service'),
        resource_admission_sha256=ORIGINAL_ADMISSION_SHA,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),
        locked_tests=dict(path=str(Path(tests_path).resolve()),sha256=tests_sha256,receipt=checked),release_authorized=False)
    verify_resource(manifest,ORIGINAL_ADMISSION_SHA,cold=True)
    values=reload_references(manifest)
    best=json.loads((Path(manifest['reference_root'])/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if best['candidate']!='DE3_IRON_USER_REQUESTED' or best['score']!=96.3749:raise ValueError('Latest incumbent differs before freeze')
    output.parent.mkdir(parents=True,exist_ok=False);write_new(output,manifest)
    return dict(path=str(output),sha256=file_hash(output))


def verify_manifest(path,expected_sha256,*,references=True):
    manifest=load_anchored(path,expected_sha256);workspace=Path(manifest['workspace']);private_path(workspace,path)
    if (manifest['experiment']!='ptarl_unbudgeted_v1' or Path(__file__).resolve()!=workspace/'src/bf_tap_r2/ptarl_unbudgeted_freeze.py'
            or source_snapshot(workspace)!=manifest['source_hashes'] or runtime_snapshot()!=manifest['runtime']
            or sys.executable!=manifest['execution_python'] or manifest['release_authorized']):
        raise ValueError('Formal source/runtime/workspace/scope changed')
    spec=yaml.safe_load((workspace/SPEC).read_text())
    if spec!=manifest['spec'] or validate_spec(spec)!=manifest['settings'] or validate_authority(workspace)!=manifest['runtime_authority']:
        raise ValueError('Formal recipe/non-time authority changed')
    tests=checked_receipt(workspace,manifest['locked_tests']['path'],manifest['locked_tests']['sha256'],manifest['source_hashes'])
    if tests!=manifest['locked_tests']['receipt']:raise ValueError('Locked-suite receipt differs')
    verify_resource(manifest,manifest['resource_admission_sha256'])
    if references:reload_references(manifest)
    return manifest
