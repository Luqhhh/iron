"""Freeze a separate controller only after original DANet resource admission."""
from pathlib import Path
import json
import os
import subprocess
import sys

import psutil
import yaml

from .data import FEATURES,TARGETS
from .danet_preflight import anchored,private_path,source_hashes,runtime,checked_receipt,validate_spec as model_spec
from .danet_ledger import file_hash,write_new
from .dnnr_reference_bridge import load_reference,INCUMBENT,SCORE
from .danet_phase_protocol import phase_limits,CONTROLS

EXECUTION='configs/danet_abstract/EXECUTION.yaml'
RESOURCE_MANIFEST_SHA='60a1c14b8c0d663110bb5c122912af67ed3869003fcb08ef31aae07e3d6940a2'
REQUIRED=(EXECUTION,'src/bf_tap_r2/danet_phase_run.py','src/bf_tap_r2/danet_phase_controller.py',
    'src/bf_tap_r2/danet_phase_arithmetic.py','src/bf_tap_r2/danet_phase_protocol.py','src/bf_tap_r2/danet_scoring.py',
    'scripts/run_danet_frozen.py','scripts/monitor_danet_once.py')


def validate_spec(workspace):
    workspace=Path(workspace);model=model_spec(workspace)
    spec=yaml.safe_load((workspace/EXECUTION).read_text())
    if (model['platform_reference']['candidate']!=INCUMBENT or model['platform_reference']['score']!=SCORE
            or spec['version']!='danet-complete-coverage-execution-v1' or spec['model_spec']!='configs/danet_abstract/SPEC.yaml'
            or spec['split_seeds']!=[42,3407] or spec['confirmation_seeds']!=[7777,12011] or spec['folds']!=5
            or spec['reference_by_target']!={t:'CURRENT_DE3' for t in TARGETS}
            or spec['candidates']!={t:['DANET_LEARNED'] for t in TARGETS}
            or spec['tie_preference_by_target']!={t:['DANET_LEARNED'] for t in TARGETS}
            or spec['matched_controls']!=CONTROLS or spec['blend_weight']!=.2 or spec['other_target']!='unchanged_current'
            or spec['workers']!=4 or spec['encoded_dimension_maximum']!=26 or spec['development_limits']!=phase_limits('development')
            or spec['confirmation']!='only_earned_targets_plus_matched_fixed_control_no_failed_target_fit'
            or spec['candidate_tier_policy']!='configs/candidate_tiers.yaml'
            or spec['candidate_tiers_are_descriptive_not_mechanism_or_release_admission'] is not True
            or spec['audit_before_scoring_or_confirmation'] is not True or spec['new_full_data_fits']!=0
            or spec['automatic_packages'] or spec['desktop_writes']!=0 or spec['agent_uploads']!=0
            or spec['monitoring']!=dict(interval_seconds=600,between_check_polling=False)):
        raise ValueError('Frozen pool, controls, costs or release protocol changed')
    return model,spec


def verify_transfer(workspace,resource_workspace,admission_sha256,*,cold=False):
    workspace,old=Path(workspace).resolve(),Path(resource_workspace).resolve()
    manifest_path=private_path(old,old/'local/danet-full-resource-r1/manifest.json')
    manifest=anchored(manifest_path,RESOURCE_MANIFEST_SHA)
    if manifest['workspace']!=str(old) or manifest['runtime']!=runtime():raise ValueError('Original resource workspace/runtime differs')
    for name,sha in manifest['source_hashes'].items():
        if file_hash(old/name)!=sha or file_hash(workspace/name)!=sha:raise ValueError('Admitted source/core transfer changed: '+name)
    path=manifest_path.parent/'preflight/admission.json';report=anchored(path,admission_sha256);root=path.parent
    if (report['status']!='passed' or report['manifest_sha256']!=RESOURCE_MANIFEST_SHA or (root/'failed.json').exists()
            or report['official_label_reads']!=0 or report['official_estimator_fits']!=0
            or report['automatic_formal_execution'] or report['release_authorized']):
        raise ValueError('Passed original synthetic resource envelope required')
    if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!=set(report['artifact_hashes'])|{'admission.json'}:
        raise ValueError('Original resource closed artifact schema differs')
    for name,sha in report['artifact_hashes'].items():
        item=root/name
        if item.resolve()!=item or not item.is_relative_to(root) or file_hash(item)!=sha:raise ValueError('Original resource artifact changed or escaped')
    if cold:
        # Run the exact preserved verifier in its original source worktree;
        # its module-location guard must never be bypassed in the new wrapper.
        code='''import sys,json
from pathlib import Path
from bf_tap_r2.danet_preflight import verify_admission
report=verify_admission(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),sys.argv[4])
print(json.dumps({"status":report["status"],"new_fits":0}))
'''
        result=subprocess.run([sys.executable,'-c',code,str(path),admission_sha256,str(manifest_path),RESOURCE_MANIFEST_SHA],
            cwd=old,env=dict(os.environ,PYTHONPATH=str(old/'src')),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        if json.loads(result.stdout.splitlines()[-1])!=dict(status='passed',new_fits=0):raise ValueError('Original independent admission verifier refused')
    if psutil.virtual_memory().available/1024**2<report['required_available_mib']:raise ValueError('Current RAM below admitted requirement')
    return report


def prepare_manifest(workspace,output,tests_path,tests_sha256,resource_workspace,resource_admission_sha256,
        reference_directory,reference_sha256):
    workspace=Path(workspace).resolve();output=private_path(workspace,output);sources=source_hashes(workspace)
    checked=checked_receipt(workspace,tests_path,tests_sha256,sources,runtime())
    if not set(REQUIRED).issubset(sources):raise ValueError('Complete formal controller required')
    if subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=workspace,text=True).strip():
        raise ValueError('Commit checked controller before freeze')
    model,spec=validate_spec(workspace)
    resource_report=verify_transfer(workspace,resource_workspace,resource_admission_sha256,cold=True)
    frame,folds,_,_,meta=load_reference(reference_directory,reference_sha256)
    best=json.loads((Path(meta['reader_request']['native_root'])/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if best['candidate']!=INCUMBENT or best['score']!=SCORE:raise ValueError('Latest registered incumbent differs before freeze')
    if len(frame)!=2754:raise ValueError('Official current-reference row count differs')
    for fold_values in folds.values():
        for fold in range(5):
            training=frame.loc[fold_values!=fold]
            if len(training)>2204 or (fold_values==fold).sum()>551 or len(FEATURES)+training.spout_no.nunique()+1>26:
                raise ValueError('Formal partition exceeds admitted resource envelope')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(format='danet-formal-execution-v1',workspace=str(workspace),source_hashes=sources,runtime=runtime(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),model_spec=model,
        execution_spec=spec,settings=model['training'],checked_tests=dict(path=str(Path(tests_path).resolve()),sha256=tests_sha256,receipt=checked),
        resource_workspace=str(Path(resource_workspace).resolve()),resource_manifest_sha256=RESOURCE_MANIFEST_SHA,
        resource_admission_sha256=resource_admission_sha256,resource_required_available_mib=resource_report['required_available_mib'],
        reference_directory=str(Path(reference_directory).resolve()),reference_sha256=reference_sha256,
        reference_frame_digest=meta['frame_digest'],release_authorized=False)
    write_new(output/'manifest.json',manifest)
    return dict(manifest=str(output/'manifest.json'),manifest_sha256=file_hash(output/'manifest.json'))


def verify_manifest(path,sha256,*,reference=True):
    manifest=anchored(path,sha256);workspace=Path(manifest['workspace']);private_path(workspace,path)
    if (manifest['format']!='danet-formal-execution-v1' or Path(__file__).resolve()!=workspace/'src/bf_tap_r2/danet_phase_freeze.py'
            or manifest['source_hashes']!=source_hashes(workspace) or manifest['runtime']!=runtime() or manifest['release_authorized']
            or manifest['resource_manifest_sha256']!=RESOURCE_MANIFEST_SHA):raise ValueError('Formal source/runtime/workspace/scope changed')
    model,spec=validate_spec(workspace)
    if model!=manifest['model_spec'] or spec!=manifest['execution_spec'] or model['training']!=manifest['settings']:
        raise ValueError('Formal frozen spec changed')
    checked=checked_receipt(workspace,manifest['checked_tests']['path'],manifest['checked_tests']['sha256'],manifest['source_hashes'],manifest['runtime'])
    if checked!=manifest['checked_tests']['receipt']:raise ValueError('Locked suite receipt changed')
    report=verify_transfer(workspace,manifest['resource_workspace'],manifest['resource_admission_sha256'])
    if report['required_available_mib']!=manifest['resource_required_available_mib']:raise ValueError('Frozen RAM admission changed')
    if reference:
        values=load_reference(manifest['reference_directory'],manifest['reference_sha256'])
        if values[-1]['frame_digest']!=manifest['reference_frame_digest']:raise ValueError('Current DE3 reference changed')
    return manifest


def reload_references(manifest):
    return load_reference(manifest['reference_directory'],manifest['reference_sha256'])
