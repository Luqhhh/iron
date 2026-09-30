"""Exact-source execution freeze and byte-identical transfer of admitted core."""
from pathlib import Path
from dataclasses import asdict
import subprocess

import psutil
import yaml

from .data import FEATURES, TARGETS
from .dnnr_model import Settings
from .dnnr_preflight import (anchored, private_path, source_hashes, runtime, verify_admission)
from .dnnr_ledger import file_hash, write_new
from .dnnr_reference_bridge import load_reference, INCUMBENT, SCORE
from .dnnr_phase_protocol import phase_limits, CONTROLS

EXECUTION = 'configs/dnnr_taylor/EXECUTION.yaml'
RESOURCE_MANIFEST_SHA = 'a9f30f90e2eccb5aeef85d025bc10b64dba3bbbea9c1526c27aa957aa0bf12dd'
RESOURCE_ADMISSION_SHA = 'af3868187156a8f7adcdaf953882f275b787434ef63e07fdfd01baa1ea7f716e'
REQUIRED = ('src/bf_tap_r2/dnnr_phase_run.py','src/bf_tap_r2/dnnr_phase_controller.py',
    'src/bf_tap_r2/dnnr_phase_arithmetic.py','scripts/run_dnnr_frozen.py',
    'scripts/monitor_dnnr_once.py',EXECUTION)


def validate_spec(workspace):
    workspace=Path(workspace)
    model=yaml.safe_load((workspace/'configs/dnnr_taylor/SPEC.yaml').read_text())
    spec=yaml.safe_load((workspace/EXECUTION).read_text())
    if (model['training']!=asdict(Settings()) or model['platform_reference']['candidate']!=INCUMBENT
            or model['platform_reference']['score']!=SCORE or spec['version']!='dnnr-complete-coverage-execution-v1'
            or spec['model_spec']!='configs/dnnr_taylor/SPEC.yaml' or spec['split_seeds']!=[42,3407]
            or spec['confirmation_seeds']!=[7777,12011] or spec['folds']!=5
            or spec['reference_by_target']!={t:'CURRENT_DE3' for t in TARGETS}
            or spec['candidates']!={t:['DNNR_FIXED','DNNR_LEARNED'] for t in TARGETS}
            or spec['tie_preference_by_target']!={t:['DNNR_FIXED','DNNR_LEARNED'] for t in TARGETS}
            or spec['matched_controls']!=CONTROLS or spec['blend_weight']!=.2
            or spec['other_target']!='unchanged_current' or spec['workers']!=4 or spec['encoded_dimension_maximum']!=26
            or spec['development_limits']!=phase_limits('development')
            or spec['confirmation']!='only_earned_candidates_plus_necessary_controls_no_failed_candidate_fit'
            or spec['candidate_tier_policy']!='configs/candidate_tiers.yaml'
            or spec['candidate_tiers_are_descriptive_not_mechanism_or_release_admission'] is not True
            or spec['audit_before_scoring_or_confirmation'] is not True
            or spec['new_full_data_fits']!=0 or spec['automatic_packages'] or spec['desktop_writes']!=0 or spec['agent_uploads']!=0
            or spec['monitoring']!=dict(interval_seconds=600,between_check_polling=False)):
        raise ValueError('Frozen candidate pool, costs, controls or release protocol changed')
    stage=model['future_formal_stage']
    if (stage['development_gate']!=dict(both_complete_seeds_positive=True,mean_gain_minimum=.01,
          positive_mean_candidate_vs_matched_control=True,mean_package_score_minimum=96.25)
          or stage['confirmation_gate']!=dict(all_four_seeds_positive=True,seed_level_paired_lcb95_positive=True,
              t_multiplier=2.3533634348018264,sample_sd_ddof=1,positive_four_seed_candidate_vs_matched_control=True)
          or stage['fold_level']!='descriptive_only' or stage['cross_seed_oof_averaging']!='forbidden'):
        raise ValueError('Frozen development/four-seed gates changed')
    return model,spec


def verify_transfer(workspace, resource_workspace, *, cold=False):
    workspace,old=Path(workspace).resolve(),Path(resource_workspace).resolve()
    manifest=anchored(old/'local/dnnr-taylor-resource-r1/manifest.json',RESOURCE_MANIFEST_SHA)
    if manifest['workspace']!=str(old) or manifest['runtime']!=runtime():
        raise ValueError('Preserved G0 workspace/runtime differs')
    for name,sha in manifest['source_hashes'].items():
        if file_hash(old/name)!=sha or file_hash(workspace/name)!=sha:
            raise ValueError('Admitted G0 source/core transfer changed: '+name)
    path=old/'local/dnnr-taylor-resource-r1/preflight/admission.json'
    report=anchored(path,RESOURCE_ADMISSION_SHA)
    if report['status']!='passed' or report['manifest_sha256']!=RESOURCE_MANIFEST_SHA or report['official_fits']!=0:
        raise ValueError('Passed original G0 resource envelope required')
    root=path.parent
    for name,sha in report['artifact_hashes'].items():
        if file_hash(root/name)!=sha:raise ValueError('Original G0 resource artifact changed')
    if cold:
        verify_admission(path,RESOURCE_ADMISSION_SHA,old/'local/dnnr-taylor-resource-r1/manifest.json',RESOURCE_MANIFEST_SHA)
    if psutil.virtual_memory().available/1024**2<report['required_available_mib']:
        raise ValueError('Current RAM below original admitted resource requirement')
    return report


def prepare_manifest(workspace, output, tests_path, tests_sha256, resource_workspace, reference_directory, reference_sha256):
    workspace=Path(workspace).resolve();output=private_path(workspace,output)
    for name in REQUIRED:
        if not (workspace/name).is_file():raise ValueError('Complete controller required before freeze: '+name)
    sources=source_hashes(workspace);checked=anchored(tests_path,tests_sha256)
    if (checked['status']!='passed' or checked['exit_code']!=0 or checked['full_suite'] is not True
            or checked['source_hashes']!=sources or checked['runtime']!=runtime()):
        raise ValueError('Passed exact-source full locked Python3.12 receipt required')
    if subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=workspace,text=True).strip():
        raise ValueError('Commit checked complete implementation before freeze')
    model,spec=validate_spec(workspace)
    resource_report=verify_transfer(workspace,resource_workspace,cold=True)
    frame,folds,_,_,meta=load_reference(reference_directory,reference_sha256)
    if len(frame)!=2754:raise ValueError('Official reference row count differs')
    for seed,fv in folds.items():
        for fold in range(5):
            training=frame.loc[fv!=fold]
            if len(training)>2204 or (fv==fold).sum()>551 or len(FEATURES)+training.spout_no.nunique()+1>26:
                raise ValueError('Official partition exceeds admitted row/dimension cost envelope')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(format='dnnr-formal-execution-v1',workspace=str(workspace),source_hashes=sources,runtime=runtime(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),
        model_spec=model,execution_spec=spec,settings=model['training'],
        checked_tests=dict(path=str(Path(tests_path).resolve()),sha256=tests_sha256,receipt=checked),
        resource_workspace=str(Path(resource_workspace).resolve()),resource_manifest_sha256=RESOURCE_MANIFEST_SHA,
        resource_admission_sha256=RESOURCE_ADMISSION_SHA,resource_required_available_mib=resource_report['required_available_mib'],
        reference_directory=str(Path(reference_directory).resolve()),reference_sha256=reference_sha256,
        reference_frame_digest=meta['frame_digest'],release_authorized=False)
    write_new(output/'manifest.json',manifest)
    return dict(manifest=str(output/'manifest.json'),manifest_sha256=file_hash(output/'manifest.json'))


def verify_manifest(path,sha256, *, reference=True):
    manifest=anchored(path,sha256);workspace=Path(manifest['workspace'])
    private_path(workspace,path)
    if (manifest['format']!='dnnr-formal-execution-v1'
            or Path(__file__).resolve()!=workspace/'src/bf_tap_r2/dnnr_phase_freeze.py'
            or manifest['source_hashes']!=source_hashes(workspace) or manifest['runtime']!=runtime()
            or manifest['release_authorized']):raise ValueError('Formal source/runtime/workspace/scope changed')
    model,spec=validate_spec(workspace)
    if model!=manifest['model_spec'] or spec!=manifest['execution_spec'] or model['training']!=manifest['settings']:
        raise ValueError('Formal frozen spec changed')
    test=anchored(manifest['checked_tests']['path'],manifest['checked_tests']['sha256'])
    if test!=manifest['checked_tests']['receipt']:raise ValueError('Locked suite receipt changed')
    verify_transfer(workspace,manifest['resource_workspace'])
    if reference:
        values=load_reference(manifest['reference_directory'],manifest['reference_sha256'])
        if values[-1]['frame_digest']!=manifest['reference_frame_digest']:raise ValueError('Current reference changed')
    return manifest


def reload_references(manifest):
    return load_reference(manifest['reference_directory'],manifest['reference_sha256'])
