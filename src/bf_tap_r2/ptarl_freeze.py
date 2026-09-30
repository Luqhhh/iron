"""Immutable PTaRL source/runtime/native-reference execution identity.

Freezing never admits resource costs or schedules fits. Complete implementation
and exact-source locked suite are mandatory before this entry can be used.
"""
from pathlib import Path
import subprocess

import yaml

from .rfm_freeze import (source_snapshot, runtime_snapshot, private_path, load_anchored,
                         reference_identity, require_threads)
from .ptarl_reference import load_current_reference, INCUMBENT, SCORE
from .ptarl_protocol import file_hash, write_new, phase_limits

SPEC='configs/ptarl_space_calibration/SPEC.yaml'
REQUIRED_FILES=(
    'src/bf_tap_r2/ptarl_run.py','src/bf_tap_r2/ptarl_preflight.py',
    'src/bf_tap_r2/ptarl_controller.py','src/bf_tap_r2/ptarl_arithmetic.py',
    'scripts/run_ptarl_frozen.py','scripts/monitor_ptarl_once.py',SPEC)


def validate_spec(spec):
    from .ptarl_preflight import SYNTHETIC_SPEC,RESOURCE_SPEC
    stage=spec['future_formal_stage']
    dev=stage['development_gate'];confirm=stage['confirmation_gate']
    if (spec['arms']!=['CONTROL','PTARL_AUX'] or stage['development_seeds']!=[42,3407]
            or stage['confirmation_seeds']!=[7777,12011]
            or stage['fixed_candidate_blend']!='0.8_current_plus_0.2_member_one_target_only'
            or stage['other_target']!='unchanged_current'
            or dev!=dict(both_complete_seeds_positive=True,mean_gain_minimum=.01,
                positive_mean_PTARL_vs_CONTROL=True,mean_package_score_minimum=96.25)
            or confirm!=dict(all_four_seeds_positive=True,seed_level_paired_lcb95_positive=True,
                t_multiplier=2.3533634348018264,sample_sd_ddof=1,positive_four_seed_PTARL_vs_CONTROL=True)
            or stage['fold_level']!='descriptive_only' or stage['cross_seed_oof_averaging']!='forbidden'
            or spec['release']!=dict(automatic_packages=False,desktop_writes=0,agent_uploads=0)
            or spec['monitoring']!=dict(interval_seconds=600,between_check_polling=False)):
        raise ValueError('Frozen PTaRL gates/release/monitoring differ from implementation')
    if (spec['platform_reference']['candidate']!=INCUMBENT or spec['platform_reference']['score']!=SCORE
            or stage['reference']!='verified_four_seed_native_plus_current_DE3_iron_overlay'
            or stage.get('legacy_reference_fallback') is not False):
        raise ValueError('Latest current incumbent reference must be bound without fallback')
    limits=phase_limits('development')
    if (stage['proposed_development_pair_units']!=limits['pair_unit']
            or stage['proposed_development_optimizer_runs']!=limits['optimizer']
            or stage['proposed_development_kmeans_calls']!=limits['kmeans']):
        raise ValueError('Paired optimizer/KMeans budget mismatch')
    if spec.get('calibration')!=dict(split_seed=42,n_splits=5,held_fold=0,native_calibration_cache='forbidden'):
        raise ValueError('Inner partition/cache contract mismatch')
    if spec.get('synthetic_resource_probe')!=SYNTHETIC_SPEC or spec.get('resources')!=RESOURCE_SPEC:
        raise ValueError('Resource probe/admission contract mismatch')
    if any(spec['training'][k]!=v for k,v in dict(width=256,tabm_k=16,blocks=2,embedding_dim=16,
            n_frequencies=16,prototype_count=5,max_epochs=240,batch_size=256).items()):
        raise ValueError('Resource architecture/epoch/batch contract mismatch')
    return spec['training']


def verify_manifest(path,expected_sha256,*,references=True):
    m=load_anchored(path,expected_sha256)
    workspace,root=Path(m['workspace']),Path(m['reference_root'])
    if m.get('experiment')!='ptarl_code_auxiliary_v1' or Path(__file__).resolve()!=workspace/'src/bf_tap_r2/ptarl_freeze.py':
        raise ValueError('PTaRL experiment/imported workspace mismatch')
    if source_snapshot(workspace)!=m['source_hashes'] or runtime_snapshot()!=m['runtime']:
        raise ValueError('Frozen PTaRL source/configuration/runtime changed')
    spec=yaml.safe_load((workspace/SPEC).read_text())
    if spec!=m['spec'] or validate_spec(spec)!=m['settings']:
        raise ValueError('Frozen PTaRL execution spec mismatch')
    tests=load_anchored(m['locked_tests']['path'],m['locked_tests']['sha256'])
    if tests!=m['locked_tests']['receipt']:
        raise ValueError('Locked PTaRL suite evidence changed')
    if references:
        for name,sha in m['reference']['audit']['hashes'].items():
            path=root/name
            if not path.resolve().is_relative_to(root) or file_hash(path)!=sha:
                raise ValueError('Native reference changed: '+name)
        overlay=Path(m['incumbent_overlay_root'])
        for name,sha in m['reference']['audit']['incumbent_overlay_hashes'].items():
            path=overlay/name
            if not path.resolve().is_relative_to(overlay) or file_hash(path)!=sha:
                raise ValueError('Current incumbent reference changed: '+name)
    return m


def reload_references(manifest):
    values=load_current_reference(manifest['reference_root'],manifest['incumbent_overlay_root'],manifest['incumbent_overlay_sha256'])
    if reference_identity(*values)!=manifest['reference']:
        raise ValueError('Native reference arithmetic/identity changed')
    return values


def prepare_manifest(workspace,reference_root,output,*,tests_path,tests_sha256,overlay_root=None,overlay_sha256=None):
    workspace,reference_root=Path(workspace).resolve(),Path(reference_root).resolve()
    output=private_path(workspace,output)
    if output.exists():raise FileExistsError(output)
    for name in REQUIRED_FILES:
        if not (workspace/name).is_file():raise ValueError('Complete implementation required before freeze: '+name)
    spec=yaml.safe_load((workspace/SPEC).read_text());settings=validate_spec(spec)
    sources=source_snapshot(workspace);runtime=runtime_snapshot()
    tests=load_anchored(tests_path,tests_sha256)
    if (tests.get('status')!='passed' or tests.get('exit_code')!=0 or tests.get('full_suite') is not True
            or tests.get('source_hashes')!=sources or tests.get('runtime')!=runtime):
        raise ValueError('Passed full locked suite for these exact sources/runtime required')
    changed=subprocess.check_output(['git','status','--porcelain','--',*sources],cwd=workspace,text=True)
    if changed.strip():raise ValueError('Commit complete tested PTaRL implementation before freeze')
    if overlay_root is None or overlay_sha256 is None:
        raise ValueError('Externally anchored four-seed DE3 incumbent reference required before freeze')
    references=load_current_reference(reference_root,overlay_root,overlay_sha256)
    manifest=dict(version=1,experiment='ptarl_code_auxiliary_v1',workspace=str(workspace),
        reference_root=str(reference_root),spec=spec,settings=settings,
        incumbent_overlay_root=str(Path(overlay_root).resolve()),incumbent_overlay_sha256=overlay_sha256,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip(),
        source_hashes=sources,runtime=runtime,reference=reference_identity(*references),
        locked_tests=dict(path=str(Path(tests_path).resolve()),sha256=tests_sha256,receipt=tests),release_authorized=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_new(output,manifest)
    return dict(path=str(output),sha256=file_hash(output))
