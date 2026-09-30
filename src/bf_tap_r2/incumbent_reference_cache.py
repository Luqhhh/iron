"""Externally anchored historical DE3 reuse; no estimator fit or scoring."""
from pathlib import Path
import importlib.metadata
import hashlib
import json
import subprocess

import numpy as np
import yaml

from .component_regularization_audit import verify_saved
from .data import FEATURES,TARGETS
from .incumbent_reference_columns import Member,TRAINING_SEEDS
from .incumbent_reference_core import validate_rows
from .incumbent_reference_identity import verify_source_bytes
from .incumbent_reference_ledger import file_hash
from .v3_4_bags import group_safe_inner_folds
from .v7_periodic import digest

# These historical orchestration/test files changed after the completed DE3
# round for the separate E-COMPOSE recovery. None is executed by this pipeline.
HISTORICAL_ONLY_FILES={'src/bf_tap_r2/independent_checkpoints_run.py',
    'src/bf_tap_r2/independent_checkpoints_preflight.py','tests/test_independent_checkpoints.py'}
HISTORICAL_PRIVATE_CONFIGS={'configs/data.local.yaml','configs/predict.local.yaml'}


def read_json(path):return json.loads(Path(path).read_text())


def safe_child(root,name):
    root=Path(root).resolve();p=root/name
    if Path(name).is_absolute() or not p.resolve().is_relative_to(root):
        raise ValueError('Artifact escapes frozen root')
    return p


def source_child(root,name):
    """Historical local/runs may point to its explicitly shared private root."""
    root=Path(root).resolve();relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Source path escapes frozen root')
    if relative.parts[:2]==('local','runs'):
        p=root/relative;shared=(root/'local/runs').resolve()
        if not p.resolve().is_relative_to(shared):raise ValueError('Source escapes declared shared private cache')
        return p
    # The original manifest also hashed these two shared private baseline
    # configs. They are verified as bytes only and never parsed or executed.
    if relative.as_posix() in HISTORICAL_PRIVATE_CONFIGS:return root/relative
    return safe_child(root,name)


def verify_hashes(root,hashes,*,shared_runs=False):
    for name,sha in hashes.items():
        if file_hash((source_child if shared_runs else safe_child)(root,name))!=sha:raise ValueError('Frozen artifact changed: '+name)


def verify_unit(directory,manifest,key):
    directory=Path(directory);complete=read_json(directory/'complete.json')
    if complete['identity']!=digest(dict(manifest_identity=manifest['identity'],unit=key)) or complete['key']!=key:
        raise ValueError('Historical DE3 unit identity mismatch')
    verify_hashes(directory,complete['hashes'])
    return complete


def inspect_development(source_root,cache_root,anchors):
    """Hash/metadata inspection before any training labels are loaded."""
    source_root=Path(source_root).resolve();cache_root=Path(cache_root).resolve()
    if set(anchors)!={'manifest.json','summary.json','audit.json'}:
        raise ValueError('Three external historical anchors required')
    verify_hashes(cache_root,anchors)
    manifest=read_json(cache_root/'manifest.json');audit=read_json(cache_root/'audit.json');summary=read_json(cache_root/'summary.json')
    identity=dict(manifest);identity.pop('identity')
    if (digest(identity)!=manifest['identity'] or manifest['queue']!='DE3'
            or manifest['seeds']!=[42,3407] or manifest['development'] is not None):
        raise ValueError('Frozen original DE3 development required')
    if (audit['status']!='passed' or audit['queue']!='DE3' or audit['native_replays']!=20
            or audit['new_fits']!=0 or audit['full_batch_cold_difference']!=0
            or audit['manifest_sha256']!=anchors['manifest.json'] or audit['summary_sha256']!=anchors['summary.json']
            or summary['selected_for_confirmation']['tap_iron'] is not None):
        raise ValueError('Original passed no-finalist DE3 evidence required')
    if set(manifest['data_hashes'])!={'复赛_train/train_samples.csv','复赛_train/train_features.csv'}:
        raise ValueError('Only the frozen round-two training snapshot is accepted')
    verify_hashes(source_root,manifest['data_hashes'])
    variants=[];historical=[]
    for name,sha in manifest['source_hashes'].items():
        p=source_child(source_root,name)
        if file_hash(p)==sha:continue
        if name not in HISTORICAL_ONLY_FILES and (not name.startswith(('src/','configs/')) or p.suffix not in ('.py','.yaml','.yml')):
            raise ValueError('Historical source/evidence changed: '+name)
        if name in HISTORICAL_ONLY_FILES:
            blob=subprocess.check_output(['git','show',manifest['starting_commit']+':'+name],cwd=source_root)
            if hashlib.sha256(blob).hexdigest()!=sha:raise ValueError('Original committed historical source differs: '+name)
            historical.append(dict(path=name,reason='historical_committed_source_not_executed',
                original_commit=manifest['starting_commit'],working_sha256=file_hash(p),frozen_sha256=sha))
        else:
            blob=subprocess.check_output(['git','show','HEAD:'+name],cwd=source_root)
            variant=verify_source_bytes(p.read_bytes(),blob,sha)
            variants.append(dict(path=name,reason=variant,working_sha256=file_hash(p),frozen_sha256=sha))
    for name,version in manifest['versions'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Historical runtime differs: '+name)
    original_spec=yaml.safe_load((source_root/'configs/independent_ensemble_checkpoints/SPEC.yaml').read_text())
    if file_hash(source_root/'configs/independent_ensemble_checkpoints/SPEC.yaml')!=manifest['spec_sha256']:
        raise ValueError('Original native training settings changed')
    hashes={name:file_hash(cache_root/name) for name in anchors};costs=[]
    for seed in (42,3407):
        for fold in range(5):
            name=f'tap_iron-DE3-s{seed}-f{fold}';directory=cache_root/name
            verify_unit(directory,manifest,name)
            verify_hashes(directory,read_json(directory/'nested-hashes.json'))
            for ts in TRAINING_SEEDS:verify_unit(directory/f'training-seed-{ts}',manifest,f'{name}/training-seed-{ts}')
            hashes.update({str(p.relative_to(cache_root)):file_hash(p) for p in directory.rglob('*') if p.is_file()})
            meta=read_json(directory/'metadata.json')
            if (meta['training_seeds']!=list(TRAINING_SEEDS) or meta['aggregation']!='fixed_arithmetic_mean'
                    or meta['target']!='tap_iron' or meta['queue']!='DE3' or meta['seed']!=seed or meta['fold']!=fold
                    or (meta['selector_fits'],meta['refit_fits'],meta['reused_native_units'])!=(2,2,1)):
                raise ValueError('Historical DE3 coverage/cost identity differs')
            traces=[]
            for ts in (104729,130363):
                model=read_json(directory/f'training-seed-{ts}/metadata.json')['model']
                traces.extend(model['traces'].values())
            costs.append(dict(seconds=meta['seconds'],optimizer_updates=sum(t['updates'] for t in traces),
                peak_rss_mib=max(t['peak_rss_mib'] for t in traces)))
    private_configs={n:str((source_root/n).resolve()) for n in HISTORICAL_PRIVATE_CONFIGS if n in manifest['source_hashes']}
    return dict(status='passed',hashes=hashes,source_byte_variants=variants,historical_source_bridges=historical,
                historical_private_config_paths=private_configs,cost_witnesses=costs,
                original_settings=original_spec['training']['tap_iron'],
                original_native_mechanisms=yaml.safe_load((source_root/'configs/strong_component_regularization/SPEC.yaml').read_text())['mechanisms'],
                manifest_identity=manifest['identity'],new_fits=0)


def audit_development(cache_root,evidence,frame,folds,native_j42):
    cache_root=Path(cache_root);verify_hashes(cache_root,evidence['hashes'])
    members=[];maximum=0.;cold_models=0
    manifest=read_json(cache_root/'manifest.json')
    for seed in (42,3407):
        if manifest['fold_hashes'][str(seed)]!=digest(folds[seed].tolist()):raise ValueError('Historical DE3 outer folds differ')
        for fold in range(5):
            mask=folds[seed]==fold;training=frame.loc[~mask].reset_index(drop=True)
            query=frame.loc[mask,['sample_id','spout_no',*FEATURES]].reset_index(drop=True);validate_rows(training,query)
            directory=cache_root/f'tap_iron-DE3-s{seed}-f{fold}';meta=read_json(directory/'metadata.json')
            if meta['fit_ids_digest']!=digest(training.sample_id.tolist()):raise ValueError('Historical DE3 fit IDs differ')
            inner=np.asarray(group_safe_inner_folds(training,seed=42)['fold']);y=training[list(TARGETS)].to_numpy(float)
            fitting=training.loc[inner!=0].reset_index(drop=True);calibration=training.loc[inner==0]
            with np.load(directory/'predictions.npz',allow_pickle=False) as a:aggregate={k:a[k].copy() for k in a.files}
            if set(aggregate)!={'query_ids','DE3','seed_42','seed_104729','seed_130363'}:
                raise ValueError('Historical DE3 aggregate schema differs')
            np.testing.assert_array_equal(aggregate['query_ids'],query.sample_id.to_numpy(dtype=str))
            predictions=[]
            for ts in TRAINING_SEEDS:
                sub=directory/f'training-seed-{ts}';settings=dict(evidence['original_settings'],random_seed=ts)
                mechanisms=evidence['original_native_mechanisms'] if ts==42 else {}
                selector=verify_saved(sub/'selection.pt',fitting,y[inner!=0],'BASE',settings,mechanisms,calibration)
                final=verify_saved(sub/'refit.pt',training,y,'BASE',settings,mechanisms,expected_epoch=selector.saved['trace']['selected_epoch'])
                with np.load(sub/'predictions.npz',allow_pickle=False) as a:ids=a['query_ids'].copy();prediction=a['prediction'].copy()
                np.testing.assert_array_equal(ids,query.sample_id.to_numpy(dtype=str))
                if prediction.shape!=(len(query),2) or not np.isfinite(prediction).all():raise ValueError('Invalid historical joint predictions')
                observed=final.predict(query);np.testing.assert_array_equal(prediction,observed)
                variants=[final.predict(query.iloc[::-1])[::-1],np.concatenate([final.predict(query.iloc[i:i+37]) for i in range(0,len(query),37)])]
                delta=max(float(np.abs(p-observed).max()) for p in variants);maximum=max(maximum,delta)
                if delta>5e-4 or delta/max(1.,float(np.abs(observed).max()))>1e-6:raise ValueError('Historical cold chunk/order difference')
                np.testing.assert_array_equal(aggregate[f'seed_{ts}'],prediction[:,0]);predictions.append(prediction[:,0])
                if ts==42:np.testing.assert_array_equal(prediction[:,0],native_j42[seed][mask])
                members.append(Member(seed,fold,ts,tuple(ids.tolist()),prediction[:,0]));cold_models+=2
            np.testing.assert_array_equal(aggregate['DE3'],np.mean(predictions,axis=0))
    return members,dict(status='passed',cold_models=cold_models,native_replays=10,
        full_batch_difference=0.,maximum_order_chunk_absolute_difference=maximum,new_fits=0)
