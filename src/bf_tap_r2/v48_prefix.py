"""Read-only validation of the frozen V47 experiment and matched model prefix."""
import json

import numpy as np
import yaml

from .data import TARGETS
from .v7_periodic import file_hash,digest
from .v47_run import verify_hashes,verified_unit,unit_id
from .v47_factorized import FactorRegressor
from .v47_verify import independent_prediction


def verify_old_run(root,spec):
    ref=spec['prefix_reference'];old=root/ref['directory']
    for name in ['manifest','audit']:
        if file_hash(old/f'{name}.json')!=ref[f'{name}_sha256']:
            raise ValueError('V47 manifest/audit changed')
    manifest=json.loads((old/'manifest.json').read_text());audit=json.loads((old/'audit.json').read_text())
    if audit['status']!='passed' or file_hash(old/'summary.json')!=audit['summary_sha256']:
        raise ValueError('V47 passing audit not bound to its summary')
    if file_hash(root/'src/bf_tap_r2/v47_audit.py')!=audit['auditor_sha256']:
        raise ValueError('V47 audited checker changed')
    verify_hashes(root,manifest['source_hashes']);verify_hashes(root,manifest['data_hashes'])
    identity=dict(manifest);identity.pop('identity')
    if digest(identity)!=manifest['identity']:
        raise ValueError('V47 manifest identity changed')
    if file_hash(root/ref['spec'])!=ref['spec_sha256']:
        raise ValueError('V47 specification changed')
    original=yaml.safe_load((root/ref['spec']).read_text())
    restored=dict(spec['training'],max_epochs=original['training']['max_epochs'])
    if restored!=original['training']:
        raise ValueError('Changes beyond epoch cap are prohibited')
    for key in ['reference','calibration','split_seeds','confirmation_seeds','folds','candidates','promotion','runtime_versions']:
        if spec[key]!=original[key]:
            raise ValueError(f'V47 comparison protocol changed: {key}')
    synthetic = root/ref['synthetic_directory']
    if file_hash(synthetic/'report.json') != ref['synthetic_report_sha256']:
        raise ValueError('V47 synthetic report changed')
    synthetic_report = json.loads((synthetic/'report.json').read_text())
    if synthetic_report['status'] != 'passed': raise ValueError('V47 synthetic admission failed')
    verify_hashes(root,synthetic_report['source_hashes'])
    count=0
    for seed in spec['split_seeds']:
        for fold in range(5):
            keys=[f'reference-{role}-s{seed}-f{fold}' for role in ['outer','calibration']]
            keys += [f'{target}-{recipe}-s{seed}-f{fold}' for target in TARGETS for recipe in spec['recipes']]
            for key in keys:
                if verified_unit(old/key,unit_id(manifest,key)) is None:
                    raise ValueError(f'Missing V47 unit: {key}')
                count+=1
    return {'status':'passed','verified_units':count,'old_manifest_sha256':file_hash(old/'manifest.json')}


def verify_prefix_files(prefix_path, old_path, query, expected):
    prefix=json.loads(prefix_path.read_text()); old=json.loads(old_path.read_text())
    clean=lambda x:{k:v for k,v in x.items() if k!='peak_rss_mib'}
    if (prefix['recipe']!=old['recipe'] or prefix['settings']!=old['settings']
            or prefix['state']!=old['state'] or clean(prefix['metadata'])!=clean(old['metadata'])):
        raise ValueError('Exact saved V47 prefix mismatch')
    pred=FactorRegressor.load(prefix_path).predict(query)
    if not np.array_equal(pred,expected): raise ValueError('V47 prefix prediction not bit-identical')
    independent=independent_prediction(prefix_path,query)
    diff=float(np.max(np.abs(independent-expected)))
    if diff>1e-8: raise ValueError('Independent prefix prediction mismatch')
    return {'status':'passed','state_and_trace_exact':True,'prediction_difference':0.,
            'independent_prediction_difference':diff,'old_model_sha256':file_hash(old_path),
            'new_prefix_sha256':file_hash(prefix_path),'new_fits':0}


def verify_unit_prefix(root,spec,key,directory,query,calibration):
    old=root/spec['prefix_reference']['directory']/key
    with np.load(old/'predictions.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['query_ids'],query.sample_id.to_numpy(dtype=str)):
            raise ValueError('V47 prefix query IDs differ')
        if not np.array_equal(saved['calibration_ids'],calibration.sample_id.to_numpy(dtype=str)):
            raise ValueError('V47 prefix calibration IDs differ')
        final=verify_prefix_files(directory/'prefix_model.json',old/'model.json',query,saved['prediction'])
        selector=verify_prefix_files(directory/'prefix_calibration_model.json',old/'calibration_model.json',
                                    calibration.drop(columns=list(TARGETS)),saved['calibration_prediction'])
    return {'status':'passed','refit':final,'calibration':selector,'old_unit':str(old),'new_fits':0}
