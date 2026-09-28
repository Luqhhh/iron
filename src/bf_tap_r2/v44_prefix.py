"""Read-only validation of the frozen V43 experiment and matched model prefix."""
import json

import numpy as np
import yaml

from .data import TARGETS
from .v7_periodic import file_hash,digest
from .v43_run import verify_hashes,verified_unit,unit_id
from .v44_bart import verify_prefix


def verify_old_run(root,spec):
    ref=spec['prefix_reference'];old=root/ref['directory']
    for name in ['manifest','audit']:
        if file_hash(old/f'{name}.json')!=ref[f'{name}_sha256']:
            raise ValueError('V43 manifest/audit changed')
    manifest=json.loads((old/'manifest.json').read_text());audit=json.loads((old/'audit.json').read_text())
    if audit['status']!='passed' or file_hash(old/'summary.json')!=audit['summary_sha256']:
        raise ValueError('V43 passing audit not bound to its summary')
    if file_hash(root/'src/bf_tap_r2/v43_audit_r2.py')!=audit['auditor_sha256']:
        raise ValueError('V43 audited checker changed')
    verify_hashes(root,manifest['source_hashes']);verify_hashes(root,manifest['data_hashes'])
    identity=dict(manifest);identity.pop('identity')
    if digest(identity)!=manifest['identity']:
        raise ValueError('V43 manifest identity changed')
    if file_hash(root/ref['spec'])!=ref['spec_sha256']:
        raise ValueError('V43 specification changed')
    original=yaml.safe_load((root/ref['spec']).read_text())
    restored=dict(spec['training'],burn_sweeps=original['training']['burn_sweeps'],thin=original['training']['thin'])
    if restored!=original['training']:
        raise ValueError('Changes beyond chain schedule are prohibited')
    for key in ['reference','calibration','split_seeds','confirmation_seeds','folds','candidates','promotion','runtime_versions']:
        if spec[key]!=original[key]:
            raise ValueError(f'V43 comparison protocol changed: {key}')
    count=0
    for seed in spec['split_seeds']:
        for fold in range(5):
            keys=[f'reference-{role}-s{seed}-f{fold}' for role in ['outer','calibration']]
            keys += [f'{target}-{recipe}-s{seed}-f{fold}' for target in TARGETS for recipe in spec['recipes']]
            for key in keys:
                if verified_unit(old/key,unit_id(manifest,key)) is None:
                    raise ValueError(f'Missing V43 unit: {key}')
                count+=1
    return {'status':'passed','verified_units':count,'old_manifest_sha256':file_hash(old/'manifest.json')}


def verify_unit_prefix(root,spec,key,model,query,calibration):
    old=root/spec['prefix_reference']['directory']/key
    with np.load(old/'predictions.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['query_ids'],query.sample_id.to_numpy(dtype=str)):
            raise ValueError('V43 prefix query IDs differ')
        if not np.array_equal(saved['calibration_ids'],calibration.sample_id.to_numpy(dtype=str)):
            raise ValueError('V43 prefix calibration IDs differ')
        final=verify_prefix(model,json.loads((old/'model.json').read_text()),query,saved['prediction'])
        selector=verify_prefix(model.calibration_model_,json.loads((old/'calibration_model.json').read_text()),
                               calibration.drop(columns=list(TARGETS)),saved['calibration_prediction'])
    return {'status':'passed','refit':final,'calibration':selector,'old_unit':str(old),'new_fits':0}
