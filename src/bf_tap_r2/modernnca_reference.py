"""Reuse the anchored four-seed DE3/V32 reference export with no fitting.

Validation is adapted from the checked dnnr_reference_bridge reader. The
existing cache retains its original format and receipt, without relabeling.
"""
from pathlib import Path
import hashlib
import json
import importlib.metadata
import os
import platform
import sys
import numpy as np
import pandas as pd
from .data import FEATURES, TARGETS
from .modernnca_ledger import file_hash, canonical
from .v7_periodic import digest

SEEDS=(42,3407,7777,12011)
INCUMBENT='DE3_IRON_USER_REQUESTED'
SCORE=96.3749
OVERLAY_SHA='492c21150764c08e9714ab5321badeb2827459e23f9480126155f830021c2ab4'

def anchored(path, sha):
    if not isinstance(sha,str) or len(sha)!=64 or file_hash(path)!=sha:
        raise ValueError('External reference identity mismatch')
    return json.loads(Path(path).read_text())

def array_identity(value):
    value=np.ascontiguousarray(value,dtype='<f8')
    return hashlib.sha256(canonical(list(value.shape))+value.tobytes()).hexdigest()

def verify_external_identity(metadata):
    request = metadata['reader_request']; reader = Path(request['reader_workspace'])
    parent = anchored(request['reader_manifest'],request['reader_manifest_sha256'])
    if (parent['source_hashes'] != metadata['reader_source_hashes'] or parent['runtime'] != metadata['reader_runtime']
            or parent['experiment']!='ptarl_code_auxiliary_v1' or parent['workspace']!=str(reader)
            or parent['reference_root']!=request['native_root'] or parent['incumbent_overlay_root']!=request['overlay_root']
            or parent['incumbent_overlay_sha256']!=OVERLAY_SHA or request['overlay_sha256']!=OVERLAY_SHA
            or parent['reference']!=metadata['reference_identity'] or parent['reference']['audit']!=metadata['native_audit']):
        raise ValueError('Preserved reference reader identity changed')
    for name,sha in metadata['reader_source_hashes'].items():
        if file_hash(reader/name) != sha:
            raise ValueError('Reference reader source changed')
    audit = metadata['native_audit']; root = Path(request['native_root']); overlay = Path(request['overlay_root'])
    for name,sha in audit['hashes'].items():
        path = root/name
        if not path.resolve().is_relative_to(root) or file_hash(path) != sha:
            raise ValueError('Original native reference changed')
    for name,sha in audit['incumbent_overlay_hashes'].items():
        path = overlay/name
        if not path.resolve().is_relative_to(overlay) or file_hash(path) != sha:
            raise ValueError('Current incumbent overlay changed')


def load_reference(directory, complete_sha256, *, verify_external=True):
    directory = Path(directory).resolve()
    if (directory/'failed.json').exists():
        raise ValueError('Failed reference export cannot be consumed')
    meta = anchored(directory/'complete.json',complete_sha256)
    if (meta['format'] != 'dnnr-current-reference-v1' or meta['candidate'] != INCUMBENT
            or meta['score_user_reported'] != SCORE or meta['parent'] != 'V32_TIME_A60V7_50'
            or meta['seeds'] != list(SEEDS) or meta['new_reference_fits'] != 0
            or meta['new_candidate_fits'] != 0 or meta['release_authorized']
            or meta['native_audit']['status'] != 'passed' or meta['native_audit']['new_reference_fits'] != 0
            or meta['native_audit']['incumbent_overlay_sha256'] != OVERLAY_SHA):
        raise ValueError('Audited current incumbent without fallback required')
    if verify_external:
        verify_external_identity(meta)
        runtime=meta['reader_runtime']
        actual=dict(python=platform.python_version(),executable=str(Path(sys.executable).resolve()),
            packages={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()},
            threads={k:os.environ.get(k) for k in runtime['threads']})
        if actual!=runtime:raise ValueError('Reference reader runtime changed')
        best=json.loads((Path(meta['reader_request']['native_root'])/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
        if best['candidate']!=INCUMBENT or best['score']!=SCORE:raise ValueError('Current platform incumbent changed')
    path = directory/'reference.npz'
    if path.is_symlink() or file_hash(path) != meta['array_sha256']:
        raise ValueError('Exported reference arrays changed')
    with np.load(path,allow_pickle=False) as archive:
        arrays = {k:archive[k].copy() for k in archive.files}
    expected = {'ids','numeric','spout','targets'} | {f'{k}-{s}' for k in ('fold','current','parent') for s in SEEDS}
    n = meta['rows']
    if (type(n) is not int or n<=0 or set(arrays) != expected
            or arrays['numeric'].shape != (n,len(FEATURES)) or arrays['targets'].shape != (n,2)
            or arrays['numeric'].dtype!=np.float64 or arrays['targets'].dtype!=np.float64
            or set(meta['frame_columns'])!=set(FEATURES)|set(TARGETS)|{'sample_id','spout_no'}
            or len(meta['frame_columns'])!=len(FEATURES)+4):
        raise ValueError('Incomplete reference array schema')
    ids, spout = arrays['ids'], arrays['spout']
    if (ids.shape != (n,) or ids.dtype.kind != 'U' or len(set(ids)) != n or any(not v for v in ids)
            or spout.shape != (n,) or spout.dtype != np.int64):
        raise ValueError('Invalid reference row identities/spout')
    frame = pd.DataFrame(arrays['numeric'],columns=FEATURES)
    frame['sample_id'] = ids; frame['spout_no'] = spout
    for j,t in enumerate(TARGETS):frame[t] = arrays['targets'][:,j]
    frame = frame.loc[:,meta['frame_columns']].astype(meta['frame_dtypes'])
    reconstructed = hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    if reconstructed != meta['frame_digest'] or reconstructed != meta['native_audit']['data_digest']:
        raise ValueError('Exported training row/data identity changed')
    reference=meta['reference_identity']
    if reference['rows']!=n or reference['row_ids_sha256']!=hashlib.sha256(canonical(ids.tolist())).hexdigest():
        raise ValueError('Original reader row identity differs')
    folds, current, parent = {}, {}, {}
    for seed in SEEDS:
        fold = arrays[f'fold-{seed}']
        if (fold.shape != (n,) or fold.dtype != np.int64 or set(fold) != set(range(5))
                or digest(fold.tolist()) != meta['native_audit']['fold_hashes'][str(seed)]):
            raise ValueError('Incomplete frozen fold identity')
        groups = pd.util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy()
        if pd.DataFrame(dict(group=groups,fold=fold)).groupby('group').fold.nunique().max() != 1:
            raise ValueError('Reference outer folds split duplicate feature groups')
        folds[seed] = fold
        for name,destination in (('current',current),('parent',parent)):
            value = arrays[f'{name}-{seed}']
            if value.shape != (n,2) or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError('Incomplete reference predictions')
            destination[seed] = {t:value[:,j].copy() for j,t in enumerate(TARGETS)}
            for j,t in enumerate(TARGETS):
                route='historical' if name=='parent' and t=='tap_iron' else 'current'
                if array_identity(value[:,j])!=reference['columns'][route][str(seed)][t]:
                    raise ValueError('Original reader prediction identity differs')
        if not np.array_equal(parent[seed]['tap_time_len'],current[seed]['tap_time_len']):
            raise ValueError('Original V32 parent time changed')
    if not np.isfinite(arrays['numeric']).all() or not np.isfinite(arrays['targets']).all():
        raise ValueError('Nonfinite reference data')
    return frame,folds,current,parent,meta
