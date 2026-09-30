"""Anchored zero-fit export through the preserved PTaRL incumbent reader.

The subprocess imports that reader from its original exact-source worktree,
verifies its original manifest, and exports only already-audited references.
This module never fits or substitutes a legacy incumbent.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .dnnr_ledger import file_hash, write_new, canonical
from .dnnr_preflight import private_path, anchored
from .v7_periodic import digest

SEEDS = (42, 3407, 7777, 12011)
INCUMBENT = 'DE3_IRON_USER_REQUESTED'
SCORE = 96.3749
OVERLAY_SHA = '492c21150764c08e9714ab5321badeb2827459e23f9480126155f830021c2ab4'


def array_identity(value):
    value=np.ascontiguousarray(value,dtype='<f8')
    return hashlib.sha256(canonical(list(value.shape))+value.tobytes()).hexdigest()

EXPORT_WORKER = r'''
from pathlib import Path
import hashlib,json,sys
import numpy as np
from bf_tap_r2.data import FEATURES,TARGETS
from bf_tap_r2.ptarl_freeze import verify_manifest,reload_references
from bf_tap_r2.ptarl_protocol import write_new,file_hash
request=json.loads(sys.argv[1]);root=Path(request['output'])
m=verify_manifest(Path(request['reader_manifest']),request['reader_manifest_sha256'])
if (m['workspace']!=request['reader_workspace'] or m['reference_root']!=request['native_root']
    or m['incumbent_overlay_sha256']!=request['overlay_sha256']
    or m['incumbent_overlay_root']!=request['overlay_root']):
    raise ValueError('Reader reference/workspace identity mismatch')
frame,folds,current,historical,audit=reload_references(m)
if audit['incumbent_candidate']!='DE3_IRON_USER_REQUESTED' or audit['new_reference_fits']!=0:
    raise ValueError('Current zero-fit incumbent required')
arrays={'ids':frame.sample_id.astype(str).to_numpy(dtype=str),
    'numeric':frame[list(FEATURES)].to_numpy(dtype=np.float64),
    'spout':frame.spout_no.to_numpy(dtype=np.int64),
    'targets':frame[list(TARGETS)].to_numpy(dtype=np.float64)}
for seed in (42,3407,7777,12011):
    arrays[f'fold-{seed}']=np.asarray(folds[seed],dtype=np.int64)
    arrays[f'current-{seed}']=np.column_stack([current[seed][t] for t in TARGETS])
    # Original V32 parent keeps CURRENT time, unlike the older B0 time route.
    arrays[f'parent-{seed}']=np.column_stack([historical[seed]['tap_iron'],current[seed]['tap_time_len']])
with (root/'reference.npz').open('xb') as stream:np.savez(stream,**arrays)
verify_manifest(Path(request['reader_manifest']),request['reader_manifest_sha256'])
metadata={'format':'dnnr-current-reference-v1','candidate':'DE3_IRON_USER_REQUESTED',
    'score_user_reported':96.3749,'parent':'V32_TIME_A60V7_50','seeds':[42,3407,7777,12011],
    'rows':len(frame),'frame_columns':frame.columns.tolist(),
    'frame_dtypes':{k:str(v) for k,v in frame.dtypes.items()},
    'frame_digest':hashlib.sha256(__import__('pandas').util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),
    'array_sha256':file_hash(root/'reference.npz'),'reader_request':request,
    'reader_source_hashes':m['source_hashes'],'reader_runtime':m['runtime'],
    'native_audit':audit,'reference_identity':m['reference'],
    'new_reference_fits':0,'new_candidate_fits':0,'release_authorized':False}
write_new(root/'complete.json',metadata)
print(json.dumps({'status':'passed','rows':len(frame),'complete_sha256':file_hash(root/'complete.json'),'new_fits':0}))
'''


def export_reference(workspace, output, reader_manifest, reader_manifest_sha256, native_root, overlay_root):
    workspace = Path(workspace).resolve(); output = private_path(workspace, output)
    parent = anchored(reader_manifest,reader_manifest_sha256)
    reader = Path(parent['workspace']).resolve()
    native_root, overlay_root = Path(native_root).resolve(), Path(overlay_root).resolve()
    if (parent['experiment'] != 'ptarl_code_auxiliary_v1' or parent['reference_root'] != str(native_root)
            or parent['incumbent_overlay_root'] != str(overlay_root) or parent['incumbent_overlay_sha256'] != OVERLAY_SHA):
        raise ValueError('Original current-incumbent reader identity required')
    best = json.loads((native_root/'EVIDENCE_STATUS.json').read_text())['round2_current_platform_best']
    if best['candidate'] != INCUMBENT or best['score'] != SCORE:
        raise ValueError('Latest registered current incumbent differs before pool freeze')
    for name,sha in parent['source_hashes'].items():
        if file_hash(reader/name) != sha:
            raise ValueError('Preserved reader source changed')
    output.mkdir(parents=True,exist_ok=False)
    request = dict(reader_workspace=str(reader),reader_manifest=str(Path(reader_manifest).resolve()),
        reader_manifest_sha256=reader_manifest_sha256,native_root=str(native_root),overlay_root=str(overlay_root),
        overlay_sha256=OVERLAY_SHA,output=str(output))
    write_new(output/'started.json',dict(request=request,current_platform_best=best,
        worker_code_sha256=hashlib.sha256(EXPORT_WORKER.encode()).hexdigest(),new_fits=0))
    try:
        environment = dict(os.environ,PYTHONPATH=str(reader/'src'))
        with (output/'reader.log').open('x') as log:
            subprocess.run([sys.executable,'-c',EXPORT_WORKER,json.dumps(request)],cwd=reader,env=environment,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        result = load_reference(output,file_hash(output/'complete.json'))
        if len(result[0]) != 2754:
            raise ValueError('Official current reference row count differs')
    except BaseException as exc:
        write_new(output/'failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(complete_sha256=file_hash(output/'complete.json'),rows=len(result[0]),new_fits=0)


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
    if verify_external: verify_external_identity(meta)
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
