import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES,TARGETS
from bf_tap_r2.modernnca_reference import load_reference,array_identity,SEEDS,INCUMBENT,SCORE,OVERLAY_SHA
from bf_tap_r2.modernnca_ledger import file_hash,canonical
from bf_tap_r2.v7_periodic import digest
from test_modernnca_model import frame as artificial_frame


@pytest.fixture
def export_fixture(tmp_path):
    frame=artificial_frame()
    for j,t in enumerate(TARGETS):frame[t]=30+j+np.arange(len(frame))/10
    arrays=dict(ids=frame.sample_id.to_numpy(dtype=str),numeric=frame[list(FEATURES)].to_numpy(dtype=np.float64),
                spout=frame.spout_no.to_numpy(dtype=np.int64),targets=frame[list(TARGETS)].to_numpy(dtype=np.float64))
    columns={'current':{},'historical':{}}
    for seed in SEEDS:
        arrays[f'fold-{seed}']=np.arange(len(frame),dtype=np.int64)%5
        current=frame[list(TARGETS)].to_numpy(dtype=np.float64)+.1
        parent=current.copy();parent[:,0]+=.2
        arrays[f'current-{seed}']=current;arrays[f'parent-{seed}']=parent
        columns['current'][str(seed)]={t:array_identity(current[:,j]) for j,t in enumerate(TARGETS)}
        # Historical time is intentionally different; original V32 time is CURRENT.
        columns['historical'][str(seed)]={t:array_identity(parent[:,j] if j==0 else current[:,j]+5) for j,t in enumerate(TARGETS)}
    h=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    meta=dict(format='dnnr-current-reference-v1',candidate=INCUMBENT,score_user_reported=SCORE,parent='V32_TIME_A60V7_50',
        seeds=list(SEEDS),rows=len(frame),frame_columns=frame.columns.tolist(),frame_dtypes={k:str(v) for k,v in frame.dtypes.items()},
        frame_digest=h,new_reference_fits=0,new_candidate_fits=0,release_authorized=False,
        native_audit=dict(status='passed',new_reference_fits=0,incumbent_overlay_sha256=OVERLAY_SHA,data_digest=h,
                          fold_hashes={str(s):digest(arrays[f'fold-{s}'].tolist()) for s in SEEDS}),
        reference_identity=dict(rows=len(frame),row_ids_sha256=hashlib.sha256(canonical(frame.sample_id.tolist())).hexdigest(),columns=columns))
    def save():
        with (tmp_path/'reference.npz').open('wb') as stream:np.savez(stream,**arrays)
        meta['array_sha256']=file_hash(tmp_path/'reference.npz')
        (tmp_path/'complete.json').write_text(json.dumps(meta))
        return file_hash(tmp_path/'complete.json')
    return tmp_path,frame,arrays,meta,save


def test_export_roundtrip_preserves_rows_folds_and_correct_v32_time(export_fixture):
    root,frame,arrays,meta,save=export_fixture
    sha=save();actual,folds,current,parent,_=load_reference(root,sha,verify_external=False)
    pd.testing.assert_frame_equal(actual,frame)
    for seed in SEEDS:
        np.testing.assert_array_equal(folds[seed],arrays[f'fold-{seed}'])
        np.testing.assert_array_equal(parent[seed]['tap_time_len'],current[seed]['tap_time_len'])
        assert array_identity(parent[seed]['tap_time_len'])!=meta['reference_identity']['columns']['historical'][str(seed)]['tap_time_len']


@pytest.mark.parametrize('defect',['candidate','partial_fold','numeric_dtype','row','current','parent_time','targets','failed','stale_sha'])
def test_reference_export_rejects_rehashed_identity_and_schema_defects(export_fixture,defect):
    root,frame,arrays,meta,save=export_fixture
    initial=save()
    if defect=='candidate':meta['candidate']='V32_TIME_A60V7_50'
    elif defect=='partial_fold':arrays['fold-42'][:]=0
    elif defect=='numeric_dtype':arrays['numeric']=arrays['numeric'].astype(np.float32)
    elif defect=='row':arrays['ids'][0]='changed-row'
    elif defect=='current':arrays['current-42'][0,0]+=.1
    elif defect=='parent_time':arrays['parent-42'][0,1]+=.1
    elif defect=='targets':arrays['targets'][0,0]+=.1
    elif defect=='failed':(root/'failed.json').write_text('{}')
    else:meta['score_user_reported']=0.
    sha=save()
    with pytest.raises(ValueError):load_reference(root,initial if defect=='stale_sha' else sha,verify_external=False)


def test_duplicate_feature_groups_cannot_cross_outer_folds_even_with_new_fold_hash(export_fixture):
    root,frame,arrays,meta,save=export_fixture
    arrays['numeric'][1]=arrays['numeric'][0]
    frame.loc[1,list(FEATURES)]=frame.loc[0,list(FEATURES)].to_numpy()
    h=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    meta['frame_digest']=meta['native_audit']['data_digest']=h
    with pytest.raises(ValueError,match='duplicate'):load_reference(root,save(),verify_external=False)


def external_fixture(export_fixture):
    root,frame,arrays,meta,save=export_fixture
    native=root/'native';reader=root/'reader';overlay=root/'overlay'
    for p in (native,reader,overlay):p.mkdir()
    for p in (native,reader,overlay):(p/'marker.txt').write_text('artificial externally anchored identity')
    meta['reader_request']=dict(reader_workspace=str(reader),reader_manifest=str(reader/'manifest.json'),
        native_root=str(native),overlay_root=str(overlay),overlay_sha256=OVERLAY_SHA)
    meta['native_audit'].update(hashes={'marker.txt':file_hash(native/'marker.txt')},
        incumbent_overlay_hashes={'marker.txt':file_hash(overlay/'marker.txt')})
    meta['reference_identity']['audit']=meta['native_audit']
    meta['reader_source_hashes']={'marker.txt':file_hash(reader/'marker.txt')}
    meta['reader_runtime']=dict(python=platform.python_version(),executable=str(Path(sys.executable).resolve()),
        packages={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()},
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    parent=dict(source_hashes=meta['reader_source_hashes'],runtime=meta['reader_runtime'],experiment='ptarl_code_auxiliary_v1',
        workspace=str(reader),reference_root=str(native),incumbent_overlay_root=str(overlay),
        incumbent_overlay_sha256=OVERLAY_SHA,reference=meta['reference_identity'])
    (native/'EVIDENCE_STATUS.json').write_text(json.dumps(dict(round2_current_platform_best=dict(candidate=INCUMBENT,score=SCORE))))
    def save_all():
        (reader/'manifest.json').write_text(json.dumps(parent))
        meta['reader_request']['reader_manifest_sha256']=file_hash(reader/'manifest.json')
        return save()
    return root,meta,parent,native,reader,overlay,save_all


def test_external_source_data_overlay_runtime_and_latest_incumbent_are_revalidated(export_fixture):
    root,meta,parent,native,reader,overlay,save=external_fixture(export_fixture)
    sha=save();load_reference(root,sha)
    (reader/'marker.txt').write_text('changed source')
    with pytest.raises(ValueError,match='source changed'):load_reference(root,sha)


@pytest.mark.parametrize('defect',['native','overlay','runtime','best'])
def test_external_boundary_refuses_changed_dependencies_even_when_export_is_anchored(export_fixture,defect):
    root,meta,parent,native,reader,overlay,save=external_fixture(export_fixture)
    if defect=='native':(native/'marker.txt').write_text('changed native artifact')
    elif defect=='overlay':(overlay/'marker.txt').write_text('changed current endpoint')
    elif defect=='runtime':meta['reader_runtime']['python']='0.0.0'
    else:(native/'EVIDENCE_STATUS.json').write_text(json.dumps(dict(round2_current_platform_best=dict(candidate='historical',score=SCORE))))
    sha=save()
    with pytest.raises(ValueError,match={'native':'native reference','overlay':'incumbent overlay','runtime':'runtime','best':'incumbent'}[defect]):
        load_reference(root,sha)
