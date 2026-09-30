from copy import deepcopy

import numpy as np
import pytest

from bf_tap_r2 import ptarl_reference as module
from bf_tap_r2.ptarl_protocol import write_new,file_hash
from bf_tap_r2.v7_periodic import digest
from test_ptarl_model import frame


def overlay(tmp_path):
    f=frame();folds={s:np.arange(len(f))%5 for s in module.SEEDS}
    columns={}
    for seed in module.SEEDS:
        path=tmp_path/f'iron-{seed}.npy'
        with path.open('xb') as stream:np.save(stream,f.tap_iron.to_numpy()+seed*1e-6)
        columns[str(seed)]=dict(path=path.name,sha256=file_hash(path))
    audit=dict(status='passed',candidate=module.INCUMBENT,verified_split_seeds=list(module.SEEDS),
        new_audit_fits=0,scope='incumbent_reference_only_not_DE3_promotion',
        artifact_hashes={r['path']:r['sha256'] for r in columns.values()})
    write_new(tmp_path/'audit.json',audit)
    complete=dict(candidate=module.INCUMBENT,parent='V32_TIME_A60V7_50',
        endpoint='maximum(parent_iron+0.5*(mean(J42,J104729,J130363)-J42),0)',training_seeds=[42,104729,130363],
        iron_columns=columns,audit_sha256=file_hash(tmp_path/'audit.json'),
        row_ids_digest=digest(f.sample_id.tolist()),native_data_digest='native-data',
        fold_digests={str(s):digest(folds[s].tolist()) for s in module.SEEDS})
    write_new(tmp_path/'complete.json',complete)
    current={s:dict(tap_iron=f.tap_iron.to_numpy()+1,tap_time_len=f.tap_time_len.to_numpy()+1) for s in module.SEEDS}
    historical=deepcopy(current)
    native=(f,folds,current,historical,dict(status='passed',hashes={},data_digest='native-data',new_reference_fits=0))
    return complete,native


def test_current_overlay_keeps_time_and_historical_values_and_binds_all_four_seeds(tmp_path,monkeypatch):
    complete,native=overlay(tmp_path)
    original=deepcopy(native)
    monkeypatch.setattr(module,'native_loader',lambda _:deepcopy(native))
    f,folds,current,historical,audit=module.load_current_reference(tmp_path,tmp_path,file_hash(tmp_path/'complete.json'))
    assert audit['incumbent_candidate']=='DE3_IRON_USER_REQUESTED'
    assert audit['incumbent_score_user_reported']==96.3749
    assert audit['original_DE3_no_finalist_and_no_confirmation_unchanged']
    for seed in module.SEEDS:
        np.testing.assert_array_equal(current[seed]['tap_time_len'],original[2][seed]['tap_time_len'])
        np.testing.assert_array_equal(historical[seed]['tap_iron'],original[3][seed]['tap_iron'])
        np.testing.assert_array_equal(current[seed]['tap_iron'],f.tap_iron.to_numpy()+seed*1e-6)
        assert digest(folds[seed].tolist())==complete['fold_digests'][str(seed)]


def test_two_development_seeds_cannot_masquerade_as_four_seed_current_reference(tmp_path,monkeypatch):
    complete,_=overlay(tmp_path)
    complete['iron_columns']={k:v for k,v in complete['iron_columns'].items() if k in ['42','3407']}
    # This is an intentionally tampered test copy; all real evidence is untouched.
    (tmp_path/'complete.json').write_text(__import__('json').dumps(complete))
    read=[];monkeypatch.setattr(module,'native_loader',lambda _:read.append(True))
    with pytest.raises(ValueError,match='four-seed'):
        module.load_current_reference(tmp_path,tmp_path,file_hash(tmp_path/'complete.json'))
    assert read==[]


def test_reference_endpoint_change_rejected_before_native_label_loader(tmp_path,monkeypatch):
    _,_=overlay(tmp_path)
    with (tmp_path/'iron-7777.npy').open('ab') as stream:stream.write(b'corruption')
    read=[];monkeypatch.setattr(module,'native_loader',lambda _:read.append(True))
    with pytest.raises(ValueError,match='artifact changed'):
        module.load_current_reference(tmp_path,tmp_path,file_hash(tmp_path/'complete.json'))
    assert read==[]


def test_current_overlay_must_use_identical_outer_split_identity(tmp_path,monkeypatch):
    _,native=overlay(tmp_path)
    native[1][12011]=np.roll(native[1][12011],1)
    monkeypatch.setattr(module,'native_loader',lambda _:native)
    with pytest.raises(ValueError,match='split identity'):
        module.load_current_reference(tmp_path,tmp_path,file_hash(tmp_path/'complete.json'))
