import json

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('rtdl_revisiting_models')
from bf_tap_r2.v7_periodic import file_hash
from bf_tap_r2.v9_confirm import read_reference, _reference_events


def test_reference_cache_requires_hash_positions_iron_and_fixed_fit_identity(tmp_path):
    p=tmp_path/'fold.npz';positions=np.array([1,3,5])
    np.savez(p,positions=positions,iron=[10.,20.,30.],time=[1.,2.,3.])
    event={'sha256':file_hash(p),'reference_metadata':{'n_train':3,'n_query':3,'weights_reselected':False}}
    np.testing.assert_array_equal(read_reference(p,event,positions,6),[10.,20.,30.])
    with pytest.raises(ValueError,match='hash'):
        read_reference(p,{**event,'sha256':'incorrect'},positions,6)
    with pytest.raises(ValueError,match='positions'):
        read_reference(p,event,np.array([0,2,4]),6)
    with pytest.raises(ValueError,match='identity'):
        read_reference(p,{**event,'reference_metadata':{**event['reference_metadata'],'weights_reselected':True}},positions,6)
    np.savez(p,positions=positions,time=[1.,2.,3.])
    with pytest.raises(KeyError):
        read_reference(p,{**event,'sha256':file_hash(p)},positions,6)
    np.savez(p,positions=positions,iron=[10.,np.nan,30.])
    with pytest.raises(ValueError,match='coverage'):
        read_reference(p,{**event,'sha256':file_hash(p)},positions,6)


def test_reference_readiness_does_not_accept_partial_or_duplicate_completed_folds(tmp_path):
    config={'confirmation_seeds':[7777,12011],'folds':list(range(5))}
    assert _reference_events(tmp_path,config) is None
    rows=[{'event':'complete','seed':s,'fold':f} for s in config['confirmation_seeds'] for f in config['folds']]
    p=tmp_path/'fit_ledger.jsonl'
    p.write_text('\n'.join(map(json.dumps,rows[:-1]))+'\n')
    assert _reference_events(tmp_path,config) is None
    with p.open('a') as stream:stream.write(json.dumps(rows[-1])[:-4])
    assert _reference_events(tmp_path,config) is None
    p.write_text('\n'.join(map(json.dumps,rows))+'\n')
    assert len(_reference_events(tmp_path,config))==10
    p.write_text('\n'.join(map(json.dumps,[*rows,rows[0]]))+'\n')
    with pytest.raises(ValueError,match='Duplicate'):
        _reference_events(tmp_path,config)
    p.write_text('\n'.join(map(json.dumps,[*rows,{'event':'complete','seed':99999,'fold':0}]))+'\n')
    with pytest.raises(ValueError,match='Unexpected'):
        _reference_events(tmp_path,config)
