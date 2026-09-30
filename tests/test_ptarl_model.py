from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import torch

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.ptarl_model import (PrototypeRegressor, prototype_bank, latent_forward, teacher_pair,
                                 GeometryNetwork)
from bf_tap_r2.v49_gradients import GradientRegressor, clean
from bf_tap_r2.v7_periodic import digest


def settings():
    return dict(random_seed=42, width=16, blocks=1, tabm_k=4, dropout=.1,
                embedding_dim=4, n_frequencies=4, lite=True, frequency_init_scale=.01,
                learning_rate=.001, weight_decay=.0001, batch_size=16, max_epochs=3,
                patience=25, min_delta_standardized_mae=1e-5, gradient_norm_clip=5.,
                prototype_count=3, norm_epsilon=1e-12,
                auxiliary_weights=dict(projection=.25,diversity=.25,orthogonalization=.25))


def frame():
    rng = np.random.default_rng(541)
    x = rng.normal(size=(42,len(FEATURES)))
    f = pd.DataFrame(x,columns=FEATURES)
    f['sample_id'] = [f'ptarl-synthetic-{i}' for i in range(len(x))]
    f['spout_no'] = np.arange(len(x))%2+1
    f['tap_iron'] = 20+3*x[:,0]+x[:,1]*x[:,2]
    f['tap_time_len'] = 5+np.sin(x[:,3])
    return f


def teacher_and_bank(f, s):
    with torch.random.fork_rng(devices=[]):
        teacher = GradientRegressor('BASE',s).initialize(f,f.tap_iron.to_numpy())
        teacher.train(2)
    centers, receipt = prototype_bank(teacher,clean(f),s['prototype_count'],42)
    return teacher, centers, receipt


def test_teacher_prototypes_are_partition_bound_and_deterministic():
    f,s = frame().iloc[:30],settings()
    teacher,centers,receipt = teacher_and_bank(f,s)
    again,other = prototype_bank(teacher,clean(f),s['prototype_count'],42)
    np.testing.assert_array_equal(centers,again); assert receipt==other
    assert centers.shape==(3,16) and receipt['fit_ids_digest']==digest(f.sample_id.tolist())
    assert receipt['teacher_selected_epoch']==2
    with pytest.raises(ValueError,match='identity'):
        prototype_bank(teacher,clean(f.iloc[::-1]),3,42)
    with pytest.raises(ValueError,match='unlabeled'):
        prototype_bank(teacher,f,3,42)
    with pytest.raises(ValueError,match='identity'):
        prototype_bank(teacher,clean(frame().iloc[12:]),3,42)
    altered = clean(f).copy();altered.iloc[0,0] += 1
    with pytest.raises(ValueError,match='contents'):
        prototype_bank(teacher,altered,3,42)


def test_matched_fresh_backbones_and_zero_weight_exact_training_control():
    f,s = frame().iloc[:30],settings()
    teacher,centers,receipt = teacher_and_bank(f,s)
    s['auxiliary_weights'] = dict.fromkeys(s['auxiliary_weights'],0.)
    models = [PrototypeRegressor(a,s).initialize(f,f.tap_iron.to_numpy(),centers,receipt)
              for a in ['CONTROL','PTARL_AUX']]
    for name,value in models[0].model_.state_dict().items():
        torch.testing.assert_close(value,models[1].model_.state_dict()[name],rtol=0,atol=0)
    assert models[0].initial_native_digest_==models[1].initial_native_digest_
    assert any(not torch.equal(teacher.model_.state_dict()[k],v)
               for k,v in models[0].model_.native.state_dict().items())
    before = torch.random.get_rng_state().clone()
    for m in models:
        m.train(2)
    assert torch.equal(before,torch.random.get_rng_state())
    for name,value in models[0].model_.state_dict().items():
        torch.testing.assert_close(value,models[1].model_.state_dict()[name],rtol=0,atol=0)
    assert models[0].auxiliary_updates_==models[1].auxiliary_updates_==0


def test_auxiliary_updates_change_prototypes_and_trace_each_active_term():
    f,s = frame().iloc[:30],settings()
    _,centers,receipt = teacher_and_bank(f,s)
    m = PrototypeRegressor('PTARL_AUX',s).initialize(f,f.tap_iron.to_numpy(),centers,receipt)
    m.train(2)
    assert m.auxiliary_updates_==4
    assert not np.array_equal(m.model_.prototypes.detach().numpy(),centers)
    assert all(r['auxiliary_updates']==2 for r in m.history_)
    assert all(r[k]>0 for r in m.history_ for k in s['auxiliary_weights'])
    assert m.metadata()['task_head']=='native_direct_latent'
    assert np.isfinite(m.predict(clean(frame().iloc[30:]))).all()


def test_inference_is_native_direct_head_and_ignores_prototype_coordinates():
    f,s = frame().iloc[:30],settings()
    _,centers,receipt = teacher_and_bank(f,s)
    m = PrototypeRegressor('PTARL_AUX',s).initialize(f,f.tap_iron.to_numpy(),centers,receipt)
    m.train(1)
    q = clean(frame().iloc[30:]); expected = m.predict(q)
    with torch.no_grad():
        m.model_.prototypes.fill_(123)
        for p in m.model_.coordinates.parameters():p.fill_(456)
    np.testing.assert_array_equal(m.predict(q),expected)
    _,latent = latent_forward(m.model_.native,*m._inputs(q))
    assert latent.shape==(12,16)
    assert not m.model_.native.backbone._forward_hooks


@pytest.mark.parametrize('arm',['CONTROL','PTARL_AUX'])
def test_saved_cold_hash_order_chunk_unknown_categories_and_no_overwrite(tmp_path,arm):
    f,s = frame(),settings()
    _,centers,receipt = teacher_and_bank(f.iloc[:30],s)
    m = PrototypeRegressor(arm,s).initialize(f.iloc[:30],f.tap_iron.to_numpy()[:30],centers,receipt)
    m.train(3,(clean(f.iloc[30:36]),f.tap_iron.to_numpy()[30:36]))
    path = tmp_path/'model.pt'; sha = m.save(path)
    cold = PrototypeRegressor.load(path,sha)
    q = clean(f.iloc[36:]).copy(); q.spout_no = 999
    expected = m.predict(q)
    for actual in [cold.predict(q),cold.predict(q.iloc[::-1])[::-1],
                   np.concatenate([cold.predict(q.iloc[i:i+1]) for i in range(len(q))])]:
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
    with pytest.raises(FileExistsError):m.save(path)
    with pytest.raises(ValueError,match='anchored'):PrototypeRegressor.load(path,'0'*64)
    with pytest.raises(ValueError,match='labels'):cold.predict(f.iloc[36:])
    with pytest.raises(ValueError,match='repeated fit'):m.train(1)


def test_model_partition_receipt_calibration_and_shape_rejections():
    f,s = frame(),settings()
    _,centers,receipt = teacher_and_bank(f.iloc[:30],s)
    bad = deepcopy(receipt);bad['fit_ids_digest']='bad'
    with pytest.raises(ValueError,match='partitions'):
        PrototypeRegressor('CONTROL',s).initialize(f.iloc[:30],f.tap_iron.to_numpy()[:30],centers,bad)
    with pytest.raises(ValueError,match='receipt'):
        PrototypeRegressor('CONTROL',s).initialize(f.iloc[:30],f.tap_iron.to_numpy()[:30],centers+1,receipt)
    m = PrototypeRegressor('CONTROL',s).initialize(f.iloc[:30],f.tap_iron.to_numpy()[:30],centers,receipt)
    with pytest.raises(ValueError,match='overlaps'):
        m.train(2,(clean(f.iloc[:3]),f.tap_iron.to_numpy()[:3]))
    with pytest.raises(ValueError,match='budget'):m.train(4)
    bad=deepcopy(receipt);bad['latent_rows']=29
    with pytest.raises(ValueError,match='shape/epoch'):
        PrototypeRegressor('CONTROL',s).initialize(f.iloc[:30],f.tap_iron.to_numpy()[:30],centers,bad)
    with pytest.raises(ValueError,match='Nonzero'):
        GeometryNetwork(s,3,np.zeros_like(centers))


def test_teacher_pair_fresh_outer_preprocessing_centers_and_no_state_carry():
    f,s = frame(),settings()
    inner,outer,(ib,ir),(ob,orr) = teacher_pair(f.iloc[:24],f.iloc[24:32],f.iloc[:32],'tap_iron',s)
    assert outer.selected_epoch_==inner.selected_epoch_
    assert ir['fit_ids_digest']!=orr['fit_ids_digest']
    assert ir['latent_rows']==24 and orr['latent_rows']==32
    assert not np.array_equal(ib,ob)
    assert not np.array_equal(inner.preprocessor_.means_,outer.preprocessor_.means_)
    models = [PrototypeRegressor('CONTROL',s).initialize(part,part.tap_iron.to_numpy(),bank,receipt)
              for part,bank,receipt in [(f.iloc[:24],ib,ir),(f.iloc[:32],ob,orr)]]
    assert models[0].initial_native_digest_==models[1].initial_native_digest_
    assert models[0].center_receipt_['fit_ids_digest']!=models[1].center_receipt_['fit_ids_digest']
    with pytest.raises(ValueError,match='partitions'):
        PrototypeRegressor('CONTROL',s).initialize(f.iloc[:32],f.tap_iron.to_numpy()[:32],ib,ir)
    altered=f.iloc[:32].copy();altered.iloc[0,0] += 1
    with pytest.raises(ValueError,match='contents'):
        teacher_pair(f.iloc[:24],f.iloc[24:32],altered,'tap_iron',s)
