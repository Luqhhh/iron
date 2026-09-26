from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

torch=pytest.importorskip('torch')
pytest.importorskip('tabm'); pytest.importorskip('rtdl_num_embeddings'); pytest.importorskip('rtdl_revisiting_models')
from rtdl_num_embeddings import compute_bins
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v3_6_networks import NumericPreprocessor
from bf_tap_r2.v7_periodic import PeriodicRegressor
from bf_tap_r2.v7_release import save_and_cold
from bf_tap_r2.v13_ple import PLERegressor,evaluate_fold


def sample():
    rng=np.random.default_rng(1313)
    frame=pd.DataFrame(rng.normal(size=(100,len(FEATURES))),columns=FEATURES)
    frame['sample_id']=[f'synthetic-{i:04d}' for i in range(len(frame))]
    frame['spout_no']=np.arange(len(frame))%3+1
    frame['tap_time_len']=100+3*frame.air_volume+rng.normal(size=len(frame))
    frame['tap_iron']=400+9*frame.air_volume+rng.normal(size=len(frame))
    spec=yaml.safe_load(Path('configs/round2_v13/SPEC.yaml').read_text())
    settings=spec['training']; settings.update(width=16,tabm_k=2,max_epochs=3,patience=2,embedding_dim=4,n_frequencies=4)
    policy={**spec['ple_policy'],'n_bins':8,'d_embedding':4}
    return frame,settings,policy


def test_control_exact_v7():
    frame,settings,policy=sample()
    old=PeriodicRegressor({'backbone':'tabm','frequency':.01},settings).fit(frame,frame.tap_time_len.values)
    new=PLERegressor({'control':True},settings,policy).fit(frame,frame.tap_time_len.values)
    np.testing.assert_array_equal(old.predict(frame),new.predict(frame))


def test_activation_pair_same_initial_output_but_different_gradients():
    frame,settings,policy=sample(); models=[]
    for activation in [True,False]:
        model=PLERegressor({'activation':activation},settings,policy); model.bin_trace_=[]
        model._initialize(frame,frame.tap_time_len.values); model.model_.eval(); models.append(model)
    x,c=models[0]._inputs(frame)
    out=[m.model_(x,c) for m in models]
    torch.testing.assert_close(out[0],out[1],rtol=0,atol=0)
    for active,model,p in zip([True,False],models,out):
        p.square().mean().backward()
        gradient=model.model_.num_module.linear.weight.grad
        assert (torch.count_nonzero(gradient)==0) if active else (torch.count_nonzero(gradient)>0)


@pytest.mark.parametrize('activation',[True,False])
def test_fresh_training_bins_learned_weights_repeat_and_cold(tmp_path,activation):
    frame,settings,policy=sample(); recipe={'activation':activation}
    model=PLERegressor(recipe,settings,policy).fit(frame,frame.tap_time_len.values)
    repeat=PLERegressor(recipe,settings,policy).fit(frame,frame.tap_time_len.values)
    np.testing.assert_array_equal(model.predict(frame),repeat.predict(frame))
    inner=group_safe_inner_folds(frame,seed=42)['fold']!=0
    assert len(model.metadata_['bin_trace'])==2
    for trace,part in zip(model.metadata_['bin_trace'],[frame.loc[inner],frame]):
        numeric,_=NumericPreprocessor(structure='raw_tabm').fit(part).transform_tabm(part)
        expected=compute_bins(torch.as_tensor(numeric),n_bins=policy['n_bins'])
        assert trace['bins']==[b.tolist() for b in expected]
        assert trace['rows']==len(part)
    n=model.metadata_['nonlinear_weights_nonzero']
    assert (n==0) if activation else (n>0)
    result=save_and_cold(model,frame,model.predict(frame),tmp_path/'cold',1e-6)
    assert result['bit_identical'] and result['training_reads_prohibited']


def test_no_outer_feature_or_label_enters_bin_fits():
    frame,settings,policy=sample(); folds=np.arange(len(frame))%5
    recipe={'activation':False}
    p,meta=evaluate_fold(frame,folds,'tap_time_len',recipe,settings,policy,0)
    other=frame.copy(); other.loc[folds==0,list(TARGETS)]=np.nan
    q,meta2=evaluate_fold(other,folds,'tap_time_len',recipe,settings,policy,0)
    np.testing.assert_array_equal(p,q); assert meta==meta2
    other.loc[folds==0,list(FEATURES)]*=1e6
    _,meta3=evaluate_fold(other,folds,'tap_time_len',recipe,settings,policy,0)
    assert meta==meta3
