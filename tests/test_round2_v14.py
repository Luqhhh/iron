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
from bf_tap_r2.v12_joint import JointRegressor
from bf_tap_r2.v7_release import save_and_cold
from bf_tap_r2.v14_joint_ple import JointPLERegressor,evaluate_fold,load_joint_cache


def sample():
    rng=np.random.default_rng(1313)
    frame=pd.DataFrame(rng.normal(size=(100,len(FEATURES))),columns=FEATURES)
    frame['sample_id']=[f'synthetic-{i:04d}' for i in range(len(frame))]
    frame['spout_no']=np.arange(len(frame))%3+1
    frame['tap_time_len']=100+3*frame.air_volume+rng.normal(size=len(frame))
    frame['tap_iron']=400+9*frame.air_volume+rng.normal(size=len(frame))
    spec=yaml.safe_load(Path('configs/round2_v14/SPEC.yaml').read_text())
    settings=spec['training']; settings.update(width=16,tabm_k=2,max_epochs=3,patience=2,embedding_dim=4,n_frequencies=4)
    policy={**spec['ple_policy'],'n_bins':8,'d_embedding':4}
    return frame,settings,policy


def test_control_exact_v12_both_outputs():
    frame,settings,policy=sample()
    old=JointRegressor({'backbone':'tabm','frequency':.01},settings).fit(frame,frame[list(TARGETS)].values)
    new=JointPLERegressor({'control':True},settings,policy).fit(frame,frame[list(TARGETS)].values)
    np.testing.assert_array_equal(old.predict(frame),new.predict(frame))


def test_activation_pair_same_initial_output_but_different_gradients():
    frame,settings,policy=sample(); models=[]
    for activation in [True,False]:
        model=JointPLERegressor({'activation':activation},settings,policy); model.bin_trace_=[]
        model._initialize(frame,frame[list(TARGETS)].values); model.model_.eval(); models.append(model)
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
    model=JointPLERegressor(recipe,settings,policy).fit(frame,frame[list(TARGETS)].values)
    repeat=JointPLERegressor(recipe,settings,policy).fit(frame,frame[list(TARGETS)].values)
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
    query=frame.drop(columns=list(TARGETS)).copy(); query['spout_no']=999
    result=save_and_cold(model,query,model.predict(query),tmp_path/'cold',1e-6)
    assert result['bit_identical'] and result['training_reads_prohibited']


def test_no_outer_feature_or_label_enters_bin_fits():
    frame,settings,policy=sample(); folds=np.arange(len(frame))%5
    recipe={'activation':False}
    p,meta=evaluate_fold(frame,folds,recipe,settings,policy,0)
    other=frame.copy(); other.loc[folds==0,list(TARGETS)]=np.nan
    q,meta2=evaluate_fold(other,folds,recipe,settings,policy,0)
    np.testing.assert_array_equal(p,q); assert meta==meta2
    other.loc[folds==0,list(FEATURES)]*=1e6
    _,meta3=evaluate_fold(other,folds,recipe,settings,policy,0)
    assert meta==meta3


def test_both_targets_share_repaired_embedding_gradients_and_training_scales():
    frame,settings,policy=sample(); y=frame[list(TARGETS)].values
    model=JointPLERegressor({'activation':False},settings,policy).fit(frame,y)
    inner=group_safe_inner_folds(frame,seed=42)['fold']!=0
    for label,part in [('inner',y[inner]),('outer',y)]:
        np.testing.assert_allclose(model.metadata_[label+'_target_mean'],part.mean(0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model.metadata_[label+'_target_std'],part.std(0),rtol=0,atol=1e-12)
    x,c=model._inputs(frame)
    for output in range(2):
        model.model_.zero_grad(set_to_none=True)
        model.model_(x,c)[:,:,output].square().mean().backward()
        assert torch.count_nonzero(model.model_.num_module.linear.weight.grad)>0


@pytest.mark.parametrize('corruption',['prediction','folds','training_ids','audit_hash'])
def test_joint_cache_rejects_wrong_predictions_or_identity(tmp_path,corruption):
    import json,hashlib
    from bf_tap_r2.v7_periodic import file_hash,digest
    frame,_,_=sample(); fv=np.arange(len(frame))%5; directory=tmp_path/'cache'; directory.mkdir()
    manifest={'data_digest':hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),
              'fold_digests':{'42':digest(fv.tolist())}}
    audit={'status':'passed','hashes':{}}; events=[]
    for f in range(5):
        key=f'joint-test-s42-f{f}'; path=directory/(key+'.npy')
        np.save(path,frame.loc[fv==f,list(TARGETS)].values)
        sha=file_hash(path); audit['hashes'][key]=sha
        events.append({'event':'complete','key':key,'prediction_sha256':sha,
                       'metadata':{'fit_ids_digest':digest(frame.loc[fv!=f,'sample_id'].tolist())}})
    def write():
        (directory/'manifest.json').write_text(json.dumps(manifest))
        (directory/'audit-r1.json').write_text(json.dumps(audit))
        (directory/'fit_ledger.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    write()
    actual,_=load_joint_cache(tmp_path,Path('cache'),frame,{42:fv},'test')
    np.testing.assert_array_equal(actual[42],frame[list(TARGETS)].values)
    if corruption=='prediction':np.save(directory/'joint-test-s42-f0.npy',np.zeros((20,2)))
    elif corruption=='folds':manifest['fold_digests']['42']='wrong'
    elif corruption=='training_ids':events[0]['metadata']['fit_ids_digest']='wrong'
    else:audit['hashes']['joint-test-s42-f0']='wrong'
    write()
    with pytest.raises(ValueError):load_joint_cache(tmp_path,Path('cache'),frame,{42:fv},'test')
