from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip('torch')
pytest.importorskip('tabm'); pytest.importorskip('rtdl_num_embeddings'); pytest.importorskip('rtdl_revisiting_models')
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v7_release import save_and_cold
from bf_tap_r2.v12_joint import JointRegressor
from bf_tap_r2.v15_task_experts import TaskExpertRegressor, evaluate_fold, choose_confirmation


def sample():
    rng = np.random.default_rng(1515)
    frame = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i:04d}' for i in range(len(frame))]
    frame['spout_no'] = np.arange(len(frame)) % 3 + 1
    frame['tap_iron'] = 400 + 9*frame.air_volume + rng.normal(size=len(frame))
    frame['tap_time_len'] = 100 + 3*frame.air_volume + rng.normal(size=len(frame))
    spec = yaml.safe_load(Path('configs/round2_v15/SPEC.yaml').read_text())
    settings = {**spec['training'], 'width':16, 'tabm_k':2, 'max_epochs':3, 'patience':2,
                'embedding_dim':4, 'n_frequencies':4}
    policy = {**spec['expert_policy'], 'width':12, 'latent_dim':4}
    return frame, settings, policy


def test_periodic_control_matches_both_v12_outputs():
    frame, settings, policy = sample(); y = frame[list(TARGETS)].values
    old = JointRegressor({'backbone':'tabm','frequency':.01}, settings).fit(frame,y)
    new = TaskExpertRegressor({'control':True},settings,policy).fit(frame,y)
    np.testing.assert_array_equal(old.predict(frame),new.predict(frame))


def test_equal_initial_outputs_simplex_and_explicit_mixture():
    frame, settings, policy = sample(); models=[]
    for learned in [False,True]:
        m=TaskExpertRegressor({'learned_gates':learned},settings,policy)
        m._initialize(frame,frame[list(TARGETS)].values);m.model_.eval();models.append(m)
    x,c=models[0]._inputs(frame); a,b=[m.model_(x,c) for m in models]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    net=models[1].model_; weights=net.gate_weights(x,c)
    torch.testing.assert_close(weights,torch.full_like(weights,.5),rtol=0,atol=0)
    with torch.no_grad():
        net.gate.weight.normal_();net.gate.bias.normal_()
    weights=net.gate_weights(x,c)
    torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights[...,0]))
    experts=[m(x,c) for m in net.experts]
    expected=torch.stack([net.heads[t](sum(weights[:,t,e,None,None]*experts[e] for e in range(2))).squeeze(-1) for t in range(2)],dim=-1)
    torch.testing.assert_close(net(x,c),expected)
    assert a.shape==(len(frame),settings['tabm_k'],2)


def test_each_task_reaches_its_gate_and_both_experts():
    frame, settings, policy = sample()
    model=TaskExpertRegressor({'learned_gates':True},settings,policy)
    model._initialize(frame,frame[list(TARGETS)].values);model.model_.eval();x,c=model._inputs(frame)
    for task in range(2):
        model.model_.zero_grad(set_to_none=True)
        model.model_(x,c)[:,:,task].square().mean().backward()
        gradient=model.model_.gate.weight.grad.reshape(2,2,-1)
        assert torch.count_nonzero(gradient[task])>0
        assert torch.count_nonzero(gradient[1-task])==0
        for expert in model.model_.experts:
            assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in expert.parameters())


@pytest.mark.parametrize('learned',[False,True])
def test_partition_scales_gates_repeat_and_cold(tmp_path,learned):
    frame,settings,policy=sample();y=frame[list(TARGETS)].values;recipe={'learned_gates':learned}
    model=TaskExpertRegressor(recipe,settings,policy).fit(frame,y)
    repeat=TaskExpertRegressor(recipe,settings,policy).fit(frame,y)
    np.testing.assert_array_equal(model.predict(frame),repeat.predict(frame))
    inner=group_safe_inner_folds(frame,seed=42)['fold']!=0
    for label,mask in [('inner',inner),('outer',np.ones(len(frame),dtype=bool))]:
        np.testing.assert_allclose(model.metadata_[label+'_target_mean'],y[mask].mean(0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model.metadata_[label+'_target_std'],y[mask].std(0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model.metadata_[label+'_feature_means'],frame.loc[mask,list(FEATURES)].mean(),rtol=0,atol=1e-12)
    x,c=model._inputs(frame);weights=model.model_.gate_weights(x,c).detach().numpy()
    if learned:
        assert model.metadata_['gate_diagnostics']['parameter_norm']>0
        assert np.std(weights,axis=0).max()>0 and np.max(np.abs(weights[:,0]-weights[:,1]))>0
    else:
        np.testing.assert_array_equal(weights,np.full_like(weights,.5))
        assert model.metadata_['gate_diagnostics']['parameter_norm']==0
    query=frame.drop(columns=list(TARGETS)).copy();query['spout_no']=999
    cold=save_and_cold(model,query,model.predict(query),tmp_path/'cold',1e-6)
    assert cold['bit_identical'] and cold['training_reads_prohibited']


def test_outer_labels_and_features_excluded_from_fits():
    frame,settings,policy=sample();fv=np.arange(len(frame))%5;recipe={'learned_gates':True}
    p,meta=evaluate_fold(frame,fv,recipe,settings,policy,0)
    changed=frame.copy();changed.loc[fv==0,list(TARGETS)]=np.nan
    q,other=evaluate_fold(changed,fv,recipe,settings,policy,0)
    np.testing.assert_array_equal(p,q);assert meta==other
    changed.loc[fv==0,list(FEATURES)]*=1e6
    _,other=evaluate_fold(changed,fv,recipe,settings,policy,0);assert meta==other


def selection_fixture():
    spec=yaml.safe_load(Path('configs/round2_v15/SPEC.yaml').read_text());records=[]
    for target in spec['targets']:
        for recipe in spec['recipes']:
            comp={label:{'seed_gains':{42:.01,3407:.01},'seed_summary':{'mean':.01},
                         'cells':[{'seed':s,'fold':f} for s in [42,3407] for f in range(5)]}
                  for label in ['A60','A35','Q20','CURRENT']}
            records.append({'target':target,'recipe':recipe,'comparisons':comp})
    return spec,records


def test_selection_requires_latest_platform_and_current_both_seeds():
    spec,records=selection_fixture()
    assert choose_confirmation(records,spec)=={'target':'tap_iron','recipe':'uniform_experts'}
    for row in records:row['comparisons']['A60']['seed_gains'][3407]=-.001
    assert choose_confirmation(records,spec) is None
    for row in records:
        row['comparisons']['A60']['seed_gains'][3407]=.01
        row['comparisons']['CURRENT']['seed_gains'][42]=0
    assert choose_confirmation(records,spec) is None


@pytest.mark.parametrize('change',['missing_recipe','duplicate_recipe','missing_fold','missing_seed'])
def test_selection_rejects_incomplete_pool(change):
    spec,records=selection_fixture()
    if change=='missing_recipe':records.pop()
    elif change=='duplicate_recipe':records[-1]=deepcopy(records[0])
    elif change=='missing_fold':records[0]['comparisons']['A60']['cells'].pop()
    else:records[0]['comparisons']['CURRENT']['seed_gains'].pop(3407)
    with pytest.raises(ValueError):choose_confirmation(records,spec)
