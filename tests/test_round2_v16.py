from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip('torch')
pytest.importorskip('pytorch_tabnet'); pytest.importorskip('tabm')
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v7_release import save_and_cold
from bf_tap_r2.v12_joint import JointRegressor, joint_loss
from bf_tap_r2.v16_sequential_masks import (
    SequentialMaskRegressor, evaluate_fold, choose_confirmation, training_batches,
    implementation_hashes)


def sample():
    rng = np.random.default_rng(1616)
    frame = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    frame['sample_id'] = [f'synthetic-{i:04d}' for i in range(len(frame))]
    frame['spout_no'] = np.arange(len(frame)) % 3 + 1
    frame['tap_iron'] = 400 + 9*frame.air_volume + rng.normal(size=len(frame))
    frame['tap_time_len'] = 100 + 3*frame.air_volume + rng.normal(size=len(frame))
    spec = yaml.safe_load(Path('configs/round2_v16/SPEC.yaml').read_text())
    settings = {**spec['training'], 'width':16, 'tabm_k':2, 'max_epochs':3, 'patience':2,
                'embedding_dim':4, 'n_frequencies':4}
    policy = {**spec['tabnet_policy'], 'n_d':8, 'n_a':8}
    return frame, settings, policy


def test_implementation_is_pinned_and_control_replays_both_targets():
    spec = yaml.safe_load(Path('configs/round2_v16/SPEC.yaml').read_text())
    assert implementation_hashes() == spec['implementation_hashes']
    frame, settings, policy = sample(); y = frame[list(TARGETS)].values
    old = JointRegressor({'backbone':'tabm','frequency':.01}, settings).fit(frame,y)
    new = SequentialMaskRegressor({'control':True}, settings, policy).fit(frame,y)
    np.testing.assert_array_equal(old.predict(frame),new.predict(frame))


def test_uniform_masks_equal_sparsemax_with_zero_attention_logits():
    frame,settings,policy=sample();models=[]
    for learned in [False,True]:
        model=SequentialMaskRegressor({'learned_masks':learned},settings,policy)
        model._initialize(frame,frame[list(TARGETS)].values);model.model_.eval();models.append(model)
    for attention in models[1].model_.network.encoder.att_transformers:
        torch.nn.init.zeros_(attention.fc.weight)
    x,c=models[0]._inputs(frame)
    torch.testing.assert_close(models[0].model_(x,c),models[1].model_(x,c),atol=1e-7,rtol=1e-6)
    masks=models[0].model_.masks(x,c)
    torch.testing.assert_close(masks,torch.full_like(masks,1/masks.shape[-1]))
    assert models[0].model_.entropy_penalty.item() == pytest.approx(np.log(masks.shape[-1]))


@pytest.mark.parametrize('learned',[False,True])
def test_attention_gradient_attribution_and_entropy_sign(learned):
    frame,settings,policy=sample()
    model=SequentialMaskRegressor({'learned_masks':learned},settings,policy)
    model._initialize(frame,frame[list(TARGETS)].values);x,c=model._inputs(frame)
    pred=model.model_(x,c);penalty=model.model_.entropy_penalty
    assert penalty>=0 and penalty<=np.log(x.shape[-1]+model.model_.n_categories)+1e-5
    joint_loss(pred,torch.zeros(len(frame),2)).backward()
    attention=[p for a in model.model_.network.encoder.att_transformers for p in a.parameters()]
    if learned:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in attention)
        assert (model.model_.masks(x,c)==0).any()
    else:assert all(p.grad is None for p in attention)
    assert any(p.grad is not None and p.grad.abs().sum()>0
               for p in model.model_.network.encoder.feat_transformers.parameters())


@pytest.mark.parametrize('learned',[False,True])
def test_partition_scales_determinism_and_cold_eval_batch_invariance(tmp_path,learned):
    frame,settings,policy=sample();y=frame[list(TARGETS)].values;recipe={'learned_masks':learned}
    model=SequentialMaskRegressor(recipe,settings,policy).fit(frame,y)
    repeat=SequentialMaskRegressor(recipe,settings,policy).fit(frame,y)
    np.testing.assert_array_equal(model.predict(frame),repeat.predict(frame))
    inner=group_safe_inner_folds(frame,seed=42)['fold']!=0
    for label,mask in [('inner',inner),('outer',np.ones(len(frame),dtype=bool))]:
        np.testing.assert_allclose(model.metadata_[label+'_target_mean'],y[mask].mean(0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model.metadata_[label+'_target_std'],y[mask].std(0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model.metadata_[label+'_feature_means'],frame.loc[mask,list(FEATURES)].mean(),rtol=0,atol=1e-12)
    diagnostics=model.metadata_['mask_diagnostics']
    assert diagnostics['simplex_max_error']<1e-5
    if learned:
        assert diagnostics['attention_gradient_norm']>0 and diagnostics['maximum_row_std']>0
    else:
        assert diagnostics['attention_gradient_norm']==diagnostics['maximum_row_std']==0
        assert diagnostics['nonzero_fraction_by_step']==[1.]*policy['n_steps']
    query=frame.drop(columns=list(TARGETS)).copy();query['spout_no']=999
    cold=save_and_cold(model,query,model.predict(query),tmp_path/'cold',1e-6)
    assert cold['bit_identical'] and cold['training_reads_prohibited']


def test_outer_target_and_feature_changes_do_not_change_training():
    frame,settings,policy=sample();fv=np.arange(len(frame))%5;recipe={'learned_masks':True}
    p,meta=evaluate_fold(frame,fv,recipe,settings,policy,0)
    changed=frame.copy();changed.loc[fv==0,list(TARGETS)]=np.nan
    q,other=evaluate_fold(changed,fv,recipe,settings,policy,0)
    np.testing.assert_array_equal(p,q);assert meta==other
    changed.loc[fv==0,list(FEATURES)]*=1e6
    _,other=evaluate_fold(changed,fv,recipe,settings,policy,0);assert meta==other


@pytest.mark.parametrize('n',[2,255,256,257,513])
def test_training_batches_cover_all_rows_without_singletons(n):
    order=np.arange(n)[::-1];batches=training_batches(order,256)
    assert all(len(b)>=2 for b in batches)
    np.testing.assert_array_equal(np.concatenate(batches),order)


def selection_fixture():
    spec=yaml.safe_load(Path('configs/round2_v16/SPEC.yaml').read_text());records=[]
    for target in spec['targets']:
        for recipe in spec['recipes']:
            comp={label:{'seed_gains':{42:.01,3407:.01},'seed_summary':{'mean':.01},
                         'cells':[{'seed':s,'fold':f} for s in [42,3407] for f in range(5)]}
                  for label in ['V12_PLATFORM','A60','A35','Q20','CURRENT']}
            records.append({'target':target,'recipe':recipe,'comparisons':comp})
    return spec,records


def test_selection_requires_actual_v12_platform_and_pending_v7_reference():
    spec,records=selection_fixture()
    assert choose_confirmation(records,spec)=={'target':'tap_iron','recipe':'uniform_masks'}
    for row in records:row['comparisons']['V12_PLATFORM']['seed_gains'][3407]=-.001
    assert choose_confirmation(records,spec) is None
    for row in records:
        row['comparisons']['V12_PLATFORM']['seed_gains'][3407]=.01
        row['comparisons']['CURRENT']['seed_gains'][42]=0
    assert choose_confirmation(records,spec) is None


@pytest.mark.parametrize('change',['missing_recipe','duplicate_recipe','missing_fold','missing_seed'])
def test_selection_rejects_incomplete_pool(change):
    spec,records=selection_fixture()
    if change=='missing_recipe':records.pop()
    elif change=='duplicate_recipe':records[-1]=deepcopy(records[0])
    elif change=='missing_fold':records[0]['comparisons']['V12_PLATFORM']['cells'].pop()
    else:records[0]['comparisons']['CURRENT']['seed_gains'].pop(3407)
    with pytest.raises(ValueError):choose_confirmation(records,spec)
