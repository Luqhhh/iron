"""Synthetic contracts for the zero-fit model restore and target composition."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from bf_tap.exceptions import ContractError
from bf_tap.optimization import platform_probe as probe
from bf_tap.optimization.v13_common import zero_fit


def test_sorted_original_comparator_is_aligned_to_caller_and_batch():
    original = pd.DataFrame(dict(sample_id=['a','b','c','d'], pred_tap_iron=[100.,110.,120.,130.],
                                pred_tap_time_len=[30.,40.,50.,60.], pred_rate=[0.,1e-6,.5,2.]))
    samples = pd.DataFrame(dict(sample_id=['d','b','a','c'], spout_no=['1']*4,
                               reference_time=pd.date_range('2024-12-02',periods=4,tz='Asia/Shanghai')))
    class HR:
        def predict_raw(self,x):
            return x[['pred_tap_iron','pred_tap_time_len']].copy()
    class Old:
        entries={'OR':{}, 'HR':{}}
        loaded={'OR':(None,None), 'HR':(HR(),None)}
        def predict(self, rows, *args):
            chosen = original[original.sample_id.isin(rows.sample_id)].sort_values('sample_id')
            return chosen[probe.COLS].copy(), chosen.copy()
    class Builder:
        def X(self, rows, *args):
            return original.set_index('sample_id').loc[rows.sample_id].set_axis(rows.index)
    class New:
        def predict(self,x):return x.pred_tap_iron.to_numpy()+10.
    def predict(rows):return probe.iron_branch(rows, Old(), Builder(), New(), .4, None,None,None)[0]
    full = predict(samples)
    assert full.sample_id.tolist()==samples.sample_id.tolist()
    expected_base = original.pred_tap_iron.to_numpy()+8.
    rate = original.pred_rate.to_numpy()
    direction = np.where(rate>1e-6, rate*original.pred_tap_time_len.to_numpy()-expected_base, 0.)
    expected = pd.DataFrame({'sample_id':original.sample_id,'pred_tap_iron':expected_base+.4*direction})
    assert np.array_equal(full.pred_tap_iron, probe.align(expected, samples.sample_id.tolist()).pred_tap_iron)
    for rows in [samples.iloc[::-1], samples.iloc[::2], samples.iloc[[2]]]:
        want = full.set_index('sample_id').loc[rows.sample_id].reset_index()
        assert predict(rows).equals(want)
    chunks = pd.concat([predict(samples.iloc[i:i+2]) for i in range(0,4,2)])
    assert probe.align(chunks,samples.sample_id.tolist()).equals(full)


def test_mean_joins_by_id_and_copies_original_iron_text():
    rows=[dict(sample_id='001', pred_tap_iron='123.000001',pred_tap_time_len='4.000000'),
          dict(sample_id='010', pred_tap_iron='456.000009',pred_tap_time_len='5.000000')]
    arrays=dict(ids=np.array(['010','001']),median=np.array([5.,4.]),mean=np.array([6.1234567,7.]))
    mean = probe.mean_rows(rows, arrays)
    assert [r['sample_id'] for r in mean]==['001','010']
    assert [r['pred_tap_iron'] for r in mean]==['123.000001','456.000009']
    assert [r['pred_tap_time_len'] for r in mean]==['7.000000','6.123457']
    assert probe.serialize_rows(mean).startswith(b'sample_id,pred_tap_iron,pred_tap_time_len\n001,')


@pytest.mark.parametrize('bad', ['duplicate','missing','integer','nan','negative','statistic'])
def test_combination_rejects_bad_worker_identity_and_values(bad):
    arrays=dict(ids=np.array(['001','010']),median=np.array([1.,2.]),mean=np.array([3.,4.]))
    iron=pd.DataFrame({'sample_id':['010','001'], 'pred_tap_iron':[100.,200.]})
    statistic='median'
    if bad=='duplicate':arrays['ids']=np.array(['001','001'])
    if bad=='missing':arrays['ids']=np.array(['001','020'])
    if bad=='integer':arrays['ids']=np.array([1,10])
    if bad=='nan':arrays['median'][0]=np.nan
    if bad=='negative':arrays['median'][0]=-1
    if bad=='statistic':statistic='blend'
    with pytest.raises(ContractError):probe.combine(iron, arrays, ['001','010'],statistic)


@pytest.mark.parametrize('extra',['train_samples','labels','external_history','test_a_samples'])
def test_stage_configuration_rejects_label_and_noncurrent_stage_paths(tmp_path,extra):
    paths={key:key+'.csv' for key in ['test_b_samples',*probe.stage_api.SOURCES]}
    paths[extra]='forbidden.csv'
    config=tmp_path/'data.yaml';config.write_text(yaml.safe_dump(dict(schema_version=1,paths=paths)))
    with pytest.raises(ContractError):probe.stage_api.stage_paths(config,'test_b')


@pytest.mark.parametrize('mutation',['training_id','before_cutoff','extra_target'])
def test_stage_metadata_blocks_history_overlap_future_model_and_targets(mutation):
    rows=pd.DataFrame(dict(sample_id=['prediction'],spout_no=['1'],reference_time=[pd.Timestamp('2025-01-01',tz='Asia/Shanghai')]))
    old=SimpleNamespace(m={'training':{'fit_cutoff':'2024-12-01 01:44:00+08:00'}},
                        loaded={'OR':(None,pd.DataFrame({'sample_id':['training']}))},history=pd.DataFrame({'sample_id':['training']}))
    if mutation=='training_id':rows.loc[0,'sample_id']='training'
    if mutation=='before_cutoff':rows.loc[0,'reference_time']=pd.Timestamp('2024-12-01',tz='Asia/Shanghai')
    if mutation=='extra_target':rows['tap_iron']=1
    with pytest.raises(ContractError):probe.stage_api.validate_metadata(rows,old)


def test_inference_fit_guards_block_recency_and_scalar_calibration():
    from bf_tap.optimization import structural
    with zero_fit() as counts:
        with pytest.raises(ContractError):probe.RecencyModel().fit(None,None,None,None)
        with pytest.raises(ContractError):structural.lad_coefficient([1],[1],[1])
    assert counts['attempted_target_fits']==counts['attempted_calibration_fits']==1
    assert counts['completed_model_fits']==counts['completed_calibration_fits']==0


def test_changed_component_manifest_rejected_before_model_load(tmp_path):
    path=tmp_path/'bundle.json';path.write_text('{}')
    with pytest.raises(ContractError):probe.bundle_assets(path,'0'*64)


def test_evidence_cannot_be_created_on_desktop(tmp_path):
    with pytest.raises(ContractError):probe.freeze(tmp_path/'evidence','test',[])
