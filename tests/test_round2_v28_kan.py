"""Regression contracts for the frozen V28 edge spline experiment."""
import pickle
import numpy as np
import pandas as pd
import pytest
from bf_tap_r2.data import FEATURES


def frame(rows=20):
    rng = np.random.default_rng(123)
    result = pd.DataFrame(rng.uniform(-1, 1, (rows, len(FEATURES))), columns=FEATURES)
    result['sample_id'] = [f's{i}' for i in range(rows)]
    result['spout_no'] = np.resize([1, 2], rows)
    return result


def test_paired_zero_initialization_and_basis_partition():
    torch = pytest.importorskip('torch')
    from bf_tap_r2.v28_kan import EdgeKAN
    edge, shared = EdgeKAN(24, 'EDGE'), EdgeKAN(24, 'SHARED')
    for key, value in edge.state_dict().items():
        torch.testing.assert_close(value, shared.state_dict()[key], atol=0, rtol=0)
    x = torch.linspace(-1, 1, 240).reshape(10, 24)
    torch.testing.assert_close(edge(x), shared(x), atol=0, rtol=0)
    layer = edge.layers[0]
    torch.testing.assert_close(layer.b_splines(x).sum(-1), torch.ones_like(x), atol=1e-6, rtol=0)
    edge(x).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in edge.parameters())
    assert edge.layers[0].spline_weight.grad.abs().sum() > 0
    assert sum(p.numel() for p in edge.parameters()) == 56960


def test_shared_curves_tie_shapes_before_scaling():
    torch = pytest.importorskip('torch')
    from bf_tap_r2.v28_kan import SplineLinear
    edge, shared = SplineLinear(1, 2, 'EDGE'), SplineLinear(1, 2, 'SHARED')
    with torch.no_grad():
        for layer in (edge, shared):
            layer.base_weight.zero_()
            layer.spline_weight[0].fill_(1)
            layer.spline_weight[1].fill_(3)
            layer.spline_scaler.copy_(torch.tensor([[1.], [2.]]))
    x = torch.tensor([[-.5], [0.], [.5]])
    torch.testing.assert_close(edge(x), torch.tensor([[1., 6.]]).repeat(3, 1), atol=1e-6, rtol=0)
    torch.testing.assert_close(shared(x), torch.tensor([[2., 4.]]).repeat(3, 1), atol=1e-6, rtol=0)


def test_transform_is_training_only_and_unknown_category_zero():
    from bf_tap_r2.v28_kan import KANPreprocessor
    train = frame()
    pre = KANPreprocessor().fit(train)
    held = train.iloc[:2].copy()
    held.loc[:, FEATURES] = 100
    held['spout_no'] = 99
    before = pickle.dumps(pre)
    result = pre.transform(held)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, :21], 1)
    np.testing.assert_allclose(result[:, 21:], 0)
    assert pickle.dumps(pre) == before
    held.loc[held.index[0], FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match='finite'):
        pre.transform(held)

def test_refit_scaling_cold_chunk_and_optimizer_start_accounting():
    pytest.importorskip('torch')
    from bf_tap_r2.v28_kan import KANRegressor
    train = frame(60)
    y = np.linspace(10, 30, len(train))
    events = []
    model = KANRegressor('EDGE', {'max_epochs': 1}).fit(train, y, events.append)
    assert [e['stage'] for e in events] == ['inner', 'outer']
    assert all(e['event'] == 'optimizer_started' for e in events)
    assert model.mean_ == pytest.approx(20)
    assert model.std_ == pytest.approx(np.std(y))
    assert model.metadata_['inner_fit_rows'] < 60
    assert model.metadata_['selected_epoch'] == 1
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(model.predict(train), restored.predict(train, chunk_rows=7), atol=1e-6, rtol=0)
    assert model.metadata_['spline_norms'][0] > 0

def test_resource_projection_refuses_over_two_hours():
    from bf_tap_r2.v28_protocol import resource_decision
    measurements = {a: {'peak_rss_mib': 500, 'train_step_p95_seconds': .1,
        'validation_forward_p95_seconds': .01} for a in ('EDGE', 'SHARED')}
    decision = resource_decision(measurements, 8000)
    assert decision['passed']
    assert decision['projected_development_seconds'] == pytest.approx(5832)
    measurements['EDGE']['train_step_p95_seconds'] = .13
    assert not resource_decision(measurements, 8000)['passed']
    measurements['EDGE']['train_step_p95_seconds'] = .1
    assert not resource_decision(measurements, 3000)['passed']

def test_finalist_requires_complete_pairs_and_current_anchor_increment():
    from bf_tap_r2.v28_protocol import select_finalist
    rows = [{'target': t, 'arm': a, 'seed_gains': {'42': .012, '3407': .014}}
        for t in ('tap_iron','tap_time_len') for a in ('SHARED','EDGE')]
    assert select_finalist(rows) is None
    rows[0]['seed_gains'] = {'42': .001, '3407': .002}
    assert select_finalist(rows) == 'tap_iron'
    rows[1]['seed_gains']['3407'] = -.001
    assert select_finalist(rows) is None
    with pytest.raises(ValueError):
        select_finalist(rows[:3])


def test_confirmation_rejects_tampered_summary_and_unearned_target(tmp_path):
    import json, hashlib
    from bf_tap_r2.v28_protocol import require_audited_development
    identity = {'spec_sha256':'abc','source_hashes':{'core':'def'},'gate_sha256':'ghi'}
    rows = [{'target': t, 'arm': a, 'seed_gains': {'42': -.01, '3407': -.02}}
        for t in ('tap_iron','tap_time_len') for a in ('SHARED','EDGE')]
    summary = {'records':rows,'selected_for_confirmation':'tap_iron','fits':40,'failed_fits':0}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    audit = {'status':'passed','fits':40,**identity,
        'summary_sha256':hashlib.sha256((tmp_path/'summary.json').read_bytes()).hexdigest()}
    (tmp_path/'audit-r1.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='eligible'):
        require_audited_development(tmp_path,identity)
    summary['records'][0]['seed_gains']={'42':0.,'3407':0.}
    summary['records'][1]['seed_gains']={'42':.012,'3407':.015}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='hash'):
        require_audited_development(tmp_path,identity)
    audit['summary_sha256']=hashlib.sha256((tmp_path/'summary.json').read_bytes()).hexdigest()
    (tmp_path/'audit-r1.json').write_text(json.dumps(audit))
    assert require_audited_development(tmp_path,identity) == 'tap_iron'


def test_promotion_rejects_negative_seed_even_positive_mean():
    from bf_tap_r2.v28_protocol import promotion_decision
    gains={'42':.10,'3407':.11,'7777':.12,'12011':-.001}
    assert not promotion_decision(gains,[96.3,96.3])['promoted']
    gains['12011']=.09
    assert promotion_decision(gains,[96.3,96.3])['promoted']
    assert not promotion_decision(gains,[96.2,96.2])['promoted']

def test_official_ledger_requires_two_optimizers_without_duplicate_start():
    from bf_tap_r2.v28_run import validate_events
    key='tap_iron-EDGE-s42-f0'
    events=[{'event':'fit_started','key':key},
        {'event':'optimizer_started','stage':'inner','key':key},
        {'event':'optimizer_started','stage':'outer','key':key},
        {'event':'complete','key':key}]
    assert list(validate_events(events,{key})) == [key]
    with pytest.raises(ValueError):
        validate_events(events+[events[1]],{key})
    with pytest.raises(ValueError):
        validate_events(events[:2]+events[3:],{key})
    with pytest.raises(ValueError):
        validate_events(events,{key,'unstarted'})


def test_independent_score_retains_historical_increment_separately():
    from bf_tap_r2.v28_run import independent_score
    f=pd.DataFrame({'tap_iron':[100.,100.], 'tap_time_len':[10.,10.]})
    refs={'tap_iron':np.array([90.,90.]),'tap_time_len':np.array([9.,9.])}
    old={'tap_iron':np.array([80.,80.]),'tap_time_len':np.array([8.,8.])}
    result=independent_score(f,refs,old,np.array([100.,100.]),'tap_iron')
    assert result['candidate_score']==pytest.approx(91.)
    assert result['gain']==pytest.approx(1.)
    assert result['same_candidate_column_gain_vs_B0']==pytest.approx(6.)

def test_nested_run_paths_cannot_repeat_frozen_budget(tmp_path):
    from bf_tap_r2.v28_run import require_phase_paths, RUNS
    parent=tmp_path/RUNS
    parent.mkdir(parents=True)
    require_phase_paths(tmp_path,parent/'development-r1',None)
    require_phase_paths(tmp_path,parent/'confirmation-r1',parent/'development-r1')
    for output, development in ((parent/'another/development-r1',None),
            (parent/'another/confirmation-r1',parent/'development-r1'),
            (parent/'confirmation-r1',parent/'another/development-r1')):
        with pytest.raises(ValueError, match='canonical'):
            require_phase_paths(tmp_path,output,development)


def test_original_unit_chunk_invariance_at_large_scale():
    torch=pytest.importorskip('torch')
    from bf_tap_r2.v28_kan import TrainingSession, KANPreprocessor, EdgeKAN
    train=frame(322)
    session=TrainingSession.__new__(TrainingSession)
    session.preprocessor=KANPreprocessor().fit(train)
    session.model=EdgeKAN(23,'EDGE')
    with torch.no_grad():
        for layer in session.model.layers:
            layer.spline_weight.copy_(torch.linspace(-.5,.5,layer.spline_weight.numel()).reshape_as(layer.spline_weight))
            layer.base_weight.mul_(3)
    session.mean,session.std=500.,300.
    np.testing.assert_allclose(session.predict(train,256),session.predict(train,73),atol=1e-6,rtol=0)