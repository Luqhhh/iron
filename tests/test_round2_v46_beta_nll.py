"""Safety and mechanism contracts for the approved time-only beta-NLL round."""
import hashlib
import importlib
import importlib.util
import math
import numpy as np
import pandas as pd
import pytest
from bf_tap_r2.data import FEATURES


def module():
    assert importlib.util.find_spec('bf_tap_r2.v46_beta_nll') is not None, 'V46 mechanism not implemented'
    return importlib.import_module('bf_tap_r2.v46_beta_nll')


def test_source_newlines_require_git_identity_and_frozen_hash():
    m = module()
    original = b'x = 1\nprint(x)\n'
    sha = hashlib.sha256(original).hexdigest()
    assert m.verify_source_bytes(original.replace(b'\n', b'\r\n'), original, sha) == 'CRLF_only'
    assert m.verify_source_bytes(original, original, sha) == 'exact'
    with pytest.raises(ValueError, match='source'):
        m.verify_source_bytes(b'x = 2\nprint(x)\n', original, sha)
    with pytest.raises(ValueError, match='source'):
        m.verify_source_bytes(original, b'x = 2\nprint(x)\n', sha)
    with pytest.raises(ValueError, match='source'):
        m.verify_source_bytes(original, original, '0'*64)


def test_nll_values_and_detached_variance_gradients():
    torch = pytest.importorskip('torch')
    m = module()
    y = torch.tensor([1., -2.], dtype=torch.float64)
    mean = torch.tensor([.2, .3], dtype=torch.float64, requires_grad=True)
    variance = torch.tensor([.25, 4.], dtype=torch.float64, requires_grad=True)
    for beta in (0., .5):
        mean.grad = variance.grad = None
        value = m.beta_nll(y, mean, variance, beta)
        residual = y-mean.detach()
        expected = .5*(residual.square()/variance.detach()+variance.detach().log()+math.log(2*math.pi))
        torch.testing.assert_close(value, (expected*variance.detach().pow(beta)).mean())
        value.backward()
        torch.testing.assert_close(mean.grad, (mean.detach()-y)*variance.detach().pow(beta-1)/2)
        torch.testing.assert_close(variance.grad, .5*(1/variance.detach()-residual.square()/variance.detach().square())*variance.detach().pow(beta)/2)
        assert torch.isfinite(variance.grad).all()
    with pytest.raises(ValueError):
        m.beta_nll(y, mean, variance, .7)
    with pytest.raises(ValueError):
        m.beta_nll(y, mean, -variance, 0.)


def test_network_pairs_and_training_only_preprocessing():
    torch = pytest.importorskip('torch')
    m = module()
    settings = m.default_settings()
    torch.manual_seed(42)
    a = m.make_network(settings, 3).double()
    torch.manual_seed(42)
    b = m.make_network(settings, 3).double()
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
    x = torch.zeros(4, 24, dtype=torch.float64)
    _, means, scales = a(x)
    assert means.shape == scales.shape == (4,)
    assert (scales > .05).all()
    m.beta_nll(torch.ones(4, dtype=torch.float64), means, scales.square(), .5).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())
    frame = pd.DataFrame(np.arange(21*20).reshape(20,21), columns=FEATURES)
    frame['spout_no'] = np.resize([1,2],20)
    frame['sample_id'] = [f's{i}' for i in range(20)]
    pre = m.fit_preprocessor(frame)
    held = frame.iloc[:2].copy()
    held['spout_no'] = 99
    before = pre.metadata()
    _, cats = pre.transform_mlp(held)
    assert not cats.any()
    assert pre.metadata() == before


def test_fixed_recipe_and_complete_pair_eligibility():
    m = module()
    records = {a: {str(s): {'gain': .02 if a=='BETA05' else .001, 'candidate_score':96.26, 'folds':5} for s in [42,3407]} for a in ['NLL','BETA05']}
    assert m.eligible(records)
    records['BETA05']['3407']['gain'] = -.001
    assert not m.eligible(records)
    records['BETA05']['3407']['gain'] = .005
    records['BETA05']['42']['gain'] = .005
    assert not m.eligible(records)
    records['BETA05']['42']['gain'] = .02
    del records['NLL']['3407']
    with pytest.raises(ValueError, match='coverage'):
        m.eligible(records)


def test_resource_cap_includes_both_refits_and_margin():
    m = module()
    rows = {a: {'train_p95': .1, 'validation_p95':.01, 'peak_rss_mib':500} for a in ['NLL','BETA05']}
    decision = m.resource_decision(rows, 8000)
    assert decision['passed']
    assert decision['projected_development_seconds'] == pytest.approx(3216)
    rows['BETA05']['train_p95'] = 1.
    assert not m.resource_decision(rows,8000)['passed']
    rows['BETA05']['train_p95'] = .1
    assert not m.resource_decision(rows,2000)['passed']


def test_four_seed_gate_preserves_quality_and_mechanism_requirements():
    m=module()
    records={a:{str(s):{'gain':.02 if a=='BETA05' else .005,'candidate_score':96.26,'folds':5} for s in [42,3407,7777,12011]} for a in ['NLL','BETA05']}
    assert m.final_decision(records)['promoted']
    records['BETA05']['7777']['gain']=-.001
    assert not m.final_decision(records)['promoted']
    records['BETA05']['7777']['gain']=.02
    records['BETA05']['42']['candidate_score']=96.20
    assert not m.final_decision(records)['promoted']
    records['BETA05']['42']['candidate_score']=96.26
    for row in records['NLL'].values(): row['gain']=.03
    assert not m.final_decision(records)['promoted']

def test_failed_cache_audit_preserves_evidence_and_never_overwrites(tmp_path):
    pytest.importorskip('torch')
    from bf_tap_r2.v46_cache import cache_audit
    root=tmp_path/'repo'
    config=root/'configs/round2_v46_beta_nll/SPEC.yaml'
    config.parent.mkdir(parents=True)
    config.write_text('identity: V46_TIME_BETA_NLL\n')
    out=root/'local/runs/round2-v46-beta-nll/preflight-r1'
    report=cache_audit(root,out)
    assert report['status']=='stopped_cache_verification_failed'
    assert report['optimizer_runs']==report['official_fits']==report['new_reference_fits']==0
    evidence=(out/'report.json').read_bytes()
    with pytest.raises(FileExistsError):
        cache_audit(root,out)
    assert (out/'report.json').read_bytes()==evidence
    with pytest.raises(ValueError,match='Private'):
        cache_audit(root,root/'public-report')

def test_partition_uses_group_safe_fold_vector_before_optimizer(monkeypatch):
    pytest.importorskip('torch')
    m=module()
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    rng=np.random.default_rng(4600)
    train=pd.DataFrame(rng.normal(size=(40,len(FEATURES))),columns=FEATURES)
    train['sample_id']=[f't{i}' for i in range(40)]
    train['spout_no']=np.resize([1,2],40)
    train['tap_iron']=100.+np.arange(40)
    train['tap_time_len']=20.+np.arange(40)
    query=train.iloc[:2].drop(columns=['tap_iron','tap_time_len']).copy()
    query['sample_id']=['q0','q1']
    captured=[]
    def inspect_initialize(self,frame,y,on_start,stage):
        captured.append((frame.sample_id.tolist(),stage))
        raise RuntimeError('controlled_stop_before_optimizer')
    monkeypatch.setattr(m.GaussianRegressor,'initialize',inspect_initialize)
    with pytest.raises(RuntimeError,match='controlled_stop_before_optimizer'):
        m.fit_partition(train,query,'BETA05',m.default_settings(),lambda e: pytest.fail('unexpected optimizer'))
    fold=group_safe_inner_folds(train,seed=42)['fold']
    assert captured==[(train.loc[fold!=0,'sample_id'].tolist(),'inner')]
    with pytest.raises(ValueError,match='Query labels'):
        m.fit_partition(train,query.assign(tap_time_len=0),'BETA05',m.default_settings(),lambda e: pytest.fail('unexpected optimizer'))
    assert len(captured)==1

def test_original_reference_audit_requires_all_four_seeds():
    from bf_tap_r2.v46_cache import validate_native_audits
    seeds=[42,3407,7777,12011]
    paths7={f"local/runs/round2-v7-periodic-networks/{'development-r1/tap_time_len-tabm_plr001-s'+str(s)+'-f'+str(f) if s in [42,3407] else 'confirmation-r1/seed-'+str(s)+'-fold-'+str(f)}.npy":'sha' for s in seeds for f in range(5)}
    paths12={f"local/runs/round2-v12-joint-tabm/{'development-r1/joint-joint_plr001-s'+str(s)+'-f'+str(f) if s in [42,3407] else 'confirmation-r1/seed-'+str(s)+'-fold-'+str(f)}.npy":'sha' for s in seeds for f in range(5)}
    a7={'status':'PASS','prediction_hashes':paths7}
    a12={'status':'passed','prediction_count':20,'hashes':paths12}
    validate_native_audits(a7,a12)
    with pytest.raises(ValueError,match='complete original'):
        validate_native_audits(a7,{**a12,'prediction_count':10})
    incomplete=dict(paths12);incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError,match='complete original'):
        validate_native_audits(a7,{**a12,'hashes':incomplete})
    with pytest.raises(ValueError,match='complete original'):
        validate_native_audits({**a7,'status':'FAIL'},a12)
