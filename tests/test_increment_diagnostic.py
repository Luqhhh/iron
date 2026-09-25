"""Only synthetic tests; these are NOT competition score experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
import pytest
_HERE = Path(__file__).resolve().parents[1] / "scripts/round2_v4_1"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from increment_diagnostic import exact_l1_alpha, diagnose_direction  # noqa: E402
from diagnose_v4_oof import run, v4_frame_hash  # noqa: E402


def test_bad_single_and_half_can_hide_good_small_weight():
    # Fixed constructed example, not iron data. Candidate overcorrects by 10x.
    y, b, c = np.array([100.]), np.array([90.]), np.array([190.])
    assert exact_l1_alpha(y,b,c) == pytest.approx(0.1)
    d = diagnose_direction(y,b,c)
    assert d['single_package_delta'] < 0 and d['half_blend_package_delta'] < 0
    assert d['oracle_package_delta_same_labels'] > 0
    assert d['half_blend_missed_descent']


def test_zero_direction():
    assert exact_l1_alpha([10,20],[9,22],[9,22]) == 0


def test_exact_fit_rejects_perturbation():
    d = diagnose_direction([10.,20.],[10.,20.],[15.,16.])
    assert d['loss_right_derivative_at_zero'] == 9.0
    assert d['oracle_alpha_same_labels'] == 0.0
    assert not d['descent_direction_on_these_rows']


def test_flat_minimum_prefers_zero():
    assert exact_l1_alpha([0.,1.],[0.,0.],[1.,1.]) == 0.0


@pytest.mark.parametrize('y,b,c,expected', [
    ([10.],[9.],[8.],0.), ([10.],[9.],[9.5],1.),
    ([10.],[12.],[8.],.5), ([10.],[9.],[14.],.2),
])
def test_boundaries_and_signed_directions(y,b,c,expected):
    assert exact_l1_alpha(y,b,c) == pytest.approx(expected)


@pytest.mark.parametrize('case', [
    ([],[],[]), ([1],[1,2],[1]), ([[1]],[[1]],[[1]]),
    ([np.nan],[1],[1]), ([1],[np.inf],[1]), ([1],[1],[np.nan])
])
def test_invalid_arrays(case):
    with pytest.raises(ValueError): exact_l1_alpha(*case)


@pytest.mark.parametrize('seed', range(10))
def test_exact_minimum_agrees_with_all_breakpoints(seed):
    rng=np.random.default_rng(seed)
    y=rng.uniform(20.,200.,size=64)
    b=y+rng.normal(size=64)*4
    c=b+rng.normal(size=64)*10
    r,d=y-b,c-b
    alpha=exact_l1_alpha(y,b,c)
    points=np.r_[0.,1.,np.clip(r[d!=0]/d[d!=0],0,1)]
    best=min(np.abs(r-a*d).sum() for a in points)
    assert np.abs(r-alpha*d).sum()==pytest.approx(best,rel=1e-13,abs=1e-11)
    permutation=rng.permutation(len(y))
    assert exact_l1_alpha(y[permutation],b[permutation],c[permutation])==pytest.approx(alpha)
    scale=7.3
    assert exact_l1_alpha(y*scale,b*scale,c*scale)==pytest.approx(alpha)


def test_right_derivative_matches_tie_aware_finite_difference():
    y=np.array([100.,100.,100.,100.]); b=np.array([99.,101.,100.,100.]); c=np.array([101.,100.,103.,99.])
    d=diagnose_direction(y,b,c)
    eps=1e-7
    empirical=(np.abs(y-b-eps*(c-b)).sum()-np.abs(y-b).sum())/eps
    assert d['loss_right_derivative_at_zero']==pytest.approx(empirical,abs=1e-7)


@pytest.fixture
def synthetic_repo(tmp_path, monkeypatch):
    """A mocked loader and artificial arrays; never an official data source."""
    root=tmp_path/'repo'; source=root/'local/runs/round2-v4-mechanism-search/coarse-r2'
    (source/'oof').mkdir(parents=True); (source/'reference').mkdir()
    (root/'复赛_train').mkdir()
    for name in ['train_samples.csv','train_features.csv']:
        (root/'复赛_train'/name).write_text('SYNTHETIC_TEST_STUB\n')
    n=30
    features=[f'f{i}' for i in range(21)]
    train=pd.DataFrame({c:np.arange(n,dtype=float)+i for i,c in enumerate(features)})
    train['sample_id']=[f'R2S2_TRAIN_UNIT_TEST_{i}' for i in range(n)]
    train['tap_iron']=100.+np.arange(n); train['tap_time_len']=20.+np.arange(n)
    package=types.ModuleType('bf_tap_r2'); module=types.ModuleType('bf_tap_r2.v2_release')
    module.load_v2=lambda *args: train.copy()
    monkeypatch.setitem(sys.modules,'bf_tap_r2',package)
    monkeypatch.setitem(sys.modules,'bf_tap_r2.v2_release',module)
    folds_dir=root/'local/runs/round2-v2/comparison-r1'; folds_dir.mkdir(parents=True)
    env={'data_hash':v4_frame_hash(train,features),'feature_order':features,'fold_hashes':{}}
    refs={}; lines=[]
    for seed in [42,3407]:
        folds=(np.arange(n)%5).astype(np.int64)
        a=pd.DataFrame({'sample_id':train.sample_id,'group_id':[f'g{i}' for i in range(n)],'fold':folds,'seed':seed})
        a.to_csv(folds_dir/f'folds-{seed}.csv',index=False)
        env['fold_hashes'][str(seed)]=hashlib.sha256(folds.tobytes()).hexdigest()
        mask=np.isin(folds,[0,1])
        for target in ['tap_iron','tap_time_len']:
            refs[f'{seed}_{target}']=train[target].to_numpy()+1.
        for index in range(1,41):
            method=f'v4-F-{index:03d}'; target='tap_iron' if index<=20 else 'tap_time_len'
            e={'event':'complete','status':'available','method_id':method,'family':'F','target':target,'seed':seed,'mechanism':'SYNTHETIC_TEST'}
            if index>=39:
                e.update(status='blocked',reason='SYNTHETIC_TEST_BLOCK')
            else:
                c=np.full(n,np.nan); c[mask]=train.loc[mask,target]+5.
                np.save(source/'oof'/f'{method}-seed{seed}.npy',c)
                diag=diagnose_direction(train.loc[mask,target],refs[f'{seed}_{target}'][mask],c[mask])
                e.update(data_hash=env['data_hash'],fold_hash=env['fold_hashes'][str(seed)],folds=[0,1],
                         metrics={'baseline_target_wmape':diag['baseline_target_wmape'],
                                  'package_delta_single':diag['single_package_delta'],
                                  'package_delta_equal_blend':diag['half_blend_package_delta']})
            lines.append(json.dumps(e))
    (source/'environment.json').write_text(json.dumps(env))
    (source/'fit_ledger.jsonl').write_text('\n'.join(lines)+'\n')
    np.savez_compressed(source/'reference/a_dev_oof.npz',**refs)
    return root, source


def test_adapter_complete_synthetic_ledger_and_no_overwrite(synthetic_repo):
    root,source=synthetic_repo
    source_before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    result=run(root,'local/runs/round2-v4-mechanism-search/coarse-r2','local/runs/v41-test')
    assert result['model_training_fits']==0 and result['available_seed_unit_events']==76
    manifest=json.loads((root/'local/runs/v41-test/manifest.json').read_text())
    assert manifest['available_method_target_units']==38
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in source_before.items())
    assert not list((root/'local/runs/v41-test').glob('result.csv'))
    with pytest.raises(FileExistsError):
        run(root,'local/runs/round2-v4-mechanism-search/coarse-r2','local/runs/v41-test')


def test_adapter_rejects_metric_mismatch(synthetic_repo):
    root,source=synthetic_repo; ledger=source/'fit_ledger.jsonl'
    lines=ledger.read_text().splitlines(); e=json.loads(lines[0]); e['metrics']['package_delta_single']+=1.
    lines[0]=json.dumps(e); ledger.write_text('\n'.join(lines))
    with pytest.raises(ValueError,match='Metric replay mismatch'):
        run(root,'local/runs/round2-v4-mechanism-search/coarse-r2','local/runs/v41-bad')
    assert not (root/'local/runs/v41-bad').exists()


def test_adapter_rejects_group_crossing(synthetic_repo):
    root,_=synthetic_repo; path=root/'local/runs/round2-v2/comparison-r1/folds-42.csv'
    a=pd.read_csv(path); a.loc[1,'group_id']=a.loc[0,'group_id']; a.to_csv(path,index=False)
    with pytest.raises(ValueError,match='group crosses'):
        run(root,'local/runs/round2-v4-mechanism-search/coarse-r2','local/runs/v41-bad')


def test_adapter_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError,match='nothing trained or written'):
        run(tmp_path,'local/runs/absent','local/runs/new')
    assert not (tmp_path/'local/runs/new').exists()


def test_adapter_rejects_public_output(tmp_path):
    with pytest.raises(ValueError,match='beneath'):
        run(tmp_path,'local/runs/absent','docs/results')
