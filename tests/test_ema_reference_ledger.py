"""Native accounting guards; no official data or scientific training runs."""
import json
from pathlib import Path
import threading

import pytest

import bf_tap_r2.ema_reference_ledger as ledger_module
from bf_tap_r2.ema_reference_artifacts import sha
from bf_tap_r2.ema_reference_ledger import (
    Binding, KINDS, NativeLedger, binding_sources, native_hooks, reference_bindings,
)


class SolverFixture:
    def solve(self, value, *, fail=False):
        self.entered = getattr(self, 'entered', 0) + 1
        if fail:
            raise RuntimeError('native fixture failure')
        return value


def scope(tmp_path, budget=None, sources=None):
    budget = budget or {'catboost_fit':1}
    return NativeLedger(tmp_path/'scope',
        identity=dict(source_directory=str(tmp_path.resolve()), split_seed=271828,
                      fold=0,trial_id='accounting-fixture'),
        expected={k:budget.get(k,0) for k in KINDS}, training_ids=['T0','T1','T2'], query_ids=['Q0','Q1'],
        source_hashes=sources or {str(Path(__file__).resolve()):sha(__file__)})


def test_actual_hook_preserves_result_arguments_and_original_method(tmp_path):
    original=SolverFixture.solve; obj=SolverFixture(); s=scope(tmp_path)
    token=object()
    with native_hooks([Binding(SolverFixture,'solve','catboost_fit')]):
        with s.partition(['T0','T2'],['T1']):
            assert obj.solve(token,fail=False) is token
    assert SolverFixture.solve is original and obj.entered==1
    complete=s.close()
    assert complete['counts']['catboost_fit']==1
    start=json.loads((s.directory/'call-0001/start.json').read_text())
    assert start['partition']=={'training_ids':['T0','T2'],'calibration_ids':['T1']}


def test_budget_denies_next_native_entry_and_keeps_failed_attempt(tmp_path):
    obj=SolverFixture();s=scope(tmp_path)
    with native_hooks([Binding(SolverFixture,'solve','catboost_fit')]):
        with s.partition(['T0','T1','T2']):
            obj.solve(5)
            with pytest.raises(ValueError,match='budget exceeded'):obj.solve(6)
    assert obj.entered==1
    failure=json.loads((s.directory/'call-0002/failure.json').read_text())
    assert failure['native_invoked'] is False
    with pytest.raises(ValueError,match='did not close'):s.close()
    assert (s.directory/'scope-failure.json').exists()
    with pytest.raises(FileExistsError):scope(tmp_path)


def test_swallowed_solver_failure_still_fails_closed_scope(tmp_path):
    obj=SolverFixture();s=scope(tmp_path)
    with native_hooks([Binding(SolverFixture,'solve','catboost_fit')]):
        with s.partition(['T0','T1','T2']):
            try:obj.solve(5,fail=True)
            except RuntimeError:pass
    assert obj.entered==1
    with pytest.raises(ValueError,match='Native scope failed'):s.close()
    assert (s.directory/'call-0001/failure.json').exists()


def test_solver_failure_value_is_preserved_but_scope_cannot_pass(tmp_path):
    s=scope(tmp_path); bad=(RuntimeError('returned solver error'),None)
    with s.partition(['T0','T1','T2']):
        assert s.call('catboost_fit',lambda:bad,result_failed=lambda r:r[0] is not None) is bad
    with pytest.raises(ValueError,match='Native scope failed'):s.close()


@pytest.mark.parametrize('ids,cal',[(['Q0'],[]),(['T0'],['T0']),(['T0','T0'],[]),(['T0'],['OTHER'])])
def test_partition_or_overlap_refused_before_any_native_call(tmp_path,ids,cal):
    s=scope(tmp_path)
    with pytest.raises(ValueError):
        with s.partition(ids,cal):raise AssertionError('must not enter')
    assert list(s.directory.glob('call-*'))==[]


def test_unscoped_entry_is_denied_and_hooks_restore_after_error(tmp_path):
    obj=SolverFixture();original=SolverFixture.solve
    with pytest.raises(ValueError,match='Unscoped'):
        with native_hooks([Binding(SolverFixture,'solve','catboost_fit')]):obj.solve(8)
    assert SolverFixture.solve is original and not hasattr(obj,'entered')


def test_incomplete_budget_and_repeated_close_cannot_pass(tmp_path):
    s=scope(tmp_path,budget={'catboost_fit':2})
    with s.partition(['T0','T1','T2']):s.call('catboost_fit',lambda:None)
    with pytest.raises(ValueError,match='did not close'):s.close()
    before=(s.directory/'scope-failure.json').read_bytes()
    with pytest.raises(ValueError,match='already closed'):s.close()
    assert before==(s.directory/'scope-failure.json').read_bytes()


def test_sources_changed_after_start_cannot_enter_native_solver(tmp_path):
    marker=tmp_path/'frozen_source.py';marker.write_text('original')
    s=scope(tmp_path,sources={str(marker):sha(marker)});marker.write_text('changed')
    called=[]
    with s.partition(['T0','T1','T2']):
        with pytest.raises(ValueError,match='source changed'):
            s.call('catboost_fit',lambda:called.append(1))
    assert called==[] and (s.directory/'call-0001/failure.json').exists()


def test_ledger_cannot_cross_worker_threads(tmp_path):
    s=scope(tmp_path);failures=[]
    def work():
        try:
            with s.partition(['T0','T1','T2']):pass
        except ValueError as error:failures.append(str(error))
    t=threading.Thread(target=work);t.start();t.join()
    assert failures==['Native ledger cannot be shared across workers or threads']


def test_real_torch_optimizer_constructors_are_counted_without_training(tmp_path):
    import torch
    bindings=reference_bindings();sources=binding_sources(bindings)
    sources[str(Path(ledger_module.__file__).resolve())]=sha(ledger_module.__file__)
    s=scope(tmp_path,budget={'torch_optimizer':2},sources=sources)
    parameter=torch.nn.Parameter(torch.tensor(2.))
    with native_hooks(bindings):
        with s.partition(['T0','T1','T2']):
            first=torch.optim.Adam([parameter],lr=.001)
            second=torch.optim.AdamW([parameter],lr=.001)
    assert isinstance(first,torch.optim.Adam) and isinstance(second,torch.optim.AdamW)
    assert float(parameter.detach())==2. and parameter.grad is None
    assert s.close()['counts']['torch_optimizer']==2


def test_binding_sources_cover_original_native_fit_implementations():
    bindings=reference_bindings();sources=binding_sources(bindings)
    assert len(bindings)==5 and {b.kind for b in bindings}==set(KINDS)
    assert len(sources)==5 and all(Path(p).is_file() for p in sources)


def test_invalid_hook_setup_rolls_back_installed_methods():
    original=SolverFixture.solve
    with pytest.raises(ValueError,match='Duplicate'):
        with native_hooks([Binding(SolverFixture,'solve','catboost_fit')]*2):pass
    assert SolverFixture.solve is original
    with native_hooks([Binding(SolverFixture,'solve','catboost_fit')]):pass


def test_inherited_native_method_restores_original_class_dictionary():
    class Child(SolverFixture):pass
    assert 'solve' not in vars(Child)
    with native_hooks([Binding(Child,'solve','catboost_fit')]):
        assert 'solve' in vars(Child)
    assert 'solve' not in vars(Child) and Child.solve is SolverFixture.solve


def test_equal_total_boost_count_does_not_hide_wrong_stage_mix(tmp_path):
    s=scope(tmp_path,budget={'ebm_boost':2})
    with s.partition(['T0','T1','T2']):
        s.call('ebm_boost',lambda:None,metadata={'stage':'main'})
        s.call('ebm_boost',lambda:None,metadata={'stage':'main'})
    with pytest.raises(ValueError,match='Native scope failed'):s.close()
    assert json.loads((s.directory/'scope-failure.json').read_text())['boost_stages']=={'main':2,'interaction':0}


def test_boost_bag_metadata_uses_real_native_signature_and_ordered_fit_ids():
    import inspect
    import numpy as np
    from interpret.glassbox._ebm._boost import boost
    kwargs={name:None for name in inspect.signature(boost).parameters}
    kwargs.update(bag=np.array([1,-1,1],dtype=np.int8),term_features=[(0,1)],n_inner_bags=0)
    result=ledger_module._boost_metadata((),kwargs,{'training_ids':['T2','T1','T0']})
    assert result['stage']=='interaction' and result['gradient_ids']==['T2','T0']
    assert result['calibration_ids']==['T1']
    kwargs['bag']=np.array([1,-1])
    with pytest.raises(ValueError,match='does not match'):ledger_module._boost_metadata((),kwargs,{'training_ids':['T2','T1','T0']})
