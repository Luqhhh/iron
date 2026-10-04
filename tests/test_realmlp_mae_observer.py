import pickle
import pytest
import torch
from pytabkit.models.training.metrics import Metrics
from pytabkit.models.training.nn_creator import NNCreator
from bf_tap_r2.realmlp_native_audit import observe_native
from bf_tap_r2.realmlp_mae_observer import observe_mae_native


def factory_without_model():
    creator=object.__new__(NNCreator)
    creator.config={'train_metric_name':'mae','val_metric_name':'mae'}
    creator.n_classes=0
    return creator


def test_original_observer_retains_local_callable_and_mae_guard_preserves_author_function(tmp_path):
    creator=factory_without_model();native=Metrics.apply
    with observe_native(tmp_path/'original'):
        old,metrics=creator.get_criterions()
    with pytest.raises((AttributeError,pickle.PicklingError)):pickle.dumps(old)
    with observe_mae_native(tmp_path/'guarded') as trace:
        new,metrics=creator.get_criterions()
        assert new.func is native
    restored=pickle.loads(pickle.dumps(new))
    assert restored.func is native and restored.keywords=={'metric_name':'mae'}
    assert metrics==['mae'] and trace==dict(optimizers=[],modules=[],preprocessing=[])
    y=torch.tensor([[1.],[100.]])
    a=torch.zeros_like(y,requires_grad=True);b=torch.zeros_like(y,requires_grad=True)
    restored(a,y).backward();native(b,y,'mae').backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
