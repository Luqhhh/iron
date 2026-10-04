from copy import deepcopy
from pathlib import Path
import torch
import yaml
from bf_tap_r2.realmlp_mae_recipe import mae_recipe


def test_author_constructor_accepts_only_the_frozen_loss_change_without_network_fitting():
    from pytabkit import RealMLP_TD_Regressor
    recipe=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/round2_v9/SPEC.yaml').read_text())['recipes']['realmlp_td']
    unchanged=deepcopy(recipe);changed=mae_recipe(recipe)
    assert recipe==unchanged
    estimator=RealMLP_TD_Regressor(**changed['constructor'])
    assert estimator.get_config()==changed['resolved']
    assert not hasattr(estimator,'alg_interface_')
    for key in ['constructor','resolved']:
        changed[key].pop('train_metric_name')
    assert changed==recipe


def test_native_mae_gradient_does_not_scale_with_residual_magnitude():
    from pytabkit.models.training.metrics import Metrics
    from pytabkit.models.training.nn_creator import mse as native_mse
    y=torch.tensor([[1.],[100.]])
    mae_pred=torch.zeros_like(y,requires_grad=True);mae=Metrics.apply(mae_pred,y,'mae');mae.backward()
    mse_pred=torch.zeros_like(y,requires_grad=True);mse=native_mse(mse_pred,y);mse.backward()
    assert float(mae.detach())==50.5
    assert abs(float(mae_pred.grad[0]/mae_pred.grad[1]))==1
    torch.testing.assert_close(mse_pred.grad[1]/mse_pred.grad[0],torch.tensor([100.]))
