from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bf_tap_r2.danet_optimizer import QHAdam
from bf_tap_r2.danet_network import Network
from bf_tap_r2.danet_model import Settings


def test_qhadam_matches_independent_numpy_recurrence_with_coupled_decay():
    parameter=torch.nn.Parameter(torch.tensor([2.,-3.,.1],dtype=torch.float64))
    optimizer=QHAdam([parameter],lr=.008,nus=(.8,.7),weight_decay=.02)
    values=parameter.detach().numpy().copy();first=np.zeros(3);second=np.zeros(3);weight1=weight2=0.
    for step in range(1,21):
        gradient=np.array([.3*step,-.1,.2/step]);parameter.grad=torch.tensor(gradient,dtype=torch.float64)
        coupled=gradient+.02*values;weight1=1+.9*weight1;weight2=1+.999*weight2
        first=(1-1/weight1)*first+coupled/weight1;second=(1-1/weight2)*second+coupled**2/weight2
        values-=.008*(.8*first+.2*coupled)/(np.sqrt(.7*second+.3*coupled**2)+1e-8)
        optimizer.step()
        np.testing.assert_allclose(parameter.detach(),values,atol=1e-14,rtol=0)
        np.testing.assert_allclose(optimizer.state[parameter]['exp_avg'],first,atol=1e-14,rtol=0)
        np.testing.assert_allclose(optimizer.state[parameter]['exp_avg_sq'],second,atol=1e-14,rtol=0)


def test_qhadam_frozen_parameter_has_no_decay_or_optimizer_state():
    parameter=torch.nn.Parameter(torch.tensor([2.]),requires_grad=False)
    optimizer=QHAdam([parameter]);optimizer.step()
    assert parameter.item()==2. and not optimizer.state


def test_qhadam_rejects_nonfinite_gradient_without_a_parameter_update():
    parameter=torch.nn.Parameter(torch.tensor([2.]));optimizer=QHAdam([parameter])
    parameter.grad=torch.tensor([float('nan')])
    with pytest.raises(ValueError,match='gradient'):optimizer.step()
    assert parameter.item()==2.


def test_frozen_spec_keeps_full_depth_and_all_original_promotion_gates():
    spec=yaml.safe_load(Path('configs/danet_abstract/SPEC.yaml').read_text())
    assert spec['training']==asdict(Settings())
    assert spec['future_formal_stage']['development_gate']['mean_gain_minimum']==.01
    assert spec['future_formal_stage']['confirmation_gate']['t_multiplier']==2.3533634348018264
    network=Network(26);assert len(network.blocks)==10
    assert sum(hasattr(module,'logits') for module in network.modules())==30


def test_full_network_control_starts_identically_and_raw_shortcuts_affect_output():
    torch.manual_seed(42);learned=Network(26).double().eval()
    torch.manual_seed(42);fixed=Network(26,learn_masks=False).double().eval()
    for key,value in learned.state_dict().items():assert torch.equal(value,fixed.state_dict()[key])
    query=torch.randn(5,26,dtype=torch.float64)
    torch.testing.assert_close(learned(query),fixed(query),atol=0,rtol=0)
    original=learned(query).detach()
    with torch.no_grad():learned.blocks[-1].shortcut.projection.bias.add_(1.)
    assert not torch.equal(original,learned(query))
    torch.testing.assert_close(learned(query),learned(query.flip(0)).flip(0),atol=1e-12,rtol=0)
