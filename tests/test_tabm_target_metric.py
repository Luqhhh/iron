import importlib
import pytest
import numpy as np

def test_supervised_metric_finite_gradient_and_target_order():
 torch=pytest.importorskip("torch");m=importlib.import_module("bf_tap_r2.tabm_metric_models")
 # Identical-label pairs should prefer identical normalized representation.
 h=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],requires_grad=True);y=torch.tensor([0.,0.,1.,1.])
 loss=m.target_metric_loss(h,y);loss.backward();assert torch.isfinite(loss) and torch.isfinite(h.grad).all()
 assert m.target_metric_loss(h.detach(),y)<m.target_metric_loss(h.detach(),torch.tensor([0.,1.,0.,1.]))
 perm=torch.tensor([2,0,3,1]);torch.testing.assert_close(m.target_metric_loss(h.detach()[perm],y[perm]),loss.detach())

def test_metric_singleton_zero_and_invalid_target_rejected():
 torch=pytest.importorskip("torch");m=importlib.import_module("bf_tap_r2.tabm_metric_models")
 h=torch.ones((1,4),requires_grad=True);m.target_metric_loss(h,torch.ones(1)).backward();assert not h.grad.any()
 with pytest.raises(ValueError):m.target_metric_loss(torch.ones((2,4)),torch.tensor([float("nan"),1.]))

def test_budget_failure_counts_and_mechanism_gate(tmp_path):
 m=importlib.import_module("bf_tap_r2.tabm_metric_protocol");ledger=m.Ledger(tmp_path)
 ledger.reserve("engineering","optimizer","failed-selection")
 with pytest.raises(ValueError):ledger.reserve("engineering","optimizer","failed-selection")
 assert ledger.counts("engineering")["reserved"]["optimizer"]==1
 assert ledger.counts("engineering")["completed"]["optimizer"]==0
 assert not m.decision([.01,.02],[.02,.03],confirmation=False)["selected_for_confirmation"]
