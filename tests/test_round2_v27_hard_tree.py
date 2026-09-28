"""Hard-tree behavior tests: catch wrong routing, gating and straight-through gradients."""
import importlib
import math
import numpy as np
import pytest

def module():
    pytest.importorskip("torch")
    try:
        return importlib.import_module("bf_tap_r2.v27_hard_tree")
    except ModuleNotFoundError as exc:
        if exc.name != "bf_tap_r2.v27_hard_tree":
            raise
        pytest.fail("V27 hard-tree learner is not implemented")

def toy(arm):
    torch=pytest.importorskip("torch")
    cls=module().HardTreeEnsemble
    model=cls(1,arm,{"n_estimators":2,"depth":1,"selected_variables":1.0,"dropout":0.0,"random_seed":42})
    with torch.no_grad():
        model.split_values.zero_()
        model.split_index_array.zero_()
        model.leaf_classes_array.copy_(torch.tensor([[1.,3.],[5.,9.]]))
        model.estimator_weights.copy_(torch.tensor([[math.log(3),0.],[0.,math.log(3)]]))
    return model.eval()

def test_hard_leaf_routing_and_instance_weighted_predictions():
    torch=pytest.importorskip("torch")
    model=toy("INSTANCE")
    x=torch.tensor([[-1.],[1.]])
    p,w=model.routing_weights(x)
    assert torch.equal(p,torch.tensor([[[1.,0.],[1.,0.]],[[0.,1.],[0.,1.]]]))
    torch.testing.assert_close(w,torch.tensor([[.75,.25],[.25,.75]]))
    torch.testing.assert_close(model(x),torch.tensor([2.,7.5]))

def test_global_gating_remains_query_invariant():
    torch=pytest.importorskip("torch")
    model=toy("GLOBAL")
    x=torch.tensor([[-1.],[1.]])
    _,w=model.routing_weights(x)
    torch.testing.assert_close(w,torch.tensor([[.5,.5],[.5,.5]]))
    torch.testing.assert_close(model(x),torch.tensor([3.,6.]))

def test_straight_through_routing_propagates_finite_nonzero_threshold_gradient():
    torch=pytest.importorskip("torch")
    model=toy("INSTANCE")
    model(torch.tensor([[-.5],[.5]])).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.split_values.grad.abs().sum().item()>0
    assert model.estimator_weights.grad.abs().sum().item()>0

def test_paired_initialization_and_predictions_are_identical():
    torch=pytest.importorskip("torch")
    cls=module().HardTreeEnsemble
    settings={"n_estimators":8,"depth":3,"selected_variables":.8,"dropout":.2,"random_seed":42}
    a,b=[cls(21,arm,settings).eval() for arm in ["GLOBAL","INSTANCE"]]
    for name,pa in a.state_dict().items():
        assert torch.equal(pa,b.state_dict()[name]),name
    torch.testing.assert_close(a(torch.ones(3,21)),b(torch.ones(3,21)),rtol=0,atol=0)

def protocol():
    try:
        return importlib.import_module("bf_tap_r2.v27_protocol")
    except ModuleNotFoundError as exc:
        if exc.name != "bf_tap_r2.v27_protocol":
            raise
        pytest.fail("V27 admission protocol is not implemented")

def test_resource_gate_rejects_memory_and_cost_without_shrinking():
    measurements={a:{"peak_rss_mib":1000.,"train_step_p95_seconds":.1,"validation_forward_p95_seconds":.02} for a in ["GLOBAL","INSTANCE"]}
    spec={"workers_max":4,"peak_worker_rss_mib_max":1536,"available_ram_margin_mib":1024,"development_projected_wall_seconds_max":21600}
    result=protocol().resource_decision(measurements,6000,spec)
    assert result["passed"]
    assert result["projected_development_seconds"]==pytest.approx(6150)
    assert not protocol().resource_decision(measurements,4500,spec)["passed"]
    measurements["GLOBAL"]["peak_rss_mib"]=1537
    assert not protocol().resource_decision(measurements,12000,spec)["passed"]
    measurements["GLOBAL"]["peak_rss_mib"]=1000
    measurements["INSTANCE"]["train_step_p95_seconds"]=.4
    assert not protocol().resource_decision(measurements,12000,spec)["passed"]

def test_finalist_requires_current_reference_gain_and_paired_mechanism_gain():
    records=[
        {"target":"tap_iron","arm":"GLOBAL","seed_gains":{"42":.02,"3407":.02}},
        {"target":"tap_iron","arm":"INSTANCE","seed_gains":{"42":.015,"3407":.015}},
        {"target":"tap_time_len","arm":"GLOBAL","seed_gains":{"42":.001,"3407":.001}},
        {"target":"tap_time_len","arm":"INSTANCE","seed_gains":{"42":.01,"3407":.01}}]
    assert protocol().select_finalist(records)=="tap_time_len"
    records[-1]["seed_gains"]["3407"]=-.001
    assert protocol().select_finalist(records) is None
    records[-1]["seed_gains"]={"42":.009,"3407":.009}
    assert protocol().select_finalist(records) is None

def test_promotion_requires_all_four_seeds_positive_and_unchanged_local_gate():
    gains={str(s):.02 for s in [42,3407,7777,12011]}
    out=protocol().promotion_decision(gains,[96.26,96.26])
    assert out["promoted"] and out["lcb95"]==pytest.approx(.02)
    assert not protocol().promotion_decision(gains,[96.24,96.24])["promoted"]
    gains["12011"]=-.001
    assert not protocol().promotion_decision(gains,[96.26,96.26])["promoted"]
    with pytest.raises(ValueError,match="four"):
        protocol().promotion_decision({"42":.02,"3407":.02},[96.26,96.26])
