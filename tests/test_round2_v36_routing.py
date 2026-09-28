import pytest
torch=pytest.importorskip("torch")
from bf_tap_r2.v36_hard_tree_reference import HardTreeEnsemble as Reference
from bf_tap_r2.v36_hard_tree import HardTreeEnsemble


@pytest.mark.parametrize("arm",["GLOBAL","INSTANCE"])
@pytest.mark.parametrize("training",[False,True])
@pytest.mark.parametrize("nonzero_weights",[False,True])
def test_routing_outputs_and_all_gradients_match_reference(arm,training,nonzero_weights):
    torch.set_num_threads(1)
    s=dict(n_estimators=12,depth=4,dropout=.2,selected_variables=.8,random_seed=42)
    old=Reference(24,arm,s);new=HardTreeEnsemble(24,arm,s)
    if nonzero_weights:
        with torch.no_grad():old.estimator_weights.copy_(torch.randn_like(old.estimator_weights))
    new.load_state_dict(old.state_dict())
    old.train(training);new.train(training)
    x=torch.randn(17,24,requires_grad=True);x2=x.detach().clone().requires_grad_(True)
    torch.manual_seed(901);a=old(x)
    torch.manual_seed(901);b=new(x2)
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
    a.square().sum().backward();b.square().sum().backward()
    assert torch.allclose(x.grad,x2.grad,atol=1e-6,rtol=1e-5)
    for (name,p),(other,q) in zip(old.named_parameters(),new.named_parameters()):
        assert name==other
        assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5),name
    new.eval();paths,_=new.routing_weights(x2.detach())
    assert torch.equal(paths.sum(-1),torch.ones_like(paths.sum(-1)))
def test_resource_admission_keeps_original_limits():
    from bf_tap_r2.v36_preflight import admission
    rows = [{"arm": arm, "training_p95_seconds": .3,
             "evaluation_p95_seconds": .1, "peak_rss_mib": 1000}
            for arm in ("GLOBAL", "INSTANCE")]
    assert admission(rows, 8000)["passed"]
    rows[1]["training_p95_seconds"] = .4
    assert not admission(rows, 8000)["checks"]["time"]
    rows[1]["training_p95_seconds"] = .3
    rows[1]["peak_rss_mib"] = 1537
    assert not admission(rows, 8000)["checks"]["worker_rss"]
    rows[1]["peak_rss_mib"] = 1000
    assert not admission(rows, 5000)["checks"]["available_ram"]
