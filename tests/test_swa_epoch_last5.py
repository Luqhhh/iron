import importlib
import importlib.util
import numpy as np
import pytest

def api():
    name="bf_tap_r2.swa_epoch_last5"
    assert importlib.util.find_spec(name) is not None, "Last-five implementation missing"
    return importlib.import_module(name)

def witness(torch, epoch=10):
    return {"window":10,"selected_epoch":epoch,"epochs":list(range(max(1,epoch-9),epoch+1)),
            "states":[{"weight":torch.tensor([[float(i)]]),"counter":torch.tensor(7)} for i in range(max(1,epoch-9),epoch+1)]}

def test_last_five_parameter_mean_excludes_old_five_and_preserves_original():
    torch=pytest.importorskip("torch");a=api();w=witness(torch)
    assert a.reweighted_state(w,"EPOCH_LAST5")["weight"].item()==8.
    assert a.reweighted_state(w,"EPOCH_LAST5")["counter"].item()==7
    assert w["states"][0]["weight"].item()==1.

def test_early_epoch_uses_only_existing_states():
    torch=pytest.importorskip("torch");a=api();w=witness(torch,3)
    assert a.reweighted_state(w,"EPOCH_LAST5")["weight"].item()==2.

def test_rejects_sparse_window_and_inconsistent_buffers():
    torch=pytest.importorskip("torch");a=api();w=witness(torch);w["epochs"][0]=0
    with pytest.raises(ValueError,match="window"):a.reweighted_state(w,"EPOCH_LAST5")
    w=witness(torch);w["window"]=5
    with pytest.raises(ValueError,match="window"):a.reweighted_state(w,"EPOCH_LAST5")
    w=witness(torch);w["states"][-1]["counter"]=torch.tensor(8)
    with pytest.raises(ValueError,match="buffer"):a.reweighted_state(w,"EPOCH_LAST5")
    w=witness(torch);w["states"][-1]["weight"].fill_(float("nan"))
    with pytest.raises(ValueError,match="Nonfinite"):a.reweighted_state(w,"EPOCH_LAST5")
    with pytest.raises(ValueError,match="candidate"):a.reweighted_state(witness(torch),"unknown")

def test_prediction_restores_original_model_even_on_failure():
    torch=pytest.importorskip("torch");a=api();w=witness(torch)
    class Fake:
        def __init__(self):
            self.model_=torch.nn.Linear(1,1,bias=False);self.model_.weight.data.fill_(5.5)
            self.saved={"state":{"weight":torch.tensor([[5.5]])},"trace":{"selected_epoch":10}}
        def predict(self,frame):
            if frame is None:raise RuntimeError("requested prediction failure")
            return self.model_(torch.tensor(frame,dtype=torch.float32)).detach().numpy()
    w["states"]=[{"weight":s["weight"]} for s in w["states"]]
    model=Fake();pred=a.replay_candidates(model,w,[[1.],[2.]],column=0)
    np.testing.assert_allclose(pred["EPOCH_LAST5"],[8.,16.])
    assert model.model_.weight.item()==5.5
    with pytest.raises(RuntimeError,match="prediction failure"):a.replay_candidates(model,w,None,column=0)
    assert model.model_.weight.item()==5.5

def test_used_seeds_never_claim_formal_promotion():
    a=api();d=a.exploration_decision({"EPOCH_LAST5":[.02,.01,-.001,.01]},[0]*4)
    assert d["exploration_selected"]==["EPOCH_LAST5"]
    assert d["formal_promoted"] is False
    assert d["release_authorized"] is False
    assert "nonpositive_seed_gain" in d["candidates"]["EPOCH_LAST5"]["failure_reasons"]

def test_positive_mean_needs_positive_mechanism_and_complete_four_seeds():
    a=api();d=a.exploration_decision({"EPOCH_LAST5":[.01]*4},[.02]*4)
    assert not d["exploration_selected"]
    with pytest.raises(ValueError,match="complete"):a.exploration_decision({"EPOCH_LAST5":[.01]*3},[0]*4)

def test_independent_numpy_mean_matches_production():
    torch=pytest.importorskip("torch");a=api();w=witness(torch)
    from scripts.swa_epoch_last5.replay import cold_states
    independent=cold_states(w)["EPOCH_LAST5"];production=a.reweighted_state(w,"EPOCH_LAST5")
    for key in production:np.testing.assert_allclose(production[key].numpy(),independent[key].numpy(),rtol=0,atol=5e-7)
