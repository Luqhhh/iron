import importlib
import importlib.util
import numpy as np
import pytest

def api():
    name='bf_tap_r2.swa_cycle_reweight'
    assert importlib.util.find_spec(name) is not None,'Cycle recency implementation is missing'
    return importlib.import_module(name)

def witness(torch):
    return {'window':5,'warmup_epochs':80,'cycle_length':20,'selected_epoch':180,'epochs':[100,120,140,160,180],
            'states':[{'weight':torch.tensor([[float(i)]]),'counter':torch.tensor(7)} for i in range(1,6)]}

def test_last3_discards_old_two_cycles_and_linear5_is_weighted():
    torch=pytest.importorskip('torch');a=api();w=witness(torch)
    assert a.reweighted_state(w,'CYCLE_LAST3')['weight'].item()==4.
    assert a.reweighted_state(w,'CYCLE_LINEAR5')['weight'].item()==pytest.approx(11/3)
    assert a.reweighted_state(w,'CYCLE_LAST3')['counter'].item()==7
    assert w['states'][0]['weight'].item()==1.

def test_one_cycle_has_no_fabricated_extra_state():
    torch=pytest.importorskip('torch');a=api();w=witness(torch);w.update(selected_epoch=100,epochs=[100],states=w['states'][:1])
    for mode in ['CYCLE_LAST3','CYCLE_LINEAR5']:assert a.reweighted_state(w,mode)['weight'].item()==1.

def test_sparse_window_and_integer_buffers_are_validated():
    torch=pytest.importorskip('torch');a=api();w=witness(torch);w['epochs'][0]=0
    with pytest.raises(ValueError,match='window'):a.reweighted_state(w,'CYCLE_LAST3')
    w=witness(torch);w['states'][-1]['counter']=torch.tensor(8)
    with pytest.raises(ValueError,match='buffer'):a.reweighted_state(w,'CYCLE_LAST3')
    with pytest.raises(ValueError,match='candidate'):a.reweighted_state(witness(torch),'unknown')

def test_prediction_restores_original_model_even_on_failure():
    torch=pytest.importorskip('torch');a=api();w=witness(torch)
    class Fake:
        def __init__(self):
            self.model_=torch.nn.Linear(1,1,bias=False);self.model_.weight.data.fill_(3)
            self.saved={'state':{'weight':torch.tensor([[3.]])},'trace':{'selected_epoch':180}}
        def predict(self,frame):
            if frame is None:raise RuntimeError('requested prediction failure')
            return self.model_(torch.tensor(frame,dtype=torch.float32)).detach().numpy()
    w['states']=[{'weight':s['weight']} for s in w['states']]
    model=Fake();pred=a.replay_candidates(model,w,[[1.],[2.]],column=0)
    np.testing.assert_allclose(pred['CYCLE_LAST3'],[4.,8.])
    assert model.model_.weight.item()==3.
    with pytest.raises(RuntimeError,match='prediction failure'):a.replay_candidates(model,w,None,column=0)
    assert model.model_.weight.item()==3.

def test_selection_caps_one_and_never_claims_formal_promotion():
    a=api();d=a.exploration_decision({'CYCLE_LAST3':[.02,.01,-.001,.01],'CYCLE_LINEAR5':[.01,.01,.01,.01]},[0,0,0,0])
    assert d['exploration_selected']==['CYCLE_LINEAR5']
    assert d['formal_promoted'] is False
    assert 'nonpositive_seed_gain' in d['candidates']['CYCLE_LAST3']['failure_reasons']

def test_candidate_needs_positive_mean_and_mechanism_with_frozen_tie_order():
    a=api();d=a.exploration_decision({'CYCLE_LAST3':[.01]*4,'CYCLE_LINEAR5':[.01]*4},[.02]*4)
    assert not d['exploration_selected']
    d=a.exploration_decision({'CYCLE_LAST3':[.01]*4,'CYCLE_LINEAR5':[.01]*4},[0]*4)
    assert d['exploration_selected']==['CYCLE_LAST3']
def test_independent_numpy_window_weights_agree_with_production():
    torch=pytest.importorskip('torch');a=api();w=witness(torch)
    from scripts.swa_cycle_reweight.replay import cold_states
    independent=cold_states(w)
    for mode in ['CYCLE_LAST3','CYCLE_LINEAR5']:
        production=a.reweighted_state(w,mode)
        for key in production:np.testing.assert_allclose(production[key].numpy(),independent[mode][key].numpy(),rtol=0,atol=5e-7)