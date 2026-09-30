"""Behavioral checks for the frozen T2G variable graph."""
import pytest
torch = pytest.importorskip("torch")
from bf_tap_r2.t2g_core import make_model

@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)

def test_fixed_edges_closed_support():
    model=make_model("LEARNED_GRAPH",3)
    att=model.blocks[0].attention
    with torch.no_grad():
        att.column_head.fill_(1)
        att.column_tail.fill_(-1)
    a=model.topology()
    assert a.shape==(4,23,23)
    assert (a[:,1:,0]==0).all() and (a[:,0,1:]==1).all()
    assert (a[:,0,0]==0).all() and (a.sum(-1)>=1).all()
    assert (a[:,range(1,23),range(1,23)]==1).all()
    assert a[:,1:,1:].sum()==88
    x=torch.randn(2,23,64)
    w=att.weights(x)
    assert torch.isfinite(w).all()
    assert (w[...,a==0] if False else w.masked_select((a==0).unsqueeze(0))).max()==0
    assert torch.allclose(w.sum(-1),torch.ones(2,4,23))
    w.sum().backward()
    assert torch.isfinite(att.column_head.grad).all()

def test_ste_gradient():
    model=make_model("LEARNED_GRAPH",3)
    model.eval()
    y=model(torch.randn(4,21),torch.tensor([0,1,2,1]))
    y.square().sum().backward()
    for p in (model.blocks[0].attention.column_head,model.blocks[0].attention.column_tail):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum()>0

def test_paired_initial_state():
    graph=make_model("LEARNED_GRAPH",3)
    dense=make_model("DENSE_CONTROL",3)
    gs,ds=graph.state_dict(),dense.state_dict()
    assert gs.keys()==ds.keys()
    assert all(torch.equal(gs[k],ds[k]) for k in gs)
    a=dense.topology()
    assert a[:,1:,1:].sum()==4*22*22

def test_readout_only_last_block():
    model=make_model("LEARNED_GRAPH",3).eval()
    shapes=[]
    hooks=[b.register_forward_hook(lambda m,i,o:shapes.append(o.shape)) for b in model.blocks]
    n=torch.randn(4,21); c=torch.tensor([0,1,2,1])
    y=model(n,c)
    assert shapes==[torch.Size([4,23,64]),torch.Size([4,1,64])]
    for h in hooks: h.remove()
    assert y.shape==(4,) and torch.isfinite(y).all()
    assert torch.equal(y,model(n,c))
    changed=n.clone(); changed[1]+=20
    assert torch.equal(y[0],model(changed,c)[0])

@pytest.mark.parametrize("numeric,category",[
    (torch.zeros(2,20),torch.zeros(2,dtype=torch.long)),
    (torch.zeros(2,21),torch.tensor([0,3])),
    (torch.zeros(2,21),torch.tensor([0.,1.])),
    (torch.full((2,21),float("nan")),torch.tensor([0,1])),
])
def test_bad_shapes(numeric,category):
    with pytest.raises(ValueError):
        make_model("LEARNED_GRAPH",3)(numeric,category)

def test_invalid_arm_rejected():
    with pytest.raises(ValueError):
        make_model("OTHER",3)
