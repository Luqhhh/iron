import importlib
import numpy as np
import pytest
torch=pytest.importorskip("torch")

def mod(): return importlib.import_module("bf_tap_r2.vime_model")

def test_actual_corruption_mask_and_same_spout_donors():
 m=mod(); bank=torch.arange(40*23,dtype=torch.float32).reshape(40,23); sp=torch.arange(40)%2
 v,mask=m.corrupt(bank[:12],sp[:12],bank,sp,torch.Generator().manual_seed(12))
 assert torch.equal(mask,v[:,:21]!=bank[:12,:21])
 assert torch.equal(v[:,21:],bank[:12,21:])
 for i,j in torch.nonzero(mask): assert v[i,j].item() in bank[sp==sp[i],j].tolist()
 assert mask.any() and not mask.all()

def test_mask_plus_reconstruction_loss_independent_formula_and_gradients():
 m=mod(); z=torch.randn(7,42,requires_grad=True);x=torch.randn(7,23);mask=torch.rand(7,21)>.7
 loss=m.pretext_loss(z,x,mask)
 logits=z.detach().numpy()[:,:21];truth=mask.numpy().astype(float)
 bce=np.maximum(logits,0)-logits*truth+np.log1p(np.exp(-np.abs(logits)))
 mse=np.mean((z.detach().numpy()[:,21:]-x.numpy()[:,:21])**2)
 assert loss.item()==pytest.approx(bce.mean()+2*mse,abs=1e-6)
 loss.backward();assert torch.isfinite(z.grad).all()

def test_vime_preserves_control_initial_encoder_and_supervised_head():
 m=mod(); a,p=m.network(23)
 from bf_tap_r2.scarf_model import network
 b,_=network(23)
 assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
 assert p(torch.ones(2,256)).shape==(2,42)
