import importlib
import pytest

def test_uniform_window_epoch_identity_and_copy_isolation():
 torch=pytest.importorskip("torch");m=importlib.import_module("bf_tap_r2.tabm_swa_window");w=m.EpochWindow(3)
 original={"w":torch.tensor([1.,3.]),"counter":torch.tensor(2)}
 w.append(1,original);original["w"].fill_(99)
 torch.testing.assert_close(w.average()["w"],torch.tensor([1.,3.]))
 for epoch in (2,3,4):w.append(epoch,{"w":torch.tensor([float(epoch),float(epoch+2)]),"counter":torch.tensor(2)})
 assert w.epochs==[2,3,4]
 torch.testing.assert_close(w.average()["w"],torch.tensor([3.,5.]))
 snapshot=w.snapshot();snapshot[0][1]["w"].fill_(99)
 torch.testing.assert_close(w.average()["w"],torch.tensor([3.,5.]))
 with pytest.raises(ValueError):w.append(4,{"w":torch.ones(2),"counter":torch.tensor(2)})

def test_window_rejects_nonfinite_and_changed_state_identity():
 torch=pytest.importorskip("torch");m=importlib.import_module("bf_tap_r2.tabm_swa_window");w=m.EpochWindow(10)
 with pytest.raises(ValueError):w.average()
 with pytest.raises(ValueError):w.append(1,{"w":torch.tensor([float("nan")])})
 w.append(1,{"w":torch.ones(2)})
 with pytest.raises(ValueError):w.append(2,{"w":torch.ones(3)})

def test_uniform_window_matches_independent_float64_arithmetic():
 torch=pytest.importorskip("torch");m=importlib.import_module("bf_tap_r2.tabm_swa_window");w=m.EpochWindow(10)
 for epoch in range(1,15):w.append(epoch,{"w":torch.tensor([epoch*.1,(-1.)**epoch])})
 expected=torch.stack([torch.tensor([epoch*.1,(-1.)**epoch],dtype=torch.float64) for epoch in range(5,15)]).mean(0).float()
 torch.testing.assert_close(w.average()["w"],expected,atol=1e-7,rtol=1e-6)
