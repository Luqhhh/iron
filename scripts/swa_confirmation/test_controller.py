import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
P=Path(__file__).with_name("controller.py")
spec=importlib.util.spec_from_file_location("swa_continuation_controller",P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def frozen():return json.loads((ROOT/"configs/tabm_time_swa_continuation/CONFIRMATION_SPEC.json").read_text())
def zero():return {"reserved":{"state":0,"optimizer":0},"completed":{"state":0,"optimizer":0}}
def test_registered_stage_accepts_only_replacement_budget():m.validate_stage(frozen(),zero())
def test_old_confirmation_consumption_rejected():
 c=zero();c["reserved"]["optimizer"]=1
 with pytest.raises(ValueError,match="consumed"):m.validate_stage(frozen(),c)
def test_changed_seed_or_weight_rejected():
 for k,v in [("confirmation_seeds",[7777,12011]),("alpha",.3)]:
  s=frozen();s[k]=v
  with pytest.raises(ValueError):m.validate_stage(s,zero())
def test_score_and_four_seed_gate_are_separate():
 import numpy as np
 y=np.array([10.,20.]);parent=np.array([11.,22.]);member=np.array([10.,20.])
 assert m.gain(y,parent,member)==pytest.approx(1.)
