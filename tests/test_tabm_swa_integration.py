import importlib
from pathlib import Path
import pytest

def test_missing_current_time_reference_fails_before_directory_or_label_access(tmp_path):
 m=importlib.import_module("bf_tap_r2.tabm_swa_run")
 target=tmp_path/"local/runs/not-created"
 with pytest.raises(FileNotFoundError,match="Q75 time reference"):m.preflight(tmp_path,target)
 assert not target.exists()

def test_new_fit_budgets_exclude_reused_controls():
 m=importlib.import_module("bf_tap_r2.tabm_swa_protocol")
 assert m.BUDGETS=={"engineering":{"state":2,"optimizer":2},"development":{"state":20,"optimizer":20},"confirmation":{"state":40,"optimizer":40}}
 assert m.decision([.01,.02],[.009,.019],confirmation=False)["selected_for_confirmation"]==["SWA_WINDOW10"]

def test_cold_witness_recalculation_rejects_wrong_epoch_or_average():
 torch=pytest.importorskip("torch");m=importlib.import_module("bf_tap_r2.tabm_swa_audit")
 states=[{"w":torch.tensor([float(i),i+.1])} for i in range(3,13)]
 witness={"window":10,"selected_epoch":12,"epochs":list(range(3,13)),"states":states}
 payload={"trace":{"selected_epoch":12},"state":{"w":torch.stack([s["w"] for s in states]).mean(0)}}
 assert m.verify_window(payload,witness)<5e-7
 witness["epochs"][0]=2
 with pytest.raises(ValueError):m.verify_window(payload,witness)
 witness["epochs"][0]=3;payload["state"]["w"]+=.01
 with pytest.raises(AssertionError):m.verify_window(payload,witness)


def _two_seed_binding(root):
 import json
 from bf_tap_r2.tabm_swa_protocol import sha
 prefix="local/imported-test"
 directory=root/prefix;directory.mkdir(parents=True)
 manifest=directory/"manifest.json";report=directory/"report.json";audit=directory/"audit.json"
 manifest.write_text("{}")
 report.write_text("{}")
 audit.write_text(json.dumps({"G0":"passed","manifest_sha256":sha(manifest),"report_sha256":sha(report)}))
 return {"status":"passed_independent_import_available_seeds","candidate":"EMA_TIME_Q75","verified_split_seeds":[42,3407],"missing_split_seeds":[7777,12011],"new_fits":0,
  "frozen_files":{str(p.relative_to(root)):sha(p) for p in [manifest,report,audit]},
  "original_evidence":{"manifest":str(manifest.relative_to(root)),"report":str(report.relative_to(root)),"audit":str(audit.relative_to(root))},
  "source_audit_claims":{"candidate_identity_verified":True,"available_seed_complete_coverage":True,"zero_new_fits_on_import":True,"source_data_partition_and_prediction_hashes_verified":True}}

def test_two_seed_admission_is_honest_and_does_not_imply_confirmation(tmp_path):
 m=importlib.import_module("bf_tap_r2.tabm_swa_reference")
 b=_two_seed_binding(tmp_path)
 assert m.validate_import(tmp_path,b)==(42,3407)
 with pytest.raises(ValueError,match="required"):
  m.validate_import(tmp_path,b,required_seeds=(42,3407,7777,12011))
 b["missing_split_seeds"]=[]
 with pytest.raises(ValueError,match="coverage"):
  m.validate_import(tmp_path,b)

def test_import_refuses_original_audit_identity_tampering(tmp_path):
 import json
 m=importlib.import_module("bf_tap_r2.tabm_swa_reference")
 from bf_tap_r2.tabm_swa_protocol import sha
 b=_two_seed_binding(tmp_path);p=tmp_path/b["original_evidence"]["audit"]
 a=json.loads(p.read_text());a["report_sha256"]="0"*64;p.write_text(json.dumps(a));b["frozen_files"][str(p.relative_to(tmp_path))]=sha(p)
 with pytest.raises(ValueError,match="audit"):
  m.validate_import(tmp_path,b)

def test_qualified_development_waits_without_reserving_confirmation(tmp_path,monkeypatch):
 import json
 m=importlib.import_module("bf_tap_r2.tabm_swa_run")
 r=importlib.import_module("bf_tap_r2.tabm_swa_reference")
 from bf_tap_r2.tabm_swa_protocol import sha
 run=tmp_path/"local/runs/tabm-time-swa-v1";engineering=run/"engineering-r1";engineering.mkdir(parents=True)
 frozen={"reference":{}}
 for filename,key in [("audit.json","engineering_audit_sha256"),("finished.json","engineering_finished_sha256"),("inputs.npz","engineering_inputs_sha256")]:
  p=engineering/filename;p.write_bytes(b"test");frozen[key]=sha(p)
 (run/"preflight.json").write_text(json.dumps(frozen))
 monkeypatch.setattr(m,"verify_frozen",lambda *args:None)
 monkeypatch.setattr(r,"load",lambda *args:(None,None,{42:[],3407:[]},None,{}))
 monkeypatch.setattr(m,"partitions",lambda frame,folds,seeds:seeds)
 called=[]
 def phase(root,runroot,name,frame,units,frozen):
  called.append((name,units));p=runroot/(name+"-r1");p.mkdir();(p/"evaluation.json").write_text("{}");return p
 monkeypatch.setattr(m,"run_phase",phase)
 monkeypatch.setattr(m,"evaluate",lambda *args,**kw:{"selected_for_confirmation":["SWA_WINDOW10"],"formal_promoted":False})
 m.execute(tmp_path,run)
 assert called==[("development",[42,3407])]
 finished=json.loads((run/"controller-finished.json").read_text())
 assert finished["status"]=="completed_development_waiting_confirmation_reference"
 assert finished["phase_counts"]["confirmation"]["reserved"]=={"state":0,"optimizer":0}
 assert not (run/"confirmation-r1").exists()


def test_actual_cold_audit_entry_initializes_repository_root(tmp_path,monkeypatch):
 torch=pytest.importorskip("torch")
 import json,numpy as np
 m=importlib.import_module("bf_tap_r2.tabm_swa_audit")
 directory=tmp_path/"local/runs/unit/engineering-r1";directory.mkdir(parents=True)
 (directory/"manifest.json").write_text(json.dumps({"source_hashes":{}}))
 np.savez(directory/"inputs.npz",x=np.zeros((0,21)))
 called=[]
 def verified(root,hashes):
  called.append(root);raise RuntimeError("snapshot verification reached")
 monkeypatch.setattr(m,"verify_tree",verified)
 with pytest.raises(RuntimeError,match="snapshot verification reached"):
  m.audit(directory)
 assert called==[tmp_path]

def test_recovery_source_bridge_rejects_unapproved_or_changed_history(tmp_path):
 pytest.importorskip("torch")
 import json
 m=importlib.import_module("bf_tap_r2.tabm_swa_audit")
 from bf_tap_r2.tabm_swa_protocol import sha
 rel="src/bf_tap_r2/tabm_swa_audit.py";actual=tmp_path/rel;actual.parent.mkdir(parents=True);actual.write_text("new auditor")
 old=tmp_path/"local/old-auditor.py";old.parent.mkdir();old.write_text("old auditor")
 manifest={"source_hashes":{rel:sha(old)}};mp=tmp_path/"manifest.json";mp.write_text(json.dumps(manifest))
 bridge={"status":"user_authorized_zero_fit_auditor_repair","original_manifest_sha256":sha(mp),"changes":{rel:{"original_sha256":sha(old),"current_sha256":sha(actual),"original_archive":"local/old-auditor.py"}}}
 bp=tmp_path/"bridge.json";bp.write_text(json.dumps(bridge))
 m.verify_recovered_sources(tmp_path,manifest,mp,bp)
 old.write_text("altered old source")
 with pytest.raises(ValueError,match="archive"):
  m.verify_recovered_sources(tmp_path,manifest,mp,bp)
 old.write_text("old auditor");actual.write_text("unapproved next edit")
 with pytest.raises(ValueError,match="repair"):
  m.verify_recovered_sources(tmp_path,manifest,mp,bp)
