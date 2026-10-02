"""Exclusive authorized new-seed confirmation, original scientific modules unchanged."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback,importlib.util
import numpy as np
from bf_tap_r2.tabm_swa_protocol import Ledger,BUDGETS,write_new,verify_tree,sha
from bf_tap_r2.tabm_swa_run import runtime,assert_threads,partitions,save_inputs,evaluate
ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT/"local/swa-confirmation-20261002"
OLD=ROOT/"local/runs/tabm-time-swa-v1"
RUN=ROOT/"local/runs/tabm-time-swa-confirm-271828-314159-20261001"
BUNDLE=ROOT/"local/incoming/swa-confirmation-271828-314159-20261002-r1"
SPEC=ROOT/"configs/tabm_time_swa_continuation/CONFIRMATION_SPEC.json"
EXPECTED_MANIFEST="4f7f04e00f8f5d79dad26e3b66c5e5f6d5789c9993dca250ce6482393d015e9f"
ZERO={"reserved":{"state":0,"optimizer":0},"completed":{"state":0,"optimizer":0}}
def validate_stage(spec,old_counts):
 if old_counts!=ZERO:raise ValueError("Old confirmation budget consumed")
 if spec["confirmation_seeds"]!=[271828,314159] or spec["four_seed_order"]!=[42,3407,271828,314159] or spec["alpha"]!=.2:raise ValueError("Changed prospective stage")
 if spec["confirmation_budget"]["saved_selection_refit_states"]!=40 or spec["confirmation_budget"]["optimizer_initializations"]!=40 or spec["release_authorized"] is not False:raise ValueError("Changed budget/release")
def gain(y,parent,member):return float(50*(np.abs(y-parent).sum()-np.abs(y-(.8*parent+.2*member)).sum())/y.sum())
def verifier():
 s=importlib.util.spec_from_file_location("verified_swa_handoff",BUNDLE/"scripts/verify_swa_continuation_handoff.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def check(frozen):
 if subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()!="codex/tabm-time-swa-v1":raise ValueError("Branch changed")
 verify_tree(ROOT,frozen["source_hashes"]);verify_tree(ROOT,frozen["reference"]["frozen_files"])
 if runtime()!=frozen["runtime"]:raise ValueError("Runtime changed")
 if Ledger(OLD/"ledger").counts("confirmation")!=ZERO:raise ValueError("Original confirmation competed")

def data():
 from bf_tap_r2.tabm_metric_reference import load
 from bf_tap_r2.splits import make_folds
 frame,_,_,_,ref=load(ROOT,RUN/"ledger")
 m=verifier();v=m.verify_bundle(BUNDLE,EXPECTED_MANIFEST);folds={};parents={}
 for seed in (42,3407,271828,314159):
  fv=make_folds(frame,seed).set_index("sample_id").loc[frame.sample_id,"fold"].to_numpy();parents[seed],folds[seed]=m.reindex_reference(v,seed,frame.sample_id.astype(str).to_numpy(),fv)
 return frame,folds,parents,ref

def recompute_development(frame,parents):
 b=json.loads((ROOT/"local/tabm-swa-20261001/q75-reference-binding-r1.json").read_text());ev=json.loads((OLD/"development-r1/evaluation.json").read_text());g=[];c=[]
 y=frame.tap_time_len.to_numpy(float)
 for seed in (42,3407):
  np.testing.assert_array_equal(parents[seed],np.load(ROOT/b["columns"][str(seed)]["predictions"],allow_pickle=False))
  accum={a:np.full(len(frame),np.nan) for a in ("BASE","SWA_WINDOW10")};coverage=np.zeros(len(frame),int)
  for fold in range(5):
   d=OLD/"development-r1"/f"s{seed}-f{fold}"
   with np.load(d/"partitions.npz",allow_pickle=False) as a:q=a["query"]
   with np.load(d/"predictions.npz",allow_pickle=False) as a:
    np.testing.assert_array_equal(a["ids"],frame.iloc[q].sample_id.astype(str).to_numpy())
    for arm in accum:accum[arm][q]=a[arm]
   coverage[q]+=1
  if not np.all(coverage==1):raise ValueError("Old OOF incomplete")
  g.append(gain(y,parents[seed],accum["SWA_WINDOW10"]));c.append(gain(y,parents[seed],accum["BASE"]))
 np.testing.assert_allclose(g,ev["seed_gains"],rtol=0,atol=1e-12);np.testing.assert_allclose(c,ev["control_gains"],rtol=0,atol=1e-12)
 if not all(x>0 for x in g) or np.mean(np.asarray(g)-c)<=0:raise ValueError("Original development gate not satisfied")
 return dict(status="passed_independent_same_reference_development_recalculation",seed_gains=g,control_gains=c,new_fits=0)

def admit():
 spec=json.loads(SPEC.read_text());validate_stage(spec,Ledger(OLD/"ledger").counts("confirmation"))
 old=json.loads((OLD/"preflight.json").read_text());check(old)
 if RUN.exists() or list(OLD.glob("confirmation-replacement-claim-*.json")):raise ValueError("Prior new-phase claim/run exists")
 finished=json.loads((OLD/"controller-finished.json").read_text())
 if finished["status"]!="completed_development_waiting_confirmation_reference" or finished["decision"]["selected_for_confirmation"]!=["SWA_WINDOW10"]:raise ValueError("Original development not qualified")
 # The old scientific auditor reruns cold model checks with only its report destination redirected.
 cold_code="import torch; from pathlib import Path; from bf_tap_r2 import tabm_swa_audit as m; from bf_tap_r2.tabm_swa_protocol import write_new; torch.set_num_threads(1); torch.set_num_interop_threads(1); m.write_new=lambda path,value: write_new(Path('local/swa-confirmation-20261002/development-cold-admission-r1.json'),value); m.audit(Path('local/runs/tabm-time-swa-v1/development-r1'))"
 with (PRIVATE/"development-cold-admission-r1.log").open("x") as log:subprocess.run([sys.executable,"-c",cold_code],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 cold=json.loads((PRIVATE/"development-cold-admission-r1.json").read_text())
 if cold["status"]!="passed" or cold["cold_models"]!=40 or cold["new_fits"]!=0:raise ValueError("Old development cold admission failed")
 # One durable replacement claim shared with the historical ledger, never consuming old fits.
 with Ledger(OLD/"ledger").locked():
  validate_stage(spec,Ledger(OLD/"ledger").counts("confirmation"))
  write_new(OLD/"confirmation-replacement-claim-271828-314159-r1.json",dict(authorization="user approved new confirmation seeds in this conversation",run_root=str(RUN.relative_to(ROOT)),combined_confirmation_cap=40,time=time.time()))
 RUN.mkdir(parents=True,exist_ok=False)
 frame,folds,parents,ref=data();proof=recompute_development(frame,parents);write_new(PRIVATE/"original-development-admission-r1.json",proof)
 units=partitions(frame,folds,[271828,314159]);input_path=RUN/"admission-inputs.npz";save_inputs(input_path,frame,units)
 sources=dict(old["source_hashes"])
 for path in [SPEC,PRIVATE/"controller.py",PRIVATE/"test_controller.py",PRIVATE/"monitor_once.py",PRIVATE/"tests-complete-r1.json"]:sources[str(path.relative_to(ROOT))]=sha(path)
 deps=dict(old["reference"]["frozen_files"]);deps.update(ref["frozen_files"])
 for base in [BUNDLE,OLD/"development-r1"]:
  for path in base.rglob("*"):
   if path.is_file():deps[str(path.relative_to(ROOT))]=sha(path)
 for path in [OLD/"preflight.json",OLD/"freeze.json",OLD/"controller-finished.json",OLD/"ledger/fits.jsonl",PRIVATE/"development-cold-admission-r1.json",PRIVATE/"original-development-admission-r1.json",input_path]:deps[str(path.relative_to(ROOT))]=sha(path)
 frozen=dict(status="passed_ready_for_first_confirmation",source_hashes=sources,runtime=runtime(),reference={"frozen_files":deps},spec_sha256=sha(SPEC),reference_manifest_sha256=EXPECTED_MANIFEST,new_reference_fits=0,new_development_fits=0,new_G0_optimizers=0,confirmation_budget=BUDGETS["confirmation"],seeds=[271828,314159],four_seed_order=[42,3407,271828,314159],source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
 check(frozen);write_new(RUN/"preflight.json",frozen);print("Confirmation admitted; no new optimizer",flush=True)

def execute():
 from bf_tap_r2.tabm_swa_units import train_unit
 import yaml
 frozen=json.loads((RUN/"preflight.json").read_text());check(frozen);write_new(RUN/"controller-started.json",dict(time=time.time(),pid=os.getpid(),preflight_sha256=sha(RUN/"preflight.json")))
 try:
  frame,folds,parents,ref=data();check(frozen);units=partitions(frame,folds,[271828,314159]);directory=RUN/"confirmation-r1";directory.mkdir(exist_ok=False)
  settings=yaml.safe_load((ROOT/"configs/round2_v12/SPEC.yaml").read_text())["training"];science_spec=json.loads((ROOT/"configs/tabm_time_swa_v1/SPEC.json").read_text());ledger=Ledger(RUN/"ledger")
  if ledger.counts("confirmation")!=ZERO:raise ValueError("Confirmation phase consumed")
  save_inputs(directory/"inputs.npz",frame,units)
  if sha(directory/"inputs.npz")!=sha(RUN/"admission-inputs.npz"):raise ValueError("Admission partitions changed")
  write_new(directory/"manifest.json",dict(phase="confirmation",units=list(units),input_sha256=sha(directory/"inputs.npz"),source_hashes=frozen["source_hashes"],runtime=frozen["runtime"],budgets=BUDGETS["confirmation"],spec=science_spec,continuation_spec_sha256=sha(SPEC),settings=settings))
  write_new(directory/"started.json",dict(time=time.time(),phase="confirmation",pid=os.getpid()))
  for unit,(train,query,inner,cal) in units.items():
   check(frozen);train_unit(ROOT,frame,train,query,directory/unit,science_spec,settings,ledger,"confirmation",unit);print(unit,"complete",flush=True)
  check(frozen)
  with (directory/"cold-audit.log").open("x") as log:subprocess.run([sys.executable,"-m","bf_tap_r2.tabm_swa_audit",str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  write_new(directory/"finished.json",dict(status="passed_complete_cold_audit",audit_sha256=sha(directory/"audit.json"),counts=ledger.counts("confirmation")))
  decision=evaluate(RUN,"confirmation",frame,folds,parents,{},[271828,314159],development=OLD/"development-r1");check(frozen)
  write_new(RUN/"controller-finished.json",dict(status="completed",time=time.time(),decision=decision,confirmation_counts=ledger.counts("confirmation"),reference_fits=0,development_repeats=0,packages=0,uploads=0))
 except BaseException as e:
  write_new(RUN/"controller-failed.json",dict(status="failed_evidence_preserved",time=time.time(),error=repr(e),traceback=traceback.format_exc(),counts=Ledger(RUN/"ledger").counts("confirmation")));raise
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("action",choices=["admit","execute"]);args=parser.parse_args();assert_threads()
 if args.action=="admit":admit()
 else:execute()
