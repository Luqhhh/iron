"""Audited current-Q75 import boundary; absent binding forbids formal admission."""
from pathlib import Path
import json
import numpy as np
from .tabm_metric_reference import load as load_native
from .tabm_swa_protocol import sha,verify_tree,digest,child,SEEDS

BINDING="local/tabm-swa-20261001/q75-reference-binding-r1.json"
def load(root,ledger_root):
 root=Path(root).resolve();binding=root/BINDING
 if not binding.exists():raise FileNotFoundError("Current Q75 time OOF binding missing; no formal admission")
 b=json.loads(binding.read_text())
 if b.get("status")!="passed_independent_import" or b.get("candidate")!="EMA_TIME_Q75" or b.get("verified_split_seeds")!=list(SEEDS) or b.get("new_fits")!=0:raise ValueError("Q75 import has no complete audited identity")
 # Existing anchored loader is limited to the official round2 snapshot; zero reference fits.
 frame,folds,iron,de3,native=load_native(root,ledger_root)
 if b["data_digest"]!=native["data_digest"] or b["row_ids_digest"]!=native["row_ids_digest"] or b["fold_digests"]!=native["fold_digests"]:raise ValueError("Q75 data/row/fold identity differs")
 files=b["frozen_files"]
 if not files or any(not p.startswith("local/") for p in files):raise ValueError("Imported reference dependencies must be private workspace files")
 verify_tree(root,files)
 evidence=b["original_evidence"]
 for key in ("manifest","audit"):
  if evidence[key] not in files:raise ValueError("Original reference evidence absent")
 audit=json.loads(child(root,evidence["audit"]).read_text())
 # Original heterogeneous audit stays byte-anchored; import must explicitly record its validated claims.
 if b["source_audit_claims"]!={"candidate_identity_verified":True,"four_seed_complete_coverage":True,"zero_new_fits_on_import":True,"source_data_partition_and_prediction_hashes_verified":True}:raise ValueError("Incomplete native-source import claims")
 if not isinstance(audit,dict) or audit.get("status")!="passed":raise ValueError("Original audit empty")
 parents={}
 for seed in SEEDS:
  col=b["columns"][str(seed)]
  paths=[col[k] for k in ("predictions","ids","folds")]
  if any(p not in files for p in paths):raise ValueError("Imported column missing frozen bytes")
  v=np.load(child(root,col["predictions"]),allow_pickle=False);ids=np.load(child(root,col["ids"]),allow_pickle=False);fv=np.load(child(root,col["folds"]),allow_pickle=False)
  if v.shape!=(2754,) or not np.isfinite(v).all() or (v<0).any():raise ValueError("Invalid complete Q75 time column")
  np.testing.assert_array_equal(ids,frame.sample_id.astype(str).to_numpy());np.testing.assert_array_equal(fv,folds[seed])
  if digest(fv.tolist())!=b["fold_digests"][str(seed)]:raise ValueError("Q75 split digest differs")
  parents[seed]=v
 current=dict(native);current.update(current_platform="EMA_TIME_Q75",current_time_reference_complete=True,reference_fits=0,
  q75_binding_sha256=sha(binding),frozen_files={**native["frozen_files"],BINDING:sha(binding),**files})
 return frame,folds,parents,de3,current

def controls(root):
 root=Path(root);p=root/"local/tabm-swa-20261001/control-catalogue-r1.json";c=json.loads(p.read_text())
 if c["status"]!="passed" or c["cold_models"]!=22 or c["new_fits"] or c["new_optimizer_runs"] or c["ledger_before"]!=c["ledger_after"]:raise ValueError("BASE catalogue incomplete")
 verify_tree(root,c["frozen_files"])
 return c,{**c["frozen_files"],str(p.relative_to(root)):sha(p)}
