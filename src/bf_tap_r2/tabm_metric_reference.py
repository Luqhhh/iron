"""Byte-anchored reuse of previous completed four-seed reference audit. Zero refits."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from .tabm_metric_protocol import sha,digest,verify_tree,child,SEEDS
from .v5_library import load_v5_training_frame,fold_vector
COMPLETE_SHA="492c21150764c08e9714ab5321badeb2827459e23f9480126155f830021c2ab4"
AUDIT_SHA="ed03cd99e17c6514053eb62494a00cbe40ad7d6600a83de991a98c9b801b9f06"
MANIFEST_SHA="fba318a03e76c7e2adc31bebaf376ab7b2550178b916cb3ba72a3c8924ca8a8c"
POLICY_SHA="4bc88981583c0307de485bb422a38cb760ae0470fb5114a309dce1b1ade7e638"

def load(root,ledger_root):
    root=Path(root).resolve(); private=root/"local"
    mapping=root/"local/amf-handoff-20261001/reference-transfer-r1.json"
    t=json.loads(mapping.read_text()); overlay=Path(t["overlay_root"]).resolve()
    for key in ("overlay_root","reference_root","source_root","development_source_root"):
        if not Path(t[key]).resolve().is_relative_to(private): raise ValueError("Transferred path outside local")
    for filename,h in (("complete.json",COMPLETE_SHA),("audit.json",AUDIT_SHA),("manifest.json",MANIFEST_SHA)):
        if sha(overlay/filename)!=h: raise ValueError("External reference anchor changed")
    c=json.loads((overlay/"complete.json").read_text()); a=json.loads((overlay/"audit.json").read_text()); m=json.loads((overlay/"manifest.json").read_text())
    if a["status"]!="passed" or a["verified_split_seeds"]!=list(SEEDS) or a["new_audit_fits"]!=0 or not a["source_hashes_unchanged"]: raise ValueError("Incomplete reference audit")
    if c["parent"]!="V32_TIME_A60V7_50" or c["candidate"]!="DE3_IRON_USER_REQUESTED" or c["audit_sha256"]!=AUDIT_SHA or c["manifest_sha256"]!=MANIFEST_SHA: raise ValueError("Reference semantics changed")
    if a["ledger_counts_before"]!=a["ledger_counts_after"] or a["ledger_counts_after"]["started"]!={"estimator":20,"optimizer":40} or a["ledger_counts_after"]["completed"]!={"estimator":20,"optimizer":40}: raise ValueError("Reference counts incomplete")
    for k,original in (("original_reference_root","reference_root"),("original_workspace","workspace"),("original_development_source_root","development_source_root")):
        if t[k]!=m[original]: raise ValueError("Transfer original identity changed")
    verify_tree(overlay,a["artifact_hashes"])
    verify_tree(t["source_root"],m["source_hashes"])
    verify_tree(t["reference_root"],m["reference"]["native_audit"]["hashes"])
    rel=Path(m["development_cache"]).relative_to(m["reference_root"])
    development=child(t["reference_root"],rel)
    verify_tree(development,m["development_evidence"]["hashes"])
    dm=json.loads((development/"manifest.json").read_text()); verify_tree(t["development_source_root"],dm["source_hashes"])
    # Main official snapshot is the exact snapshot bound by the old completed audit.
    data_files={}; newline_variants=[]
    for name,h in m["reference"]["native_audit"]["hashes"].items():
        if name.startswith("复赛_train/"):
            if sha(root/name)!=h:
                transferred=child(t["reference_root"],name)
                if sha(transferred)!=h or (root/name).read_bytes().replace(b"\r\n",b"\n")!=transferred.read_bytes().replace(b"\r\n",b"\n"): raise ValueError("Official snapshot differs beyond line endings")
                newline_variants.append(dict(path=name,original_sha256=h,working_sha256=sha(root/name),reason="CRLF_LF_only_exact_normalized_bytes"))
            data_files[name]=sha(root/name)
    if not data_files or sha(root/"configs/protection.yaml")!=POLICY_SHA: raise ValueError("Data protection identity missing")
    from .tabm_metric_protocol import Ledger
    ledger=Ledger(ledger_root)
    with ledger.locked():
        import os
        with (Path(ledger_root)/"label-access.jsonl").open("a") as f:
            f.write(json.dumps(dict(time=time.time(),event="before_round2_snapshot_label_read",protection_sha256=POLICY_SHA,manifest_sha256=MANIFEST_SHA,
                lifecycle="development",protected_labels_requested=False,reference_fits=0),sort_keys=True)+"\n");f.flush();os.fsync(f.fileno())
    frame=load_v5_training_frame(root); r=m["reference"]
    data_sha=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    if len(frame)!=2754 or data_sha!=r["native_data_digest"] or digest(frame.sample_id.tolist())!=r["row_ids_digest"]: raise ValueError("Official row/data identity differs")
    receipt_path=root/"local/next-direction-20261001/native-iron-reference-r1.json"; receipt=json.loads(receipt_path.read_text())
    folds={}; parents={}; de3={}; frozen={str(mapping.relative_to(root)):sha(mapping),str(receipt_path.relative_to(root)):sha(receipt_path),**data_files}
    frozen["configs/protection.yaml"]=POLICY_SHA
    for filename,h in (("complete.json",COMPLETE_SHA),("audit.json",AUDIT_SHA),("manifest.json",MANIFEST_SHA)):
        frozen[str((overlay/filename).relative_to(root))]=h
    for seed in SEEDS:
        fv=fold_vector(root,frame,seed); col=receipt["columns"][str(seed)]; folds[seed]=fv
        if digest(fv.tolist())!=r["fold_digests"][str(seed)] or col["fold_digest"]!=digest(fv.tolist()): raise ValueError("Fold identity differs")
        p=root/f"local/next-direction-20261001/parent-iron-seed-{seed}.npy"
        if sha(p)!=col["file_sha256"]: raise ValueError("Native iron bytes changed")
        v=np.load(p,allow_pickle=False)
        if v.shape!=(2754,) or not np.isfinite(v).all() or hashlib.sha256(np.asarray(v,float).tobytes()).hexdigest()!=r["parent_digests"][str(seed)]["tap_iron"]: raise ValueError("Native iron parent identity differs")
        parents[seed]=v; frozen[str(p.relative_to(root))]=sha(p)
        entry=c["iron_columns"][str(seed)]; p=child(overlay,entry["path"])
        if sha(p)!=entry["sha256"]: raise ValueError("DE3 auxiliary bytes changed")
        de3[seed]=np.load(p,allow_pickle=False); frozen[str(p.relative_to(root))]=sha(p)
    return frame,folds,parents,de3,dict(frozen_files=frozen,data_digest=data_sha,fold_digests=r["fold_digests"],row_ids_digest=r["row_ids_digest"],
        csv_newline_variants=newline_variants,reference_fits=0,current_platform="EMA_TIME_Q75",platform_score_user_reported=96.392,
        historical_B0_iron_equals_current=True,complete_Q75_local_package_score_available=False,
        audit_reuse="externally_anchored_completed_audit_plus_byte_partition_identity_revalidation")
