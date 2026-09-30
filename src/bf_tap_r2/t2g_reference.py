"""Zero-fit DE3 reference transfer verification.
Overlay schema/arithmetic provenance: Luqhhh/iron 6761240 incumbent_reference_phase.
The completed overlay's externally pinned audit is reused; no reference refits.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys
import numpy as np
from .t2g_model import file_hash
from .v7_periodic import digest
from .v46_cache import load_reference_cache as native_loader

SEEDS=(42,3407,7777,12011)
COMPLETE_SHA="492c21150764c08e9714ab5321badeb2827459e23f9480126155f830021c2ab4"
AUDIT_SHA="ed03cd99e17c6514053eb62494a00cbe40ad7d6600a83de991a98c9b801b9f06"
ENDPOINT="maximum(parent_iron+0.5*(mean(J42,J104729,J130363)-J42),0)"

@dataclass
class ReferenceBundle:
    frame: object
    folds: dict
    current: dict
    historical: dict
    identity: dict

def array_hash(a):
    return hashlib.sha256(np.asarray(a,float).tobytes()).hexdigest()

def runtime():
    packages={}
    for n in ("torch","tabm","rtdl-num-embeddings","numpy","pandas","scikit-learn","scipy","PyYAML","psutil"):
        try: packages[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: packages[n]=None
    return {"python":sys.version,"packages":packages}

def child(root,name):
    root=Path(root).resolve(); rel=Path(name)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("Reference path escapes root")
    p=root/rel
    if not p.resolve().is_relative_to(root): raise ValueError("Reference symlink escapes root")
    return p

def verify_hash_tree(root,hashes):
    if not isinstance(hashes,dict): raise ValueError("Reference hash table required")
    for name,sha in hashes.items():
        p=child(root,name)
        if not p.is_file() or file_hash(p)!=sha: raise ValueError("Reference dependency changed: "+name)

def anchored(path,sha):
    if not Path(path).is_file(): raise FileNotFoundError("Missing private reference: "+str(path))
    if file_hash(path)!=sha: raise ValueError("External reference anchor mismatch")
    return json.loads(Path(path).read_text())

def mapped_relative(path,original_root):
    try: return Path(path).relative_to(original_root).as_posix()
    except ValueError as e: raise ValueError("Transferred dependency lacks original relative identity") from e

def verify_reference(native_root,overlay_root):
    native_root,overlay_root=Path(native_root),Path(overlay_root)
    c=anchored(overlay_root/"complete.json",COMPLETE_SHA)
    a=anchored(overlay_root/"audit.json",AUDIT_SHA)
    if (c["candidate"]!="DE3_IRON_USER_REQUESTED" or c["parent"]!="V32_TIME_A60V7_50"
        or c["endpoint"]!=ENDPOINT or c["training_seeds"]!=[42,104729,130363]
        or c["audit_sha256"]!=AUDIT_SHA or set(c["iron_columns"])!={str(s) for s in SEEDS}
        or a["status"]!="passed" or a["candidate"]!=c["candidate"]
        or a["verified_split_seeds"]!=list(SEEDS) or a["new_audit_fits"]!=0
        or a["full_batch_difference"]!=0 or a["unchanged_parent_time"] is not True
        or a["source_hashes_unchanged"] is not True):
        raise ValueError("Incomplete current-reference audit identity")
    counts=a["ledger_counts_before"]
    limits={"estimator":20,"optimizer":40}
    if (a["ledger_counts_after"]!=counts or counts["started"]!=limits or counts["completed"]!=limits
        or any(counts["failed"].values()) or any(counts["incomplete"].values())):
        raise ValueError("Reference execution ledger incomplete")
    verify_hash_tree(overlay_root,a["artifact_hashes"])
    m=anchored(overlay_root/"manifest.json",c["manifest_sha256"])
    run=anchored(overlay_root/"run.finished.json",c["run_sha256"])
    if run["manifest_sha256"]!=c["manifest_sha256"] or run["counts"]!=counts:
        raise ValueError("Reference run manifest/ledger changed")
    historical_runtime=m["runtime"]
    now=runtime()
    if (historical_runtime["packages"]!=now["packages"]
        or not historical_runtime["python"].startswith("3.12.")
        or sys.version_info[:2]!=(3,12)):
        raise ValueError("Native reference runtime differs")
    # Never execute transferred source; verify its byte identities as completed evidence.
    verify_hash_tree(native_root,m["source_hashes"])
    verify_hash_tree(native_root,m["reference"]["native_audit"]["hashes"])
    original=m["reference_root"]
    if m["development_source_root"]!=original:
        raise ValueError("Multiple historical source roots require explicit transfer mapping")
    development=child(native_root,mapped_relative(m["development_cache"],original))
    verify_hash_tree(development,m["development_evidence"]["hashes"])
    f,folds,parent,historical,native_audit=native_loader(native_root)
    r=m["reference"]
    if (len(f)!=2754 or f.sample_id.isna().any() or f.sample_id.duplicated().any()
        or digest(f.sample_id.tolist())!=r["row_ids_digest"] or c["row_ids_digest"]!=r["row_ids_digest"]
        or native_audit["data_digest"]!=r["native_data_digest"] or c["native_data_digest"]!=r["native_data_digest"]
        or native_audit["hashes"]!=r["native_audit"]["hashes"]
        or set(folds)!=set(SEEDS) or set(parent)!=set(SEEDS)):
        raise ValueError("Native row/data/hash identity differs")
    current={}
    for s in SEEDS:
        fv=np.asarray(folds[s])
        if fv.shape!=(2754,) or set(fv.tolist())!=set(range(5)):
            raise ValueError("Reference five-fold coverage incomplete")
        if digest(fv.tolist())!=r["fold_digests"][str(s)] or c["fold_digests"]!=r["fold_digests"]:
            raise ValueError("Native fit/query partition identity differs")
        if set(parent[s])!={"tap_iron","tap_time_len"}: raise ValueError("Reference parent targets differ")
        for t,p in parent[s].items():
            if np.asarray(p).shape!=(2754,) or not np.isfinite(p).all() or array_hash(p)!=r["parent_digests"][str(s)][t]:
                raise ValueError("Native parent column identity differs")
        column=c["iron_columns"][str(s)]
        verify_hash_tree(overlay_root,{column["path"]:column["sha256"]})
        iron=np.load(child(overlay_root,column["path"]),allow_pickle=False)
        if iron.shape!=(2754,) or not np.isfinite(iron).all(): raise ValueError("Invalid DE3 column")
        current[s]={"tap_iron":iron.copy(),"tap_time_len":np.asarray(parent[s]["tap_time_len"]).copy()}
    return ReferenceBundle(f,folds,current,historical,{"candidate":c["candidate"],
        "complete_sha256":COMPLETE_SHA,"audit_sha256":AUDIT_SHA,"manifest_sha256":c["manifest_sha256"],
        "native_root":str(native_root.resolve()),"overlay_root":str(overlay_root.resolve()),
        "native_identity":r,"runtime":historical_runtime,"source_provenance":"6761240",
        "reference_fits":0,"audit_reuse":"externally_anchored_completed_zero_fit_audit"})
