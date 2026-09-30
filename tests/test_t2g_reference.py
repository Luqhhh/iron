"""Reference integrity checks without official labels or estimator starts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from bf_tap_r2 import t2g_reference as ref
from bf_tap_r2.t2g_model import file_hash
from bf_tap_r2.v7_periodic import digest

def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value))
    return file_hash(p)

@pytest.fixture
def transfer(tmp_path,monkeypatch):
    native=tmp_path/"native"; overlay=tmp_path/"overlay"
    native.mkdir(); overlay.mkdir()
    (native/"source.py").write_text("frozen source")
    (native/"data.csv").write_text("artificial dependency only")
    (overlay/"payload.bin").write_bytes(b"immutable")
    frame=pd.DataFrame({"sample_id":[f"SYNTH_{i}" for i in range(2754)]})
    folds={s:np.arange(2754)%5 for s in ref.SEEDS}
    parent={s:{"tap_iron":np.full(2754,100.),"tap_time_len":np.full(2754,25.)} for s in ref.SEEDS}
    native_audit={"data_digest":"artificial","hashes":{"source.py":file_hash(native/"source.py"),"data.csv":file_hash(native/"data.csv")}}
    reference={"row_ids_digest":digest(frame.sample_id.tolist()),"native_data_digest":"artificial",
        "fold_digests":{str(s):digest(folds[s].tolist()) for s in ref.SEEDS},
        "native_audit":native_audit,"parent_digests":{str(s):{t:ref.array_hash(p) for t,p in parent[s].items()} for s in ref.SEEDS}}
    original="/original/iron"
    m={"reference_root":original,"development_source_root":original,"development_cache":original+"/local/dev",
        "source_hashes":{"source.py":file_hash(native/"source.py")},"runtime":ref.runtime(),
        "development_evidence":{"hashes":{}},"reference":reference}
    mh=put(overlay/"manifest.json",m)
    counts={"started":{"estimator":20,"optimizer":40},"completed":{"estimator":20,"optimizer":40},
        "failed":{"estimator":0,"optimizer":0},"incomplete":{"estimator":0,"optimizer":0}}
    rh=put(overlay/"run.finished.json",{"manifest_sha256":mh,"counts":counts,"anchors":{}})
    columns={}
    for s in ref.SEEDS:
        p=overlay/f"iron-seed-{s}.npy"; np.save(p,np.full(2754,101.))
        columns[str(s)]={"path":p.name,"sha256":file_hash(p)}
    audit={"status":"passed","candidate":"DE3_IRON_USER_REQUESTED","verified_split_seeds":list(ref.SEEDS),
        "new_audit_fits":0,"full_batch_difference":0.,"unchanged_parent_time":True,"source_hashes_unchanged":True,
        "ledger_counts_before":counts,"ledger_counts_after":counts,
        "artifact_hashes":{p.name:file_hash(p) for p in overlay.iterdir() if p.is_file()}}
    ah=put(overlay/"audit.json",audit)
    complete={"candidate":"DE3_IRON_USER_REQUESTED","parent":"V32_TIME_A60V7_50",
        "endpoint":ref.ENDPOINT,"training_seeds":[42,104729,130363],"iron_columns":columns,
        "audit_sha256":ah,"manifest_sha256":mh,"run_sha256":rh,**{k:reference[k] for k in ("row_ids_digest","native_data_digest","fold_digests")}}
    ch=put(overlay/"complete.json",complete)
    monkeypatch.setattr(ref,"COMPLETE_SHA",ch); monkeypatch.setattr(ref,"AUDIT_SHA",ah)
    monkeypatch.setattr(ref,"native_loader",lambda root:(frame,folds,parent,parent,native_audit))
    return native,overlay

def test_complete_transfer_identity(transfer):
    b=ref.verify_reference(*transfer)
    assert set(b.current)=={42,3407,7777,12011}
    assert (b.current[42]["tap_iron"]==101).all()
    assert (b.current[42]["tap_time_len"]==25).all()
    assert b.identity["reference_fits"]==0

@pytest.mark.parametrize("location",["payload.bin","iron-seed-42.npy","manifest.json","complete.json"])
def test_substituted_overlay_dependency_rejected(transfer,location):
    native,overlay=transfer
    (overlay/location).write_bytes(b"substitution")
    with pytest.raises(ValueError): ref.verify_reference(native,overlay)

def test_stale_native_source_rejected(transfer):
    native,overlay=transfer
    (native/"source.py").write_text("different source")
    with pytest.raises(ValueError): ref.verify_reference(native,overlay)

def test_reordered_ids_rejected(transfer,monkeypatch):
    native,overlay=transfer
    original=ref.native_loader
    def shuffled(root):
        f,folds,parent,historical,a=original(root)
        return f.iloc[::-1],folds,parent,historical,a
    monkeypatch.setattr(ref,"native_loader",shuffled)
    with pytest.raises(ValueError): ref.verify_reference(native,overlay)

def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError): ref.verify_hash_tree(tmp_path,{"../foreign":"a"*64})

def test_absent_overlay_is_missing_transfer(tmp_path):
    with pytest.raises(FileNotFoundError):
        ref.verify_reference(tmp_path/"native",tmp_path/"overlay")
