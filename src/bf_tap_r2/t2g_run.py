"""Frozen formal phases; no resource-probe overrides or reference fitting."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import json
import multiprocessing
import os
import numpy as np
from .data import FEATURES,TARGETS
from .t2g_model import canonical,file_hash
from .t2g_protocol import UnitKey,expected_units,initialize_ledger,reserve,outcome,inspect_ledger
from .t2g_reference import verify_reference

def write_new(path,value):
    path=Path(path)
    with path.open("x") as f:
        f.write(canonical(value)+"\n"); f.flush(); os.fsync(f.fileno())

def validate_formal_manifest(m):
    if "force_epochs" in m or m.get("kind")!="t2g-formal-v1" or m.get("identity")!="T2G_GRAPH_V1":
        raise ValueError("Formal manifest scope/override differs")
    if m.get("g0_status")!="passed" or "reference" not in m:
        raise ValueError("Passed G0 and verified current reference required")

def validate_completion(phase,targets,records):
    expected={canonical(asdict(k)) for k in expected_units(phase,targets)}
    actual=[canonical(r["key"]) for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=expected or any(r["optimizer_starts"]!=2 for r in records):
        raise ValueError("Incomplete or duplicate phase coverage")

def execute_unit(train,query,key,output,ledger_root,manifest_digest):
    from .t2g_model import fit_partition,load_predict
    output=Path(output)
    try:
        result=fit_partition(train,query,key.target,key.arm,output,
            lambda stage:reserve(ledger_root,key,stage,manifest_digest))
        if not np.array_equal(result.prediction,load_predict(output,query)):
            raise ValueError("Warm/full-batch cold mismatch")
        p=output/"predictions.npz"
        with p.open("xb") as f:
            np.savez(f,query_ids=query.sample_id.to_numpy(dtype=str),prediction=result.prediction)
        hashes={str(p.relative_to(output)):file_hash(p) for p in output.iterdir() if p.is_file()}
        record={"key":asdict(key),"optimizer_starts":2,"selected_epoch":result.selected_epoch,
            "fit_ids":train.sample_id.astype(str).tolist(),"query_ids":query.sample_id.astype(str).tolist(),
            "hashes":hashes,"manifest_digest":manifest_digest}
        write_new(output/"complete.json",record)
        outcome(ledger_root,key,"completed",{"complete_sha256":file_hash(output/"complete.json")})
        return {"directory":output.name,"complete_sha256":file_hash(output/"complete.json"),**record}
    except BaseException as e:
        if output.exists() and not (output/"failed.json").exists():
            write_new(output/"failed.json",{"type":type(e).__name__,"message":str(e),"key":asdict(key)})
        outcome(ledger_root,key,"failed",{"type":type(e).__name__,"message":str(e)})
        raise

def run_phase(manifest,phase,output,selected_targets):
    from .t2g_freeze import verify_manifest
    manifest=Path(manifest); m=verify_manifest(manifest); validate_formal_manifest(m)
    for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        if os.environ.get(n)!="1": raise ValueError("Set numerical thread variables to1")
    reference=verify_reference(m["reference"]["native_root"],m["reference"]["overlay_root"])
    if reference.identity!=m["reference"]: raise ValueError("Frozen reference differs")
    units=expected_units(phase,selected_targets)
    if phase=="confirmation":
        from .t2g_verify import decide
        development=Path(m["development"])
        d=decide(development,None,reference)
        if sorted(selected_targets)!=sorted(d["selected_for_confirmation"]):
            raise ValueError("Confirmation targets differ from audited development")
    output=Path(output)
    root=Path(__file__).resolve().parents[2]
    if not output.resolve().is_relative_to(root/"local"):
        raise ValueError("Ignored private output required")
    output.mkdir(parents=True,exist_ok=False)
    manifest_digest=file_hash(manifest)
    write_new(output/"phase.json",{"phase":phase,"selected_targets":selected_targets,
        "manifest":str(manifest.resolve()),"manifest_digest":manifest_digest,"reference":reference.identity})
    ledger=output/"ledger"; initialize_ledger(ledger,phase,selected_targets,manifest_digest)
    records=[]
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
            pending={}
            for key in units:
                mask=reference.folds[key.seed]==key.fold
                train=reference.frame.loc[~mask].reset_index(drop=True)
                query=reference.frame.loc[mask,["sample_id","spout_no",*FEATURES]].reset_index(drop=True)
                if len(train)>2204 or len(query)>551 or not len(query):
                    raise ValueError("Official partition exceeds synthetic measured bounds")
                name=f"{key.target}-{key.arm}-s{key.seed}-f{key.fold}"
                pending[pool.submit(execute_unit,train,query,key,output/name,ledger,manifest_digest)]=key
            try:
                for future in as_completed(pending): records.append(future.result())
            except BaseException:
                for future in pending: future.cancel()
                raise
        validate_completion(phase,selected_targets,records)
        counts=inspect_ledger(ledger)
        if counts["outer_fits"]!=len(units) or counts["optimizer_starts"]!=2*len(units):
            raise ValueError("Formal reservation counts differ")
        write_new(output/"complete.json",{"phase":phase,"targets":selected_targets,
            "manifest_digest":manifest_digest,"records":records,"counts":counts})
    except BaseException as e:
        write_new(output/"failed.json",{"type":type(e).__name__,"message":str(e)})
        raise
    return output/"complete.json"
