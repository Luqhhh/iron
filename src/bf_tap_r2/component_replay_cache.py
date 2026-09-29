"""Reuse verified completed native BASE units, never a partially written fit."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time
import yaml

import numpy as np

from .v7_periodic import digest,file_hash,write_new
from .v49_run import verified_unit,unit_id,append_event,verify_hashes
from .data import TARGETS


def verify_original_sources(root,hashes,commit):
    for name,sha in hashes.items():
        path=root/name
        if path.is_file() and file_hash(path)==sha:continue
        original=subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=root)
        if hashlib.sha256(original).hexdigest()!=sha:raise ValueError(f"Original BASE source identity mismatch: {name}")


def reuse_baselines(root,out,frame,folds,spec,manifest):
    cache=root/spec["base_replay_cache"]["directory"]
    original=json.loads((cache/"manifest.json").read_text())
    expected=dict(original);expected.pop("identity")
    if digest(expected)!=original["identity"]:raise ValueError("BASE manifest identity mismatch")
    verify_original_sources(root,original["source_hashes"],spec["base_replay_cache"]["expected_source_commit"])
    verify_hashes(root,original["data_hashes"])
    source_spec=spec["base_replay_cache"]["source_spec"]
    if file_hash(root/source_spec)!=original["source_hashes"][source_spec]:
        raise ValueError("BASE source specification mismatch")
    settings=yaml.safe_load((root/source_spec).read_text())["training"]
    if settings!=spec["training"]:raise ValueError("BASE training recipe mismatch")
    if original["versions"]!=manifest["versions"] or original["fold_hashes"]!=manifest["fold_hashes"]:
        raise ValueError("BASE runtime/fold mismatch")
    keys=[f"{t}-BASE-s{s}-f{f}" for s in folds for f in range(5) for t in TARGETS]
    # A bounded-frequency dependency observation, not a training restart or
    # an inner-metric poll. At most one observation per600s while waiting.
    while not all((cache/k/"complete.json").exists() for k in keys):
        terminal=cache.parent/"completion-event.json"
        if terminal.exists() and json.loads(terminal.read_text())["status"]=="failed":
            raise ValueError("Upstream workflow failed before BASE admission")
        failed=[k for k in keys if (cache/k/"replay_failure.json").exists()]
        ledger=cache/"fit_ledger.jsonl"
        if ledger.exists():
            failed += [r.get("unit",r.get("key")) for r in map(json.loads,ledger.read_text().splitlines()) if r.get("event")=="failed"]
        if failed:raise ValueError(f"Upstream BASE failures preserved: {failed}")
        append_event(out/"dependency-observations.jsonl",{"status":"waiting_for_complete_native_BASE_units",
            "complete":sum((cache/k/"complete.json").exists() for k in keys),"required":len(keys)})
        time.sleep(spec["base_replay_cache"]["wait_interval_seconds"])
    for seed,fv in folds.items():
        for fold in range(5):
            training=frame.loc[fv!=fold].reset_index(drop=True);query_ids=frame.loc[fv==fold,"sample_id"].to_numpy(dtype=str)
            with np.load(out/f"reference-s{seed}-f{fold}/predictions.npz") as refs:
                for target in TARGETS:
                    key=f"{target}-BASE-s{seed}-f{fold}";source=cache/key
                    result=verified_unit(source,unit_id(original,key))
                    if not result:raise ValueError("Missing BASE unit")
                    if not {"metadata.json","predictions.npz","selection.pt","refit.pt"}.issubset(result["hashes"]):
                        raise ValueError("Incomplete BASE artifact set")
                    metadata=json.loads((source/"metadata.json").read_text())
                    if metadata["fit_ids_digest"]!=digest(training.sample_id.tolist()) or metadata["native_replay_max_difference"]!=0:
                        raise ValueError("BASE fitting identity or original replay mismatch")
                    if (metadata["target"],metadata["arm"],metadata["seed"],metadata["fold"],metadata["training_seed"])!=(target,"BASE",seed,fold,42):
                        raise ValueError("BASE semantic identity mismatch")
                    with np.load(source/"predictions.npz") as saved:
                        np.testing.assert_array_equal(saved["query_ids"],query_ids)
                        np.testing.assert_array_equal(saved["prediction"][:,0],refs["v12_iron" if target=="tap_iron" else "v7_time"])
                    directory=out/key;directory.mkdir(exist_ok=False)
                    for name in result["hashes"]:shutil.copyfile(source/name,directory/name)
                    write_new(directory/"provenance.json",{"source_unit":str(source),"source_identity":result["identity"],
                        "source_manifest_sha256":file_hash(cache/"manifest.json"),"new_fits":0})
                    write_new(directory/"complete.json",{"identity":unit_id(manifest,key),
                        "hashes":{p.name:file_hash(p) for p in directory.iterdir()},"key":key,"reused":True})
    append_event(out/"fit_ledger.jsonl",{"event":"native_base_cache_reused","units":len(keys),"new_fits":0})
