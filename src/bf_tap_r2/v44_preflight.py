"""Two full-shape synthetic fits, cold inference and resource admission."""
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import yaml

from .data import FEATURES
from .v7_periodic import file_hash, write_new
from .v44_bart import LongBartRegressor, verify_prefix
from .v43_preflight import synthetic
from .v44_prefix import verify_old_run
from .v44_run import SPEC, append_event, check_runtime, source_hashes, verify_reference_cache


def cold(root, out):
    spec = yaml.safe_load((root/SPEC).read_text())
    _, query, _, _ = synthetic()
    results = {}
    for recipe in spec["recipes"]:
        model = LongBartRegressor.load(out/f"{recipe}.json")
        expected = np.load(out/f"{recipe}-prediction.npy", allow_pickle=False)
        values = [model.predict(query), model.predict(query.iloc[::-1])[::-1],
                  np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0,len(query),37)])]
        diff = max(float(np.max(np.abs(p-expected))) for p in values)
        if diff > spec["preflight"]["cold_predict_atol"]:
            raise ValueError("Cold/order/chunk mismatch")
        results[recipe] = diff
    print(json.dumps(results))


def run(root):
    spec = yaml.safe_load((root/SPEC).read_text())
    check_runtime(spec); verify_reference_cache(root,spec); verify_old_run(root,spec)
    prefix_spec=spec["prefix_preflight"];prefix_dir=root/prefix_spec["directory"]
    if file_hash(prefix_dir/"report.json")!=prefix_spec["report_sha256"]:
        raise ValueError("V43 preflight report changed")
    for recipe,sha in prefix_spec["model_sha256"].items():
        if file_hash(prefix_dir/f"{recipe}.json")!=sha:
            raise ValueError("V43 synthetic model changed")
    out=root/"local/runs/round2-v44/preflight-r1"
    out.mkdir(parents=True,exist_ok=False)
    train,query,y_train,y_query=synthetic()
    constant=float(np.abs(y_query-np.median(y_train)).mean())
    rows=[]
    for recipe in spec["recipes"]:
        append_event(out/"events.jsonl",{"event":"fit_start","recipe":recipe,"synthetic":True})
        start=time.monotonic()
        try:
            model=LongBartRegressor(recipe,spec["training"]).fit(train,y_train)
            pred=model.predict(query)
            mae=float(np.abs(pred-y_query).mean())
            model.save(out/f"{recipe}.json")
            with (out/f"{recipe}-prediction.npy").open("xb") as stream: np.save(stream,pred,allow_pickle=False)
            write_new(out/f"{recipe}-metadata.json",model.metadata())
            prefix=verify_prefix(model,json.loads((prefix_dir/f"{recipe}.json").read_text()),
                                 query,np.load(prefix_dir/f"{recipe}-prediction.npy",allow_pickle=False))
            write_new(out/f"{recipe}-prefix-verification.json",prefix)
            row={"recipe":recipe,"prefix_prediction_difference":prefix["max_prediction_difference"],"seconds":time.monotonic()-start,"mae":mae,"constant_mae":constant,
                 "retained_draws":len(model.draws_),"sweeps":len(model.trace_),
                 "peak_rss_mib":model.metadata()["peak_rss_mib"]}
            rows.append(row);append_event(out/"events.jsonl",{"event":"complete",**row})
            print(json.dumps(row),flush=True)
        except BaseException as exc:
            append_event(out/"events.jsonl",{"event":"failed","recipe":recipe,"error":repr(exc)})
            raise
    projection=2*max(r["seconds"] for r in rows)*spec["budget"]["development_solver_fits"]/4/3600
    peak=max(r["peak_rss_mib"] for r in rows)
    checks={"learnability":all(r["mae"]<constant for r in rows),
            "interaction_mechanism":rows[1]["mae"]<rows[0]["mae"],
            "runtime":projection<=spec["preflight"]["max_projected_hours"],
            "worker_memory":peak<=spec["preflight"]["max_worker_rss_mib"],
            "available_memory":4*peak+1024<psutil.virtual_memory().available/1024**2}
    completed=subprocess.run([sys.executable,"-m","bf_tap_r2.v44_preflight","--cold",str(out)],
                             cwd=root,check=True,text=True,capture_output=True)
    differences=json.loads(completed.stdout)
    report={"status":"passed" if all(checks.values()) else "failed","checks":checks,"results":rows,
            "projected_hours":projection,"cold_differences":differences,"reference_cache_verified":True,
            "spec_sha256":file_hash(root/SPEC),"source_hashes":source_hashes(root),"synthetic_fits":2,"official_fits":0}
    write_new(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'}),flush=True)
    if report["status"]!='passed': raise ValueError("V44 preflight refused")


if __name__ == "__main__":
    if len(sys.argv)==3 and sys.argv[1]=='--cold': cold(Path.cwd(),Path(sys.argv[2]))
    else: run(Path.cwd())
