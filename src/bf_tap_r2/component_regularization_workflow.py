"""Sequential admitted stages with 600-second observations and terminal events.

No polling loop: each wait blocks on the child process, with a ten-minute
timeout for scheduled health observations. This supervisor never restarts jobs.
"""
import json
import argparse
import yaml
import os
from pathlib import Path
import subprocess
import sys
import time

from .component_regularization_run import RUN_ROOT, SPEC
from .v49_run import append_event
from .v7_periodic import write_new


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--spec",default=SPEC);args=parser.parse_args()
    spec_path=args.spec;spec=yaml.safe_load(Path(spec_path).read_text());run_root=spec.get("run_root",RUN_ROOT)
    root=Path.cwd();directory=root/run_root
    directory.mkdir(parents=True,exist_ok=True)
    lock=directory/"workflow-start.json"
    write_new(lock,{"pid":os.getpid(),"started_time_ns":time.time_ns(),"interval_seconds":600})
    stages=[("preflight","bf_tap_r2.component_regularization_preflight",[]),
        ("development","bf_tap_r2.component_regularization_run",["--output",f"{run_root}/development-r1"]),
        ("development_audit","bf_tap_r2.component_regularization_audit",["--output",f"{run_root}/development-r1"]),
        ("conditional_confirmation","bf_tap_r2.component_regularization_run",["--output",f"{run_root}/confirmation-r1","--development",f"{run_root}/development-r1"]),
        ("confirmation_audit","bf_tap_r2.component_regularization_audit",["--output",f"{run_root}/confirmation-r1"])]
    events=[]
    for name,module,args in stages:
        if name=="confirmation_audit" and not (directory/"confirmation-r1").exists():
            event={"stage":name,"skipped":"no_development_finalist","new_fits":0,"time_ns":time.time_ns()}
            events.append(event);append_event(directory/"workflow-events.jsonl",event);continue
        with (directory/f"{name}.log").open("x") as stream:
            child=subprocess.Popen([sys.executable,"-m",module,*args,"--spec",spec_path],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
            append_event(directory/"workflow-events.jsonl",{"event":"start","stage":name,"pid":child.pid})
            print(json.dumps({"event":"start","stage":name,"pid":child.pid}),flush=True)
            while True:
                try:code=child.wait(timeout=600);break
                except subprocess.TimeoutExpired:
                    status={"event":"scheduled_observation","stage":name,"pid":child.pid,
                            "time_ns":time.time_ns(),"status":"running"}
                    # Read progress only on the declared 600-second boundary.
                    run=directory/("preflight-r1" if name=="preflight" else "confirmation-r1" if "confirmation" in name else "development-r1")
                    ledger=run/("events.jsonl" if name=="preflight" else "fit_ledger.jsonl")
                    if ledger.exists():
                        rows=[json.loads(line) for line in ledger.read_text().splitlines()]
                        status["completed_units"]=sum(r.get("event")=="complete" for r in rows)
                        status["failed_units"]=sum(r.get("event")=="failed" for r in rows)
                    append_event(directory/"scheduled-observations.jsonl",status)
                    print(json.dumps(status),flush=True)
        event={"stage":name,"returncode":code,"time_ns":time.time_ns()}
        events.append(event);append_event(directory/"workflow-events.jsonl",event)
        print(json.dumps(event),flush=True)
        if code:
            write_new(directory/"completion-event.json",{"status":"failed","events":events,"no_automatic_restart":True})
            raise SystemExit(code)
    write_new(directory/"completion-event.json",{"status":"completed","events":events,"packages":0,"uploads":0})
    print(json.dumps({"status":"workflow_completed","packages":0,"uploads":0}),flush=True)


if __name__=="__main__":main()
