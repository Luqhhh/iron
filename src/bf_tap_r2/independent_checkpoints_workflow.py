"""Durable serial supervisor: scheduled observations only, no automatic retry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .independent_checkpoints_run import RUN_ROOT
from .v49_run import append_event
from .v7_periodic import write_new


def main():
    root = Path.cwd(); directory = root/RUN_ROOT; directory.mkdir(parents=True, exist_ok=True)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1": raise ValueError(f"Set {key}=1")
    write_new(directory/"workflow-start.json", {"pid": os.getpid(), "time_ns": time.time_ns(), "interval_seconds": 600,
        "queues": ["DE3", "E-COMPOSE"], "serial": True, "automatic_retry": False})
    events = []
    for queue in ("DE3", "E-COMPOSE"):
        dev = f"{RUN_ROOT}/development-{queue}"; conf = f"{RUN_ROOT}/confirmation-{queue}"
        stages = [("preflight", "independent_checkpoints_preflight", ["--queue", queue]),
                  ("development", "independent_checkpoints_run", ["--queue", queue, "--output", dev]),
                  ("development_audit", "independent_checkpoints_audit", ["--output", dev]),
                  ("conditional_confirmation", "independent_checkpoints_run", ["--queue", queue, "--output", conf, "--development", dev]),
                  ("confirmation_audit", "independent_checkpoints_audit", ["--output", conf])]
        for stage, module, args in stages:
            if stage == "confirmation_audit" and not (root/conf).exists():
                event = {"stage": stage, "queue": queue, "skipped": "no_development_finalist", "new_fits": 0, "time_ns": time.time_ns()}
                events.append(event); append_event(directory/"workflow-events.jsonl", event); continue
            with (directory/f"{queue}-{stage}.log").open("x") as stream:
                child = subprocess.Popen([sys.executable, "-m", f"bf_tap_r2.{module}", *args], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
                event = {"event": "start", "queue": queue, "stage": stage, "pid": child.pid, "time_ns": time.time_ns()}
                append_event(directory/"workflow-events.jsonl", event); print(json.dumps(event), flush=True)
                while True:
                    try: code = child.wait(timeout=600); break
                    except subprocess.TimeoutExpired:
                        observation = {"event": "scheduled_observation", "queue": queue, "stage": stage, "pid": child.pid,
                                       "status": "running", "time_ns": time.time_ns()}
                        run = root/(conf if "confirmation" in stage else dev if "development" in stage else f"{RUN_ROOT}/preflight-{queue}")
                        ledger = run/("fit_ledger.jsonl" if "preflight" not in stage else "events.jsonl")
                        if ledger.exists():
                            rows = [json.loads(s) for s in ledger.read_text().splitlines()]
                            observation["completed_units"] = sum(r.get("event") == "complete" for r in rows)
                            observation["failed_units"] = sum(r.get("event") == "failed" for r in rows)
                        append_event(directory/"scheduled-observations.jsonl", observation); print(json.dumps(observation), flush=True)
            event = {"queue": queue, "stage": stage, "returncode": code, "time_ns": time.time_ns()}
            events.append(event); append_event(directory/"workflow-events.jsonl", event); print(json.dumps(event), flush=True)
            if code:
                write_new(directory/"completion-event.json", {"status": "failed", "events": events, "automatic_retry": False})
                raise SystemExit(code)
    write_new(directory/"completion-event.json", {"status": "completed", "events": events, "packages": 0, "uploads": 0})
    print(json.dumps({"status": "workflow_completed", "packages": 0, "uploads": 0}), flush=True)


if __name__ == "__main__": main()
