"""Serial frozen horizon iteration; no automatic retry or time budget."""
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from bf_tap_r2.joint_support_audit import local_output, sha, write_new
from bf_tap_r2.laplace_leaf_long_experiment import inventory, validate_spec, SPEC


def wait_with_health(process, on_health):
    """Wait for the bound child, observing only at the registered interval."""
    check = 0
    while True:
        try:
            return process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            check += 1
            try:
                state = psutil.Process(process.pid)
                snapshot = dict(pid=process.pid, utc_ns=time.time_ns(),
                                status=state.status(),
                                rss_mib=state.memory_info().rss/1024**2,
                                threads=state.num_threads(), automatic_action="none")
            except psutil.NoSuchProcess:
                # The child can finish between the timeout and inspection.
                # Only its original Popen object determines the real exit code.
                return process.wait()
            on_health(check, snapshot)


def main():
    root=Path.cwd().resolve()
    validate_spec(json.loads((root/SPEC).read_text()))
    sequence=local_output(root,"local/runs/laplace-leaf-long-20261002/sequence-r1")
    sequence.mkdir(parents=True,exist_ok=False)
    frozen=inventory(root)
    write_new(sequence/"manifest.json",dict(sources=frozen,pid=psutil.Process().pid,started_ns=time.time_ns(),
              monitor_seconds=600,maximum_runtime_seconds=None,automatic_retry=False))
    terminals=[]
    for index,(action,name) in enumerate((("engineering","engineering-r1"),("verify","engineering-r1"),
                                          ("development","development-r1"),("verify","development-r1"))):
        if inventory(root)!=frozen:
            raise ValueError("Long sequence frozen source changed")
        command=[sys.executable,"-B","-m","bf_tap_r2.laplace_leaf_long_experiment",action,"--output",f"local/runs/laplace-leaf-long-20261002/{name}"]
        if action=="development":
            command += ["--engineering","local/runs/laplace-leaf-long-20261002/engineering-r1"]
        with (sequence/f"{index}-{action}-stdout.log").open("xb") as stdout,(sequence/f"{index}-{action}-stderr.log").open("xb") as stderr:
            process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr)
            started=time.time_ns()
            write_new(sequence/f"{index}-{action}-started.json",dict(pid=process.pid,started_ns=started,command=command))
            print(json.dumps(dict(event="phase_started",action=action,pid=process.pid)),flush=True)
            def record_health(check, snapshot):
                write_new(sequence/f"{index}-{action}-health-{check}.json", snapshot)
                print(json.dumps(dict(event="scheduled_600_second_health",action=action,pid=process.pid)),flush=True)
            code=wait_with_health(process, record_health)
        event=dict(action=action,run=name,pid=process.pid,exit_code=code,started_ns=started,completed_ns=time.time_ns())
        write_new(sequence/f"{index}-{action}-terminal.json",event)
        terminals.append(event)
        print(json.dumps(dict(event="phase_terminal",**event)),flush=True)
        if code:
            write_new(sequence/"FAILED.json",dict(phase=event,automatic_retry=False,stderr_sha256=sha(sequence/f"{index}-{action}-stderr.log")))
            return code
    write_new(sequence/"terminal.json",dict(status="completed_all_four_phases",exit_codes=[p["exit_code"] for p in terminals],
              phases=terminals,controller_running=False,automatic_retry=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
