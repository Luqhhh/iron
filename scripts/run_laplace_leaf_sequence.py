"""Serial leaf-median research with 600-second own-process health checks."""
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from bf_tap_r2.joint_support_audit import local_output, sha, write_new
from bf_tap_r2.laplace_leaf_experiment import SPEC, inventory, validate_spec


def main():
    root=Path.cwd().resolve()
    validate_spec(json.loads((root/SPEC).read_text()))
    sequence=local_output(root,"local/runs/laplace-leaf-median-20261002/sequence-r1")
    sequence.mkdir(parents=True,exist_ok=False)
    frozen=inventory(root)
    write_new(sequence/"manifest.json",dict(source_sha256=frozen,pid=psutil.Process().pid,started_ns=time.time_ns(),
              monitor_seconds=600,maximum_runtime_seconds=None,automatic_retry=False))
    phases=(("engineering","engineering-r1"),("verify","engineering-r1"),("development","development-r1"),("verify","development-r1"))
    terminal=[]
    for index,(action,name) in enumerate(phases):
        output=f"local/runs/laplace-leaf-median-20261002/{name}"
        command=[sys.executable,"-B","-m","bf_tap_r2.laplace_leaf_experiment",action,"--output",output]
        if action=="development":
            command += ["--engineering","local/runs/laplace-leaf-median-20261002/engineering-r1"]
        if inventory(root)!=frozen:
            raise ValueError("Sequence source changed")
        with (sequence/f"{index}-{action}-stdout.log").open("xb") as stdout,(sequence/f"{index}-{action}-stderr.log").open("xb") as stderr:
            process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr)
            started=time.time_ns()
            write_new(sequence/f"{index}-{action}-started.json",dict(pid=process.pid,command=command,started_ns=started))
            print(json.dumps(dict(event="phase_started",action=action,run=name,pid=process.pid)),flush=True)
            check=0
            while True:
                try:
                    code=process.wait(timeout=600)
                    break
                except subprocess.TimeoutExpired:
                    check+=1
                    state=psutil.Process(process.pid)
                    write_new(sequence/f"{index}-{action}-health-{check}.json",dict(pid=process.pid,utc_ns=time.time_ns(),
                              status=state.status(),rss_mib=state.memory_info().rss/1024**2,threads=state.num_threads(),automatic_action="none"))
                    print(json.dumps(dict(event="scheduled_600_second_health",action=action,pid=process.pid)),flush=True)
        event=dict(action=action,run=name,pid=process.pid,exit_code=code,started_ns=started,completed_ns=time.time_ns())
        write_new(sequence/f"{index}-{action}-terminal.json",event)
        terminal.append(event)
        print(json.dumps(dict(event="phase_terminal",**event)),flush=True)
        if code:
            write_new(sequence/"FAILED.json",dict(phase=event,automatic_retry=False,stderr_sha256=sha(sequence/f"{index}-{action}-stderr.log")))
            return code
    write_new(sequence/"terminal.json",dict(status="completed_all_four_phases",exit_codes=[a["exit_code"] for a in terminal],
              phases=terminal,source_sha256=frozen,controller_running=False,automatic_retry=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
