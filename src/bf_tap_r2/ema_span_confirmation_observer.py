"""Observe only this bound controller every 600 seconds or at actual exit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def observe(workspace,out):
    workspace=Path(workspace).resolve();out=Path(out).resolve()
    manifest=json.loads((out/'manifest.json').read_text());spec=manifest['spec']
    if (manifest['workspace']!=str(workspace) or manifest['output']!=str(out)
            or manifest['scientific_execution_admitted'] is not True or spec['monitor_seconds']!=600
            or spec['workers']!=1 or spec['maximum_runtime_seconds'] is not None):
        raise ValueError('Frozen observer/controller scope changed')
    command=[sys.executable,str(workspace/'scripts/ema_span_confirmation.py'),
        '--workspace',str(workspace),'--output',str(out),'execute']
    env=dict(os.environ);env['PYTHONPATH']=str(workspace/'src')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    started=time.monotonic();next_check=started+600
    with (out/'observer.log').open('x') as log:
        process=subprocess.Popen(command,cwd=workspace,env=env,stdout=log,stderr=subprocess.STDOUT)
        launch=dict(controller_pid=process.pid,observer_pid=os.getpid(),command=command,
            started_ns=time.time_ns(),monitor_interval_seconds=600)
        with (out/'process-launch.json').open('x') as stream:
            json.dump(launch,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(launch),flush=True)
        while True:
            try:
                code=process.wait(timeout=min(60,max(.01,next_check-time.monotonic())))
            except subprocess.TimeoutExpired:
                if time.monotonic()<next_check:continue
                path=out/'events.jsonl';events=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
                record=dict(event='scheduled_600_second_check',controller_pid=process.pid,
                    elapsed_seconds=time.monotonic()-started,
                    completed_units=sum(e['event']=='unit_completed' for e in events),
                    failed_units=sum(e['event']=='unit_failed' for e in events))
                with (out/'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())
                print(json.dumps(record),flush=True);next_check+=600
                continue
            terminal=dict(event='actual_controller_completion',controller_pid=process.pid,
                observer_pid=os.getpid(),exit_code=code,completed_ns=time.time_ns(),elapsed_seconds=time.monotonic()-started)
            with (out/'process-terminal.json').open('x') as stream:
                json.dump(terminal,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            print(json.dumps(terminal),flush=True)
            return code


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raise SystemExit(observe(a.workspace,a.output))
