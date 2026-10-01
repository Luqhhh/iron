"""Serial launch with real child exits, 600-second observations and no retry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'src'))
from bf_tap_r2.ema_evaluation_diagnostics import sha, write_new


def wait_actual(process,out,phase):
    done = threading.Event(); result = {}
    def reap():
        try:
            pid,status,usage = os.wait4(process.pid,0)
            result.update(pid=pid,exit_code=os.waitstatus_to_exitcode(status),terminal_ns=time.time_ns(),peak_rss_mib=usage.ru_maxrss/1024)
            process.returncode = result['exit_code']
        except BaseException as error: result['wait_error'] = repr(error)
        finally: done.set()
    threading.Thread(target=reap,daemon=True).start(); ordinal = 0
    while not done.wait(600):
        ordinal += 1
        rows = [json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()] if (out/'events.jsonl').exists() else []
        record = dict(phase=phase,pid=process.pid,interval_seconds=600,time_ns=time.time_ns(),
            completed_units=sum(e['event']=='unit_completed' for e in rows),completed_optimizers=sum(e['event']=='optimizer_completed' for e in rows))
        write_new(out/f'{phase}-observation-{ordinal:04d}.json',record); print(json.dumps(record),flush=True)
    if 'wait_error' in result: raise RuntimeError(result['wait_error'])
    write_new(out/f'{phase}-terminal.json',result); return result


def main(output):
    out = Path(output).resolve(); manifest = json.loads((out/'manifest.json').read_text());spec = manifest['spec']
    if (Path(manifest['workspace'])!=WORK or spec['monitor_seconds']!=600 or spec['workers']!=1
            or spec['maximum_runtime_seconds'] is not None or spec['automatic_scientific_retries']):
        raise ValueError('Frozen execution scope differs')
    env = dict(os.environ,PYTHONPATH=str(WORK/'src'))
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[key] = '1'
    events = []
    for phase,command in [
        ('development',[sys.executable,'-m','bf_tap_r2.ema_dropout_development','--output',str(out),'execute']),
        ('independent-score',[sys.executable,str(WORK/'scripts/audit_ema_dropout_development.py'),'--output',str(out)])]:
        with (out/f'{phase}.log').open('x') as log:
            process = subprocess.Popen(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
            if phase=='development':
                activation = dict(supervisor_pid=os.getpid(),controller_pid=process.pid,started_ns=time.time_ns(),
                    manifest_sha256=sha(out/'manifest.json'),monitor_seconds=600,automatic_retries=False)
                write_new(out/'process-launch.json',activation);print(json.dumps(activation),flush=True)
            result = wait_actual(process,out,phase)
        events.append(result)
        if result['exit_code']!=0:
            write_new(out/'terminal-verification.json',dict(status='failed',events=events,actual_exit_codes=[e['exit_code'] for e in events],automatic_retries=0))
            return result['exit_code']
    audit = json.loads((out/'independent-score.json').read_text())
    if audit['status']!='passed' or audit['actual_optimizer_runs']!=40 or audit['new_saved_states']!=40:
        raise ValueError('Full independently audited scientific count required')
    if max(e['peak_rss_mib'] for e in events)>spec['max_worker_rss_mib']: raise ValueError('Actual process memory gate failed')
    result = dict(status='passed',actual_exit_codes=[e['exit_code'] for e in events],events=events,optimizer_runs=40,
        new_saved_states=40,old_saved_states=20,report_sha256=sha(out/'report.json'),independent_score_sha256=sha(out/'independent-score.json'),
        manifest_sha256=sha(out/'manifest.json'),maximum_runtime_seconds=None,automatic_retries=0,packages=0,uploads=0)
    write_new(out/'terminal-verification.json',result);print(json.dumps(result),flush=True);return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--output',required=True);raise SystemExit(main(p.parse_args().output))
