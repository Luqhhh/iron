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


sys.path.insert(0, str(WORK/'scripts'))
from observe_ema_dropout_development import wait_actual


def main(output):
    out = Path(output).resolve(); manifest = json.loads((out/'manifest.json').read_text());spec = manifest['spec']
    if (Path(manifest['workspace'])!=WORK or spec['monitor_seconds']!=600 or spec['workers']!=1
            or spec['maximum_runtime_seconds'] is not None or spec['automatic_scientific_retries']):
        raise ValueError('Frozen execution scope differs')
    env = dict(os.environ,PYTHONPATH=str(WORK/'src'))
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[key] = '1'
    events = []
    for phase,command in [
        ('development',[sys.executable,'-m','bf_tap_r2.sam_ema_development','--output',str(out),'execute']),
        ('independent-score',[sys.executable,str(WORK/'scripts/audit_sam_ema_development.py'),'--output',str(out)])]:
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
    if audit['status']!='passed' or audit['actual_optimizer_runs']!=20 or audit['new_saved_states']!=20 or audit['old_saved_states']!=40:
        raise ValueError('Full independently audited scientific count required')
    if max(e['peak_rss_mib'] for e in events)>spec['max_worker_rss_mib']: raise ValueError('Actual process memory gate failed')
    result = dict(status='passed',actual_exit_codes=[e['exit_code'] for e in events],events=events,optimizer_runs=20,
        new_saved_states=20,old_saved_states=40,report_sha256=sha(out/'report.json'),independent_score_sha256=sha(out/'independent-score.json'),
        manifest_sha256=sha(out/'manifest.json'),maximum_runtime_seconds=None,automatic_retries=0,packages=0,uploads=0)
    write_new(out/'terminal-verification.json',result);print(json.dumps(result),flush=True);return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--output',required=True);raise SystemExit(main(p.parse_args().output))
