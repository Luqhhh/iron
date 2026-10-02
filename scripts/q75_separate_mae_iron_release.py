"""Serial release phases with actual process exits and 600-second observations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
sys.path.insert(0, str(WORK / 'scripts'))
from bf_tap_r2.ema_evaluation_diagnostics import sha, write_new
from observe_ema_fusion_selection import wait_actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    events = []
    for phase in ['train','cold','predict','release','audit']:
        with (out / f'{phase}.log').open('xb') as stream:
            process = subprocess.Popen([sys.executable,'-m','bf_tap_r2.q75_separate_mae_iron_release',
                phase,'--output',str(out)], cwd=WORK, env=dict(os.environ,PYTHONPATH=str(WORK / 'src')),
                stdout=stream, stderr=subprocess.STDOUT)
            if phase == 'train':
                write_new(out / 'activation.json', dict(supervisor_pid=os.getpid(),worker_pid=process.pid,
                    manifest_sha256=sha(out / 'manifest.json'),automatic_retries=False))
            events.append(wait_actual(process,out,phase))
        if events[-1]['exit_code'] != 0 or events[-1]['peak_rss_mib'] > manifest['spec']['max_worker_rss_mib']:
            write_new(out / 'terminal.json',dict(status='failed',events=events,automatic_retries=False))
            raise SystemExit(events[-1]['exit_code'] or 1)
    artifacts = ['manifest.json','warm-complete.json','cold-audit.json','prediction-audit.json','package.json','package-audit.json']
    evidence = json.loads((out / 'package-audit.json').read_text())
    if evidence['status'] != 'passed' or evidence['optimizer_runs'] != 2 or evidence['cold_states'] != 2:
        raise ValueError('Actual release budget/cold/package gate failed')
    result = dict(status='passed',events=events,actual_exit_codes=[r['exit_code'] for r in events],
        artifacts={n:sha(out / n) for n in artifacts},optimizer_runs=2,cold_states=2,packages=1,
        desktop_writes=0,uploads=0,zip_sha256=evidence['zip_sha256'])
    write_new(out / 'terminal.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
