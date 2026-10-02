"""Serial Laplace engineering/development with 600-second observations."""
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
    phases = ['train', 'cold'] + (['evaluate', 'audit'] if manifest['phase'] == 'development' else [])
    events = []
    for phase in phases:
        with (out / (phase + '.log')).open('xb') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'bf_tap_r2.q75_laplace_time',
                phase, '--output', str(out)], cwd=WORK, env=dict(os.environ, PYTHONPATH=str(WORK / 'src')),
                stdout=stream, stderr=subprocess.STDOUT)
            if phase == 'train':
                write_new(out / 'activation.json', dict(supervisor_pid=os.getpid(), worker_pid=process.pid,
                    manifest_sha256=sha(out / 'manifest.json'), phase=manifest['phase'], automatic_retries=False))
            events.append(wait_actual(process, out, phase))
        if events[-1]['exit_code'] != 0:
            write_new(out / 'terminal.json', dict(status='failed', events=events, automatic_retries=False))
            raise SystemExit(events[-1]['exit_code'])
    expected = 40 if manifest['phase'] == 'development' else 4
    training = json.loads((out / 'training-complete.json').read_text())
    cold = json.loads((out / 'cold-audit.json').read_text())
    if (training['optimizer_runs'] != expected or cold['cold_states'] != expected
            or max(row['peak_rss_mib'] for row in events) > manifest['spec']['max_worker_rss_mib']):
        raise ValueError('Actual terminal budget/cold/memory gate failed')
    artifacts = ['manifest.json', 'training-complete.json', 'cold-audit.json']
    if manifest['phase'] == 'development':
        artifacts += ['report.json', 'independent-score.json']
    result = dict(status='passed', events=events, actual_exit_codes=[row['exit_code'] for row in events],
        manifest_sha256=sha(out / 'manifest.json'), optimizer_runs=expected, cold_states=expected,
        artifacts={name: sha(out / name) for name in artifacts}, full_fits=0, packages=0, uploads=0)
    write_new(out / 'terminal.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
