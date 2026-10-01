"""Serial actual-exit supervision; observations exactly 600 seconds apart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK/'src'))
from bf_tap_r2.ema_evaluation_diagnostics import sha, write_new
from bf_tap_r2.ema_retraining_validation import require_previous
from observe_ema_fusion_selection import wait_actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); out = Path(args.output).resolve()
    manifest = json.loads((out/'manifest.json').read_text()); spec = manifest['spec']
    if require_previous(spec) != manifest['previous_terminal_sha256']:
        raise ValueError('Prior terminal changed since freeze')
    env = dict(os.environ, PYTHONPATH=str(WORK/'src'))
    results = []
    for phase, command in [('development', [sys.executable, '-m', 'bf_tap_r2.ema_retraining_validation', '--run', '--output', str(out)]),
        ('independent-score', [sys.executable, str(WORK/'scripts/audit_ema_retraining_vectors.py'), '--output', str(out)])]:
        with (out/(phase+'.log')).open('xb') as stream:
            process = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if phase == 'development':
                write_new(out/'activation.json', dict(supervisor_pid=os.getpid(), controller_pid=process.pid,
                    manifest_sha256=sha(out/'manifest.json'), monitor_seconds=600, automatic_retries=False,
                    maximum_runtime_seconds=None, phase=spec['phase']))
            result = wait_actual(process, out, phase)
        results.append(result)
        if result['exit_code'] != 0:
            write_new(out/'terminal-verification.json', dict(status='failed', events=results, automatic_retries=0))
            raise SystemExit(result['exit_code'])
    report = json.loads((out/'report.json').read_text())
    if report['optimizer_runs'] != spec['optimizer_runs'] or report['cold_states'] != spec['retained_states']:
        raise ValueError('Actual full native/state coverage required')
    if max(r['peak_rss_mib'] for r in results) > spec['max_worker_rss_mib']:
        raise ValueError('Actual process RSS gate failed')
    receipt = dict(status='passed', phase=spec['phase'], actual_exit_codes=[r['exit_code'] for r in results],
        events=results, native_optimizer_runs=report['optimizer_runs'], cold_states=report['cold_states'],
        report_sha256=sha(out/'report.json'), independent_score_sha256=sha(out/'independent-score.json'),
        manifest_sha256=sha(out/'manifest.json'), packages=0, uploads=0, automatic_retries=0)
    write_new(out/'terminal-verification.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
