"""Record actual zero-fit review exits; observe only at 600-second intervals."""
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
    exits = []
    for phase in ('evaluate', 'audit'):
        with (out / (phase + '.log')).open('xb') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'bf_tap_r2.q75_joint_ema_time',
                phase, '--output', str(out)], cwd=WORK,
                env=dict(os.environ, PYTHONPATH=str(WORK / 'src')),
                stdout=stream, stderr=subprocess.STDOUT)
            exits.append(wait_actual(process, out, phase))
        if exits[-1]['exit_code'] != 0:
            write_new(out / 'terminal.json', dict(status='failed', events=exits, automatic_retries=0))
            raise SystemExit(exits[-1]['exit_code'])
    spec = json.loads((out / 'manifest.json').read_text())['spec']
    if max(x['peak_rss_mib'] for x in exits) > spec['max_worker_rss_mib']:
        raise ValueError('Process RSS gate failed')
    result = dict(status='passed', events=exits, actual_exit_codes=[x['exit_code'] for x in exits],
        manifest_sha256=sha(out / 'manifest.json'), report_sha256=sha(out / 'report.json'),
        independent_score_sha256=sha(out / 'independent-score.json'),
        new_fits=0, monitor_seconds=600, automatic_retries=0)
    write_new(out / 'terminal.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
