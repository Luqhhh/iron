"""Capture actual zero-fit review exits without overwriting old receipts."""
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
    events = []
    for phase in ('evaluate', 'audit'):
        with (out / (phase + '.log')).open('xb') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'bf_tap_r2.q75_gaussian_time',
                phase, '--output', str(out)], cwd=WORK,
                env=dict(os.environ, PYTHONPATH=str(WORK / 'src')), stdout=stream, stderr=subprocess.STDOUT)
            events.append(wait_actual(process, out, phase))
        if events[-1]['exit_code'] != 0:
            write_new(out / 'terminal.json', dict(status='failed', events=events, automatic_retries=0))
            raise SystemExit(events[-1]['exit_code'])
    write_new(out / 'terminal.json', dict(status='passed', events=events,
        actual_exit_codes=[row['exit_code'] for row in events],
        manifest_sha256=sha(out / 'manifest.json'), report_sha256=sha(out / 'report.json'),
        independent_score_sha256=sha(out / 'independent-score.json'), new_fits=0, new_optimizers=0))
    print(json.dumps(json.loads((out / 'terminal.json').read_text())), flush=True)


if __name__ == '__main__':
    main()
