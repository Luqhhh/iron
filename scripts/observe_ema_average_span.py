"""Launch one frozen batch; observe every 600 seconds or actual process exit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(workspace, out):
    workspace, out = Path(workspace).resolve(), Path(out).resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    if str(workspace) != manifest['workspace']:
        raise ValueError('Observer workspace identity mismatch')
    spec = manifest['spec']
    if spec['monitor_seconds'] != 600 or spec['workers'] != 1 or spec['maximum_runtime_seconds'] is not None:
        raise ValueError('Frozen monitoring scope changed')
    command = [sys.executable, str(workspace/'scripts/ema_average_span.py'),
               '--workspace', str(workspace), '--output', str(out), 'execute']
    env = os.environ.copy(); env['PYTHONPATH'] = str(workspace/'src')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    started = time.monotonic(); next_check = started+600
    with (out/'observer.log').open('x') as log:
        started_ns = time.time_ns()
        process = subprocess.Popen(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT)
        launch = dict(event='controller_started', controller_pid=process.pid, observer_pid=os.getpid(),
                      command=command, started_ns=started_ns, monitor_interval_seconds=600)
        with (out/'process-launch.json').open('x') as stream:
            json.dump(launch, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(launch), flush=True)
        while True:
            try:
                code = process.wait(timeout=min(60, max(.01, next_check-time.monotonic())))
            except subprocess.TimeoutExpired:
                if time.monotonic() < next_check:
                    continue
                events = [json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()] if (out/'events.jsonl').exists() else []
                record = dict(event='scheduled_600_second_check', controller_pid=process.pid,
                    elapsed_seconds=time.monotonic()-started, completed_units=sum(e['event'] == 'unit_completed' for e in events),
                    failed_units=sum(e['event'] == 'unit_failed' for e in events))
                with (out/'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n'); stream.flush(); os.fsync(stream.fileno())
                print(json.dumps(record), flush=True); next_check += 600
                continue
            terminal = dict(event='actual_controller_completion', controller_pid=process.pid,
                            exit_code=code, completed_ns=time.time_ns(), elapsed_seconds=time.monotonic()-started)
            with (out/'process-terminal.json').open('x') as stream:
                json.dump(terminal, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            print(json.dumps(terminal), flush=True)
            return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    raise SystemExit(main(args.workspace, args.output))
