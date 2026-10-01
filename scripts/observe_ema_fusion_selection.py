"""Serial launch, real exit events, 600-second observations, no retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK/'src'))
from bf_tap_r2.ema_evaluation_diagnostics import sha, write_new


def wait_actual(process, out, phase):
    done = threading.Event()
    result = {}
    def reap():
        try:
            pid, status, usage = os.wait4(process.pid, 0)
            result.update(pid=pid, exit_code=os.waitstatus_to_exitcode(status), terminal_ns=time.time_ns(),
                peak_rss_mib=usage.ru_maxrss/1024)
            process.returncode = result['exit_code']
        except BaseException as error:
            result['wait_error'] = repr(error)
        finally:
            done.set()
    threading.Thread(target=reap, daemon=True).start()
    ordinal = 0
    while not done.wait(600):
        ordinal += 1
        counts = dict(units_warm=len(list(out.glob('s*-f*/warm-complete.json'))),
            units_cold=len(list(out.glob('s*-f*/cold-complete.json'))),
            native_started=len(list(out.glob('s*-f*/native/call-*/start.json'))),
            native_completed=len(list(out.glob('s*-f*/native/call-*/complete.json'))))
        observation = dict(phase=phase, time_ns=time.time_ns(), pid=process.pid, interval_seconds=600, counts=counts)
        write_new(out/f'{phase}-observation-{ordinal:04d}.json', observation)
        print(json.dumps(observation), flush=True)
    if 'wait_error' in result:
        raise RuntimeError(result['wait_error'])
    write_new(out/(phase+'-terminal.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    previous = Path(manifest['main_root'])/'local/runs/modernnca-q75-development-20261002/launch-r1/terminal-verification.json'
    terminal = json.loads(previous.read_text())
    if terminal['status'] != 'passed' or terminal['actual_exit_codes'] != [0, 0]:
        raise ValueError('Actual audited ModernNCA terminal required before serial entry')
    env = dict(os.environ, PYTHONPATH=str(WORK/'src'))
    events = []
    for phase, command in [
        ('development', [sys.executable, '-m', 'bf_tap_r2.ema_fusion_selection', '--run', '--output', str(out)]),
        ('independent-score', [sys.executable, str(WORK/'scripts/audit_q75_strategy_vectors.py'), '--kind', 'selector', '--output', str(out)])]:
        with (out/(phase+'.log')).open('xb') as stream:
            process = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if phase == 'development':
                activation = dict(supervisor_pid=os.getpid(), controller_pid=process.pid, started_ns=time.time_ns(),
                    manifest_sha256=sha(out/'manifest.json'), previous_terminal_sha256=sha(previous),
                    observer_source_sha256=sha(__file__), monitor_seconds=600, automatic_retries=False)
                write_new(out/'activation.json', activation)
                print(json.dumps(activation), flush=True)
            result = wait_actual(process, out, phase)
        events.append(result)
        if result['exit_code'] != 0:
            write_new(out/'terminal-verification.json', dict(status='failed', events=events, automatic_retries=0))
            raise SystemExit(result['exit_code'])
    units = list(out.glob('s*-f*/cold-complete.json'))
    native = [json.loads(p.read_text()) for p in out.glob('s*-f*/native/scope-complete.json')]
    if len(units) != 10 or len(native) != 10 or sum(p['counts']['torch_optimizer'] for p in native) != 30:
        raise ValueError('Full declared native budget and cold coverage required')
    if max(p['peak_rss_mib'] for p in events) > manifest['spec']['max_worker_rss_mib']:
        raise ValueError('Actual process memory gate failed')
    checked = dict(status='passed', actual_exit_codes=[p['exit_code'] for p in events], events=events,
        native_optimizer_runs=30, cold_states=40, manifest_sha256=sha(out/'manifest.json'),
        report_sha256=sha(out/'report.json'), independent_score_sha256=sha(out/'independent-score.json'),
        maximum_runtime_seconds=None, automatic_retries=0, packages=0, desktop_writes=0, uploads=0)
    write_new(out/'terminal-verification.json', checked)
    print(json.dumps(checked), flush=True)


if __name__ == '__main__':
    main()
