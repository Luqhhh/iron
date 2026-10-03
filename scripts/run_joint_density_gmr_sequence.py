"""Serial source-bound GMR phases; no automatic retry or time budget."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from bf_tap_r2.joint_support_audit import local_output, sha, write_new
from bf_tap_r2.joint_density_gmr_experiment import inventory, validate_spec, SPEC, RUN

EXPERIMENT = "bf_tap_r2.joint_density_gmr_experiment"
PHASES = (("engineering", "engineering-r1"), ("verify", "engineering-r1"),
          ("development", "development-r1"), ("verify", "development-r1"))


def wait_with_health(process, on_health):
    """Observe only every 600 seconds; retain the bound child's actual exit."""
    check = 0
    while True:
        try:
            return process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            check += 1
            try:
                state = psutil.Process(process.pid)
                snapshot = dict(pid=process.pid, utc_ns=time.time_ns(), status=state.status(),
                                rss_mib=state.memory_info().rss/1024**2,
                                threads=state.num_threads(), automatic_action="none")
            except psutil.NoSuchProcess:
                return process.wait()
            on_health(check, snapshot)


def source_check(root, frozen):
    """Missing/unreadable source is an identity failure, never a success."""
    try:
        current = inventory(root)
    except Exception as exc:
        return dict(matches=False, source_read_error_type=type(exc).__name__,
                    source_read_error=str(exc))
    snapshot_digest = hashlib.sha256(json.dumps(current, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(matches=current==frozen, snapshot_sha256=snapshot_digest,
                changed_source_names=sorted(name for name in set(current)|set(frozen)
                                            if current.get(name)!=frozen.get(name)))


def fail_sequence(sequence, terminals, kind, controller_code, details):
    failure = dict(kind=kind, details=details, automatic_retry=False,
                   controller_exit_code=controller_code, completed_ns=time.time_ns())
    write_new(sequence/"FAILED.json", failure)
    write_new(sequence/"terminal.json", dict(status=kind,
        exit_codes=[event["exit_code"] for event in terminals], phases=terminals,
        controller_running=False, controller_exit_code=controller_code,
        automatic_retry=False, packages=0, agent_uploads=0, failure=failure))
    return controller_code


def main():
    root = Path.cwd().resolve()
    frozen = inventory(root)
    validate_spec(json.loads((root/SPEC).read_text(encoding="utf-8")))
    sequence = local_output(root, f"{RUN}/sequence-r1")
    sequence.mkdir(parents=True, exist_ok=False)
    write_new(sequence/"manifest.json", dict(sources=frozen, pid=psutil.Process().pid,
        started_ns=time.time_ns(), monitor_seconds=600, expected_phases=PHASES,
        maximum_runtime_seconds=None, automatic_retry=False,
        packages=0, agent_uploads=0))
    terminals = []
    for index, (action, name) in enumerate(PHASES):
        before = source_check(root, frozen)
        if not before["matches"]:
            return fail_sequence(sequence, terminals, "controller_failed_source_identity", 1,
                                 dict(phase_index=index, when="before_child", source_check=before))
        command = [sys.executable, "-B", "-m", EXPERIMENT,
                   action, "--output", f"{RUN}/{name}"]
        if action=="development":
            command += ["--engineering", f"{RUN}/engineering-r1"]
        stdout_path, stderr_path = sequence/f"{index}-{action}-stdout.log", sequence/f"{index}-{action}-stderr.log"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
            started = time.time_ns()
            write_new(sequence/f"{index}-{action}-started.json", dict(
                pid=process.pid, started_ns=started, command=command,
                source_check_before=before))
            print(json.dumps(dict(event="phase_started", action=action, pid=process.pid)), flush=True)

            def record_health(check, snapshot):
                write_new(sequence/f"{index}-{action}-health-{check}.json", snapshot)
                print(json.dumps(dict(event="scheduled_600_second_health", action=action,
                                      pid=process.pid)), flush=True)

            code = wait_with_health(process, record_health)
            completed = time.time_ns()
        # Always preserve the actual child terminal first; an ensuing identity
        # failure belongs to the controller, even when this child exited zero.
        after = source_check(root, frozen)
        event = dict(action=action, run=name, command=command, pid=process.pid,
            exit_code=code, started_ns=started, completed_ns=completed,
            stdout_sha256=sha(stdout_path), stderr_sha256=sha(stderr_path),
            source_check_before=before, source_check_after=after)
        write_new(sequence/f"{index}-{action}-terminal.json", event)
        terminals.append(event)
        print(json.dumps(dict(event="phase_terminal", **event)), flush=True)
        if not after["matches"]:
            return fail_sequence(sequence, terminals, "controller_failed_source_identity", 1,
                                 dict(phase_index=index, when="after_child", actual_child_exit_code=code,
                                      source_check=after))
        if code:
            return fail_sequence(sequence, terminals, "failed_child_phase", code,
                                 dict(phase_index=index, phase=event))
    write_new(sequence/"terminal.json", dict(status="completed_all_four_phases",
        exit_codes=[event["exit_code"] for event in terminals], phases=terminals,
        controller_running=False, controller_exit_code=0, source_identity_passed_all_phases=True,
        automatic_retry=False, packages=0, agent_uploads=0))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
