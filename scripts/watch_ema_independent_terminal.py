"""Wait on the owned controller's Linux exit event, then run one frozen audit.

No progress polling, training, retries, or process termination.  The original
supervisor's tool exit code must still be reconciled separately by the caller.
"""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import os
import platform
import select
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # This locked standalone Python/libc lacks the wrapper. Linux x86_64's
    # installed asm/unistd_64.h declares __NR_pidfd_open=434. The actual kernel
    # path is tested with real process exits; other architectures fail closed.
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise RuntimeError('No verified pidfd interface on this platform')
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_uint]
    syscall.restype = ctypes.c_long
    fd = syscall(434, pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def run(directory):
    directory = Path(directory).resolve()
    manifest = read(directory/'manifest.json')
    for p,h in manifest['files'].items():
        if sha(p) != h:
            raise ValueError('Frozen followup input changed: '+p)
    pid = manifest['controller_pid']
    process = Path('/proc')/str(pid)
    # pidfd is bound to this process instance; a later PID reuse cannot trigger
    # a successful wait for another job. Verify command and source on both sides.
    expected = manifest['controller_cmdline_hex']
    if process.joinpath('cmdline').read_bytes().hex() != expected:
        raise ValueError('Owned controller command differs')
    fd = open_pidfd(pid)
    try:
        if (process.joinpath('cmdline').read_bytes().hex() != expected
                or process.joinpath('cwd').resolve() != Path(manifest['controller_worktree'])):
            raise ValueError('Owned controller identity differs')
        write(directory/'started.json', dict(pid=os.getpid(), controller_pid=pid,
            time_ns=time.time_ns(), manifest_sha256=sha(directory/'manifest.json'),
            mechanism='blocking_kernel_pidfd_exit_event', progress_polling=False))
        print(json.dumps(dict(status='waiting_for_owned_controller_exit', watcher_pid=os.getpid(),
                              controller_pid=pid)), flush=True)
        select.select([fd], [], [])
    finally:
        os.close(fd)
    write(directory/'controller-exit-event.json', dict(time_ns=time.time_ns(), controller_pid=pid,
        kernel_exit_event=True, supervisor_tool_exit_code_pending=True))
    print(json.dumps(dict(status='owned_controller_exit_event_received', controller_pid=pid)), flush=True)
    for p,h in manifest['files'].items():
        if sha(p) != h:
            raise ValueError('Frozen followup input changed before audit: '+p)
    root = Path(manifest['scientific_run'])
    terminal = read(root/'execution/terminal.json')
    if terminal.get('status') != 'passed':
        raise ValueError('Original controller failed; no automatic retry or repair')
    output = root/'execution/independent-terminal-audit.json'
    if output.exists():
        raise FileExistsError('Independent audit destination already consumed')
    with (directory/'audit.log').open('x') as log:
        child = subprocess.Popen([sys.executable, manifest['auditor'], '--root', str(root),
            '--output', str(output)], cwd=manifest['main_root'], env=os.environ.copy(),
            stdout=log, stderr=subprocess.STDOUT)
        write(directory/'audit-start.json', dict(pid=child.pid, time_ns=time.time_ns()))
        _, status, usage = os.wait4(child.pid, 0)
        code = os.waitstatus_to_exitcode(status); child.returncode = code
    peak = usage.ru_maxrss/1024
    write(directory/'audit-terminal.json', dict(exit_code=code, peak_rss_mib=peak,
        completed_ns=time.time_ns(), automatic_retry=False))
    if code != 0 or peak > 1536:
        raise ValueError('Independent auditor failed or exceeded original RSS gate')
    receipt = read(output)
    if receipt['status'] != 'passed' or receipt['auditor_sha256'] != sha(manifest['auditor']):
        raise ValueError('Independent audit identity differs')
    write(directory/'complete.json', dict(status='independent_audit_passed_supervisor_tool_exit_pending',
        actual_audit_exit_code=code, audit_sha256=sha(output), manifest_sha256=sha(directory/'manifest.json'),
        original_terminal_sha256=sha(root/'execution/terminal.json'), new_fits=0, new_packages=0,
        supervisor_tool_exit_code_pending=True))
    print(json.dumps(dict(status='independent_terminal_audit_passed', actual_audit_exit_code=code,
        supervisor_tool_exit_code_pending=True)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', required=True); a = p.parse_args()
    if sys.version_info[:2] != (3,12) or any(os.environ.get(k) != '1' for k in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise ValueError('Locked Python3.12 and numerical threads=1 required')
    try:
        run(a.directory)
    except BaseException as error:
        write(Path(a.directory)/'failure.json', dict(error=repr(error), time_ns=time.time_ns(), automatic_retry=False))
        raise
