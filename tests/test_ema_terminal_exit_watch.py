"""Real Linux process-exit orchestration on synthetic files only; zero fits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

path = Path(__file__).resolve().parents[1]/'scripts/watch_ema_independent_terminal.py'
spec = importlib.util.spec_from_file_location('terminal_exit_watch', path)
watch = importlib.util.module_from_spec(spec); spec.loader.exec_module(watch)


@pytest.mark.parametrize('controller_passed,auditor_code', [(True, 0), (False, 0), (True, 3)])
def test_waits_for_real_exit_then_runs_only_one_allowed_audit(tmp_path, controller_passed, auditor_code):
    execution = tmp_path/'science/execution'; execution.mkdir(parents=True)
    (execution/'terminal.json').write_text(json.dumps({'status':'passed' if controller_passed else 'failed'}))
    auditor = tmp_path/'synthetic_auditor.py'
    auditor.write_text('''from pathlib import Path
import hashlib,json,sys
output=Path(sys.argv[sys.argv.index('--output')+1])
if CODE: raise SystemExit(CODE)
output.write_text(json.dumps(dict(status='passed',auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
'''.replace('CODE', str(auditor_code)))
    d = tmp_path/'followup'; d.mkdir()
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(1)'], cwd=tmp_path)
    try:
        cmd = Path('/proc', str(child.pid), 'cmdline').read_bytes()
        manifest = dict(files={str(auditor):hashlib.sha256(auditor.read_bytes()).hexdigest()},
            controller_pid=child.pid, controller_cmdline_hex=cmd.hex(), controller_worktree=str(tmp_path),
            scientific_run=str(tmp_path/'science'), auditor=str(auditor), main_root=str(tmp_path))
        (d/'manifest.json').write_text(json.dumps(manifest))
        if not controller_passed:
            with pytest.raises(ValueError, match='Original controller failed'):
                watch.run(d)
            assert not (d/'audit-start.json').exists()
        elif auditor_code:
            with pytest.raises(ValueError, match='auditor failed'):
                watch.run(d)
            assert watch.read(d/'audit-terminal.json')['exit_code'] == auditor_code
            assert not (d/'complete.json').exists()
        else:
            watch.run(d)
            result = watch.read(d/'complete.json')
            assert result['actual_audit_exit_code'] == 0
            assert result['supervisor_tool_exit_code_pending'] is True
            assert (execution/'independent-terminal-audit.json').exists()
        assert child.wait(timeout=5) == 0
        assert watch.read(d/'controller-exit-event.json')['kernel_exit_event'] is True
    finally:
        child.wait(timeout=5)
