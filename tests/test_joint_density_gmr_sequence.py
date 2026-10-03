"""Mock-only GMR scheduling, race, identity, and native-terminal tests."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = Path(__file__).resolve().parents[1]/"scripts/run_joint_density_gmr_sequence.py"
_SPEC = importlib.util.spec_from_file_location("gmr_sequence_under_test", _SOURCE)
controller = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(controller)


class BoundChild:
    def __init__(self, outcomes, pid=123456):
        self.pid = pid
        self.outcomes = list(outcomes)
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if not self.outcomes:
            raise AssertionError("Unexpected wait or automatic retry")
        value = self.outcomes.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.fixture(autouse=True)
def forbid_real_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not launch or inspect real processes")
    monkeypatch.setattr(controller.subprocess, "Popen", forbidden)
    monkeypatch.setattr(controller.psutil, "Process", forbidden)


@pytest.mark.parametrize("code", [0, 7, -9])
def test_actual_code_preserved_without_unscheduled_health(code):
    child, health = BoundChild([code]), []
    assert controller.wait_with_health(child, lambda *event: health.append(event)) == code
    assert child.wait_timeouts == [600]
    assert health == [] and child.outcomes == []


def test_two_health_intervals_remain_six_hundred_seconds(monkeypatch):
    child = BoundChild([controller.subprocess.TimeoutExpired("fake", 600),
                        controller.subprocess.TimeoutExpired("fake", 600), 0])
    inspected, health = [], []

    def fake_state(pid):
        inspected.append(pid)
        return SimpleNamespace(status=lambda: "running",
            memory_info=lambda: SimpleNamespace(rss=128*1024**2), num_threads=lambda: 1)

    monkeypatch.setattr(controller.psutil, "Process", fake_state)
    monkeypatch.setattr(controller.time, "time_ns", lambda: 99)
    assert controller.wait_with_health(child, lambda *event: health.append(event)) == 0
    assert child.wait_timeouts == [600, 600, 600]
    assert inspected == [child.pid, child.pid]
    assert [event[0] for event in health] == [1, 2]
    assert all(event[1] == dict(pid=child.pid, utc_ns=99, status="running", rss_mib=128.,
                              threads=1, automatic_action="none") for event in health)


@pytest.mark.parametrize("code", [0, 9])
@pytest.mark.parametrize("vanishes_at", ["construction", "status", "memory", "threads"])
def test_vanished_child_returns_only_same_bound_process_exit(monkeypatch, code, vanishes_at):
    child = BoundChild([controller.subprocess.TimeoutExpired("fake", 600), code])
    inspected, health = [], []

    def missing():
        raise controller.psutil.NoSuchProcess(child.pid)

    def fake_state(pid):
        inspected.append(pid)
        if vanishes_at=="construction":
            missing()
        return SimpleNamespace(status=missing if vanishes_at=="status" else lambda: "running",
            memory_info=missing if vanishes_at=="memory" else lambda: SimpleNamespace(rss=1),
            num_threads=missing if vanishes_at=="threads" else lambda: 1)

    monkeypatch.setattr(controller.psutil, "Process", fake_state)
    assert controller.wait_with_health(child, lambda *event: health.append(event)) == code
    assert child.wait_timeouts == [600, None]
    assert inspected == [child.pid] and health == [] and child.outcomes == []


def test_inspection_permission_error_does_not_become_success(monkeypatch):
    child = BoundChild([controller.subprocess.TimeoutExpired("fake", 600)])

    def denied(pid):
        raise controller.psutil.AccessDenied(pid)

    monkeypatch.setattr(controller.psutil, "Process", denied)
    with pytest.raises(controller.psutil.AccessDenied):
        controller.wait_with_health(child, lambda *event: None)
    assert child.wait_timeouts == [600]


def test_health_evidence_writer_failure_is_not_retried(monkeypatch):
    child = BoundChild([controller.subprocess.TimeoutExpired("fake", 600)])
    monkeypatch.setattr(controller.psutil, "Process", lambda pid: SimpleNamespace(
        status=lambda: "running", memory_info=lambda: SimpleNamespace(rss=1), num_threads=lambda: 1))

    def fail(*args):
        raise OSError("fake immutable evidence failure")

    with pytest.raises(OSError, match="immutable evidence"):
        controller.wait_with_health(child, fail)
    assert child.wait_timeouts == [600]


class MemoryFile(io.BytesIO):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def close(self):
        if not self.closed:
            self.path.storage[self.path.name] = self.getvalue()
        super().close()


class MemoryPath:
    """Only in-memory files/directories; no filesystem access."""
    def __init__(self, name, storage, directories):
        self.name, self.storage, self.directories = name, storage, directories

    def __truediv__(self, value):
        return MemoryPath(self.name+"/"+str(value), self.storage, self.directories)

    def __str__(self):
        return self.name

    def resolve(self):
        return self

    def mkdir(self, *, parents, exist_ok):
        assert parents is True and exist_ok is False
        if self.name in self.directories:
            raise FileExistsError(self.name)
        self.directories.add(self.name)

    def read_text(self, *, encoding):
        assert encoding=="utf-8"
        return self.storage[self.name].decode(encoding)

    def open(self, mode):
        assert mode=="xb"
        if self.name in self.storage:
            raise FileExistsError(self.name)
        self.storage[self.name] = None
        return MemoryFile(self)


@pytest.fixture
def memory_sequence(monkeypatch):
    storage, directories, writes, launches, inventory_calls = {}, set(), {}, [], []
    root = MemoryPath("virtual-root", storage, directories)
    storage[str(root/controller.SPEC)] = json.dumps({"identity":"fake-frozen-spec"}).encode()
    state = dict(codes=[0, 0, 0, 0], changed_at=None, missing_at=None)

    def fake_inventory(path):
        assert path is root
        index = len(inventory_calls)
        inventory_calls.append(index)
        if state["missing_at"]==index:
            raise FileNotFoundError("missing frozen source")
        return {"science.py":"changed" if state["changed_at"]==index else "frozen"}

    def fake_write_new(path, value):
        name = str(path)
        if name in storage:
            raise FileExistsError(name)
        writes[name] = deepcopy(value)
        storage[name] = json.dumps(value).encode()

    def fake_popen(command, *, cwd, stdout, stderr):
        assert cwd is root
        index = len(launches)
        child = BoundChild([state["codes"][index]], pid=10000+index)
        launches.append(dict(command=list(command), child=child))
        stdout.write(b"mock stdout")
        stderr.write(b"mock stderr")
        return child

    def fake_local_output(path, value):
        assert path is root
        if not value.startswith("local/"):
            raise ValueError("private local only")
        return root/value

    monkeypatch.setattr(controller, "Path", SimpleNamespace(cwd=lambda:root))
    monkeypatch.setattr(controller, "inventory", fake_inventory)
    monkeypatch.setattr(controller, "validate_spec", lambda value: None if value=={"identity":"fake-frozen-spec"} else (_ for _ in ()).throw(ValueError("wrong spec")))
    monkeypatch.setattr(controller, "write_new", fake_write_new)
    monkeypatch.setattr(controller, "local_output", fake_local_output)
    monkeypatch.setattr(controller, "sha", lambda path:hashlib.sha256(storage[str(path)]).hexdigest())
    monkeypatch.setattr(controller.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(controller.psutil, "Process", lambda:SimpleNamespace(pid=999))
    sequence = str(root/f"{controller.RUN}/sequence-r1")
    return dict(root=root, state=state, storage=storage, directories=directories,
                writes=writes, launches=launches, inventory_calls=inventory_calls, sequence=sequence)


def test_four_phases_are_serial_and_keep_native_commands_and_log_hashes(memory_sequence):
    memory = memory_sequence
    assert controller.main() == 0
    terminal = memory["writes"][memory["sequence"]+"/terminal.json"]
    assert terminal["status"] == "completed_all_four_phases"
    assert terminal["exit_codes"] == [0, 0, 0, 0]
    assert terminal["source_identity_passed_all_phases"] is True
    assert len(memory["launches"]) == 4
    for index, ((action, name), launch, event) in enumerate(zip(controller.PHASES, memory["launches"], terminal["phases"], strict=True)):
        assert launch["command"][:5] == [controller.sys.executable, "-B", "-m", controller.EXPERIMENT, action]
        assert launch["command"][5:7] == ["--output", f"{controller.RUN}/{name}"]
        assert ("--engineering" in launch["command"]) is (action=="development")
        assert event["pid"] == launch["child"].pid
        assert event["command"] == launch["command"]
        assert event["stdout_sha256"] == hashlib.sha256(b"mock stdout").hexdigest()
        assert event["stderr_sha256"] == hashlib.sha256(b"mock stderr").hexdigest()
        assert event["source_check_before"]["matches"] is True
        assert event["source_check_after"]["matches"] is True
        assert launch["child"].wait_timeouts == [600]
        assert memory["sequence"]+f"/{index}-{action}-started.json" in memory["writes"]
    assert len(memory["inventory_calls"]) == 9
    assert terminal["packages"] == terminal["agent_uploads"] == 0


@pytest.mark.parametrize("phase", range(4))
def test_each_child_failure_preserves_actual_exit_and_stops_without_retry(memory_sequence, phase):
    memory_sequence["state"]["codes"][phase] = 7
    assert controller.main() == 7
    terminal = memory_sequence["writes"][memory_sequence["sequence"]+"/terminal.json"]
    assert terminal["status"] == "failed_child_phase"
    assert terminal["exit_codes"] == [0]*phase+[7]
    assert len(memory_sequence["launches"]) == phase+1
    assert terminal["automatic_retry"] is False


@pytest.mark.parametrize("changed_at,children", [(1, 0), (2, 1), (8, 4)])
def test_source_change_never_erases_zero_child_exit_even_after_final_phase(memory_sequence, changed_at, children):
    memory_sequence["state"]["changed_at"] = changed_at
    assert controller.main() == 1
    terminal = memory_sequence["writes"][memory_sequence["sequence"]+"/terminal.json"]
    assert terminal["status"] == "controller_failed_source_identity"
    assert terminal["controller_exit_code"] == 1
    assert terminal["exit_codes"] == [0]*children
    assert len(memory_sequence["launches"]) == children
    if children:
        assert terminal["phases"][-1]["exit_code"] == 0
        assert terminal["phases"][-1]["source_check_after"]["matches"] is False
    assert terminal["automatic_retry"] is False


def test_disappearing_source_is_recorded_separately_from_zero_native_exit(memory_sequence):
    memory_sequence["state"]["missing_at"] = 2
    assert controller.main() == 1
    terminal = memory_sequence["writes"][memory_sequence["sequence"]+"/terminal.json"]
    assert terminal["exit_codes"] == [0]
    assert terminal["failure"]["details"]["actual_child_exit_code"] == 0
    assert terminal["phases"][0]["source_check_after"]["source_read_error_type"] == "FileNotFoundError"


def test_existing_sequence_is_not_overwritten_or_restarted(memory_sequence):
    memory_sequence["directories"].add(memory_sequence["sequence"])
    with pytest.raises(FileExistsError):
        controller.main()
    assert memory_sequence["launches"] == [] and memory_sequence["writes"] == {}
