"""Mock-only controller checks: never launch, inspect, kill, or fit a process."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = Path(__file__).resolve().parents[1]/"scripts/run_laplace_leaf_partition_sequence.py"
_SPEC = importlib.util.spec_from_file_location("leaf_partition_controller_under_test", _SOURCE)
controller = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(controller)


class BoundChild:
    pid = 123456

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if not self.outcomes:
            raise AssertionError("Unexpected child wait or retry")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def forbid_real_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not launch or inspect real processes")
    monkeypatch.setattr(controller.subprocess, "Popen", forbidden)
    monkeypatch.setattr(controller.psutil, "Process", forbidden)


@pytest.mark.parametrize("code", [0, 7])
def test_normal_exit_preserves_actual_code_without_health(code):
    child = BoundChild([code])
    health = []
    assert controller.wait_with_health(child, lambda *item: health.append(item)) == code
    assert child.wait_timeouts == [600]
    assert health == []
    assert child.outcomes == []


def test_six_hundred_second_timeout_reports_one_health_then_exits(monkeypatch):
    child = BoundChild([controller.subprocess.TimeoutExpired("bound-child", 600), 0])
    observed_pids = []
    state = SimpleNamespace(status=lambda: "running",
                            memory_info=lambda: SimpleNamespace(rss=128*1024**2),
                            num_threads=lambda: 1)

    def inspect_bound_child(pid):
        observed_pids.append(pid)
        return state

    monkeypatch.setattr(controller.psutil, "Process", inspect_bound_child)
    monkeypatch.setattr(controller.time, "time_ns", lambda: 99)
    health = []
    assert controller.wait_with_health(child, lambda *item: health.append(item)) == 0
    assert child.wait_timeouts == [600, 600]
    assert observed_pids == [child.pid]
    assert health == [(1, dict(pid=child.pid, utc_ns=99, status="running",
                              rss_mib=128.0, threads=1, automatic_action="none"))]


@pytest.mark.parametrize("code", [0, 9])
@pytest.mark.parametrize("vanishes_at", ["construction", "status", "memory", "threads"])
def test_disappearing_child_gets_actual_exit_from_same_popen(monkeypatch, code, vanishes_at):
    child = BoundChild([controller.subprocess.TimeoutExpired("bound-child", 600), code])
    inspected = []

    def missing():
        raise controller.psutil.NoSuchProcess(child.pid)

    def inspect_bound_child(pid):
        inspected.append(pid)
        if vanishes_at == "construction":
            missing()
        return SimpleNamespace(
            status=missing if vanishes_at == "status" else lambda: "running",
            memory_info=missing if vanishes_at == "memory" else lambda: SimpleNamespace(rss=1),
            num_threads=missing if vanishes_at == "threads" else lambda: 1,
        )

    monkeypatch.setattr(controller.psutil, "Process", inspect_bound_child)
    health = []
    assert controller.wait_with_health(child, lambda *item: health.append(item)) == code
    assert child.wait_timeouts == [600, None]
    assert inspected == [child.pid]
    assert health == []
    assert child.outcomes == []


def test_multiple_scheduled_checks_keep_six_hundred_second_interval(monkeypatch):
    child = BoundChild([controller.subprocess.TimeoutExpired("bound-child", 600),
                        controller.subprocess.TimeoutExpired("bound-child", 600), 3])
    inspected = []

    def inspect_bound_child(pid):
        inspected.append(pid)
        return SimpleNamespace(status=lambda: "running",
                               memory_info=lambda: SimpleNamespace(rss=2*1024**2),
                               num_threads=lambda: 1)

    monkeypatch.setattr(controller.psutil, "Process", inspect_bound_child)
    health = []
    assert controller.wait_with_health(child, lambda *item: health.append(item)) == 3
    assert child.wait_timeouts == [600, 600, 600]
    assert inspected == [child.pid, child.pid]
    assert [item[0] for item in health] == [1, 2]
    assert all(item[1]["automatic_action"] == "none" for item in health)
    assert child.outcomes == []


def test_unexpected_inspection_error_is_not_synthetic_success(monkeypatch):
    child = BoundChild([controller.subprocess.TimeoutExpired("bound-child", 600)])

    def denied(pid):
        raise controller.psutil.AccessDenied(pid)

    monkeypatch.setattr(controller.psutil, "Process", denied)
    health = []
    with pytest.raises(controller.psutil.AccessDenied):
        controller.wait_with_health(child, lambda *item: health.append(item))
    assert child.wait_timeouts == [600]
    assert health == []


def test_health_writer_failure_is_not_hidden_or_retried(monkeypatch):
    child = BoundChild([controller.subprocess.TimeoutExpired("bound-child", 600)])
    monkeypatch.setattr(controller.psutil, "Process", lambda pid: SimpleNamespace(
        status=lambda: "running", memory_info=lambda: SimpleNamespace(rss=1),
        num_threads=lambda: 1))

    def cannot_record(*args):
        raise OSError("immutable health evidence cannot be recorded")

    with pytest.raises(OSError, match="immutable health evidence"):
        controller.wait_with_health(child, cannot_record)
    assert child.wait_timeouts == [600]
