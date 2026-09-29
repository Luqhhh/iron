from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from bf_tap_r2.rfm_protocol import ReservationLedger, phase_tasks, phase_limits


def test_exact_phase_budgets_and_target_restriction():
    tasks = phase_tasks("development")
    assert len(tasks) == 40
    assert len({tuple(sorted(t.items())) for t in tasks}) == 40
    assert {t["seed"] for t in tasks} == {42, 3407}
    assert phase_limits("development") == {"outer_fit": 40, "procedure": 80, "solve": 200, "update": 120}
    assert len(phase_tasks("confirmation", ["tap_time_len"])) == 20
    assert phase_limits("confirmation", ["tap_time_len"])["solve"] == 100
    assert phase_limits("confirmation", []) == dict.fromkeys(phase_limits("development"), 0)
    with pytest.raises(ValueError):
        phase_tasks("confirmation", ["tap_iron", "tap_iron"])
    with pytest.raises(ValueError):
        phase_tasks("confirmation", ["other"])
    with pytest.raises(ValueError):
        phase_tasks("confirmation")


def test_failure_is_retained_and_consumes_budget(tmp_path):
    ledger = ReservationLedger.create(tmp_path / "ledger", {"solve": 1})
    with pytest.raises(RuntimeError, match="solver failure"):
        with ledger.event("solve", ("a", 0), {"rows": 10}):
            raise RuntimeError("solver failure")
    with pytest.raises(FileExistsError):
        with ledger.event("solve", ("a", 0), {}):
            pytest.fail("retry entered")
    with pytest.raises(ValueError, match="budget"):
        with ledger.event("solve", ("b", 0), {}):
            pytest.fail("over-budget solve entered")
    report = ledger.inspect()
    assert report["started"] == {"solve": 1}
    assert report["failed"] == {"solve": 1}
    assert report["completed"] == {"solve": 0}
    assert report["incomplete"] == {"solve": 0}
    assert len(list((tmp_path / "ledger" / "events").glob("*.failed.json"))) == 1


def test_concurrent_duplicate_and_distinct_events_are_atomic(tmp_path):
    ledger = ReservationLedger.create(tmp_path / "ledger", {"solve": 1})
    def attempt(key):
        cold = ReservationLedger.open(ledger.root, ledger.policy_sha256)
        try:
            with cold.event("solve", key, {}) as result:
                result["answer"] = 42
            return "completed"
        except (FileExistsError, ValueError):
            return "refused"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, [("same", 0)] * 3 + [("different", 0)]))
    assert results.count("completed") == 1
    assert ledger.inspect()["started"] == {"solve": 1}
    assert ledger.inspect()["completed"] == {"solve": 1}


def test_existing_directory_policy_and_receipt_tamper_rejected(tmp_path):
    path = tmp_path / "ledger"
    ledger = ReservationLedger.create(path, {"solve": 2})
    with pytest.raises(FileExistsError):
        ReservationLedger.create(path, {"solve": 3})
    with pytest.raises(ValueError, match="policy"):
        ReservationLedger.open(path, "0" * 64)
    with ledger.event("solve", ("a", 0), {}) as result:
        result["residual"] = 0.
    started = next((path / "events").glob("*.started.json"))
    data = json.loads(started.read_text())
    data["payload"]["forged"] = True
    started.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        ledger.inspect()


def test_unknown_events_and_nonfinite_payloads_rejected(tmp_path):
    ledger = ReservationLedger.create(tmp_path / "ledger", {"solve": 2})
    with pytest.raises(ValueError):
        with ledger.event("optimizer", (0,), {}):
            pass
    with pytest.raises(ValueError):
        with ledger.event("solve", (0,), {"bad": float("nan")}):
            pass
    assert ledger.inspect()["started"] == {"solve": 0}


def test_orphan_receipts_do_not_count_as_completed_work(tmp_path):
    ledger = ReservationLedger.create(tmp_path / "ledger", {"solve": 2})
    (ledger.root / "events" / "orphan.complete.json").write_text('{}')
    with pytest.raises(ValueError, match="orphan"):
        ledger.inspect()
