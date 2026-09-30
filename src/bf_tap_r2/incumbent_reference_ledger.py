"""Exclusive, append-only fit reservations for the incumbent reference completion.

This module performs no training, scheduling, data loading or platform action.
File locks serialize count-and-reserve, including between processes. A crashed
start remains an incomplete reservation and consumes its budget permanently.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    payload = canonical(value)  # Validate before creating an incomplete file.
    with Path(path).open("xb") as handle:
        handle.write(payload + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


class ReservationLedger:
    def __init__(self, root, policy_sha256, limits):
        self.root = Path(root)
        self.policy_sha256 = policy_sha256
        self.limits = dict(limits)

    @classmethod
    def create(cls, root, limits):
        if (not limits or not set(limits) <= {"estimator", "optimizer"}
                or any(type(v) is not int or v < 0 for v in limits.values())):
            raise ValueError("invalid reservation budget")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        (root / "events").mkdir()
        (root / "lock").touch(exist_ok=False)
        write_new(root / "policy.json", {"version": 1, "limits": limits})
        return cls.open(root, file_hash(root / "policy.json"))

    @classmethod
    def open(cls, root, expected_sha256):
        root = Path(root)
        if file_hash(root / "policy.json") != expected_sha256:
            raise ValueError("reservation policy identity mismatch")
        policy = json.loads((root / "policy.json").read_text())
        if policy.get("version") != 1:
            raise ValueError("unknown reservation policy")
        return cls(root, expected_sha256, policy["limits"])

    @contextmanager
    def event(self, kind, key, payload):
        if kind not in self.limits:
            raise ValueError("unknown reservation kind")
        identity = {"kind": kind, "key": list(key)}
        token = hashlib.sha256(canonical(identity)).hexdigest()
        prefix = self.root / "events" / f"{kind}-{token}"
        started = prefix.with_suffix(".started.json")
        record = {**identity, "payload": payload, "policy_sha256": self.policy_sha256,
                  "started_ns": time.time_ns()}
        canonical(record)
        with (self.root / "lock").open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if file_hash(self.root / "policy.json") != self.policy_sha256:
                raise ValueError("reservation policy changed")
            if started.exists():
                raise FileExistsError(f"reservation already consumed: {kind} {key}")
            consumed = len(list((self.root / "events").glob(f"{kind}-*.started.json")))
            if consumed >= self.limits[kind]:
                raise ValueError(f"{kind} budget exhausted")
            write_new(started, record)
            start_hash = file_hash(started)
        result = {}
        try:
            yield result
            completion = {"start_sha256": start_hash, "result": result, "ended_ns": time.time_ns()}
            write_new(prefix.with_suffix(".complete.json"), completion)
        except BaseException as exc:
            # Never delete, retry or mark this start successful on failure.
            write_new(prefix.with_suffix(".failed.json"), {
                "start_sha256": start_hash, "type": type(exc).__name__,
                "message": str(exc), "ended_ns": time.time_ns(),
            })
            raise

    def scoped(self, scope):
        scope = tuple(scope)
        return lambda kind, state, payload: self.event(kind, (*scope, kind, state), payload)

    def inspect(self):
        if file_hash(self.root / "policy.json") != self.policy_sha256:
            raise ValueError("reservation policy identity mismatch")
        counts = {status: dict.fromkeys(self.limits, 0)
                  for status in ("started", "completed", "failed", "incomplete")}
        seen = set()
        for path in sorted((self.root / "events").glob("*.started.json")):
            seen.add(path.name)
            record = json.loads(path.read_text())
            kind = record["kind"]
            identity = {"kind": kind, "key": record["key"]}
            token = hashlib.sha256(canonical(identity)).hexdigest()
            if (kind not in self.limits or path.name != f"{kind}-{token}.started.json"
                    or record["policy_sha256"] != self.policy_sha256):
                raise ValueError("reservation identity mismatch")
            counts["started"][kind] += 1
            endings = []
            for suffix, status in (("complete", "completed"), ("failed", "failed")):
                endpoint = path.with_name(path.name.replace("started.json", suffix + ".json"))
                if endpoint.exists():
                    seen.add(endpoint.name)
                    receipt = json.loads(endpoint.read_text())
                    if receipt["start_sha256"] != file_hash(path):
                        raise ValueError("reservation receipt identity mismatch")
                    counts[status][kind] += 1
                    endings.append(status)
            if len(endings) > 1:
                raise ValueError("reservation has multiple terminal receipts")
            if not endings:
                counts["incomplete"][kind] += 1
        if {p.name for p in (self.root / "events").iterdir()} != seen:
            raise ValueError("orphan or unexpected reservation artifact")
        if any(counts["started"][k] > self.limits[k] for k in self.limits):
            raise ValueError("reservation budget exceeded")
        return counts
