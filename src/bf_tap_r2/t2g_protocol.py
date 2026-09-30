"""Durable Linux reservations: crashes consume, never retry a claimed stage."""
from dataclasses import asdict,dataclass
from pathlib import Path
import fcntl
import json
import os
from .t2g_model import canonical

TARGETS=("tap_iron","tap_time_len")
ARMS=("LEARNED_GRAPH","DENSE_CONTROL")
SEEDS={"development":(42,3407),"confirmation":(7777,12011)}

@dataclass(frozen=True)
class UnitKey:
    phase: str
    target: str
    arm: str
    seed: int
    fold: int

def expected_units(phase,targets):
    if phase not in SEEDS or not targets or len(set(targets))!=len(targets) or not set(targets)<=set(TARGETS):
        raise ValueError("Invalid phase/selected targets")
    if phase=="development" and set(targets)!=set(TARGETS):
        raise ValueError("Complete two-target development required")
    return [UnitKey(phase,t,a,s,f) for t in targets for a in ARMS for s in SEEDS[phase] for f in range(5)]

def read_events(root):
    p=Path(root)/"reservations.jsonl"
    if not p.exists(): return []
    return [json.loads(line) for line in p.read_text().splitlines()]

def inspect_ledger(root):
    rows=[r for r in read_events(root) if r["event"]=="reserved"]
    ids=[(canonical(r["key"]),r["stage"]) for r in rows]
    if len(set(ids))!=len(ids): raise ValueError("Duplicate durable reservation")
    return {"outer_fits":sum(r["stage"]=="selector" for r in rows),
        "optimizer_starts":len(rows),"reservations":rows}

def append_locked(root,row):
    with (Path(root)/"reservations.jsonl").open("a") as f:
        f.write(canonical(row)+"\n"); f.flush(); os.fsync(f.fileno())

def reserve(root,key,stage,manifest_digest):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if stage not in ("selector","refit") or len(manifest_digest)!=64 or key.phase not in SEEDS:
        raise ValueError("Invalid stage/manifest/phase")
    if key not in expected_units(key.phase,list(TARGETS) if key.phase=="development" else [key.target]):
        raise ValueError("Foreign reservation")
    with (root/"reservation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=read_events(root)
        reserved=[r for r in rows if r["event"]=="reserved"]
        if any(r["manifest_digest"]!=manifest_digest or r["key"]["phase"]!=key.phase for r in reserved):
            raise ValueError("Reservation manifest/phase differs")
        same=[r for r in reserved if r["key"]==asdict(key)]
        if any(r["stage"]==stage for r in same): raise ValueError("Already consumed; retries prohibited")
        if stage=="refit" and not any(r["stage"]=="selector" for r in same):
            raise ValueError("Refit without consumed selector")
        policy_path=root/"policy.json"
        if policy_path.exists():
            policy=json.loads(policy_path.read_text())
            if (policy["phase"]!=key.phase or policy["manifest_digest"]!=manifest_digest
                or key not in expected_units(policy["phase"],policy["targets"])):
                raise ValueError("Target is outside frozen phase policy")
            max_optim=2*len(expected_units(policy["phase"],policy["targets"]))
        else: max_optim=80
        if len(reserved)>=max_optim: raise ValueError("Phase optimizer budget exhausted")
        append_locked(root,{"event":"reserved","key":asdict(key),"stage":stage,"manifest_digest":manifest_digest})

def outcome(root,key,event,details):
    if event not in ("completed","failed"): raise ValueError("Invalid unit outcome")
    root=Path(root)
    with (root/"reservation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=read_events(root)
        if any(r["key"]==asdict(key) and r["event"] in ("completed","failed") for r in rows):
            raise ValueError("Unit already terminal")
        append_locked(root,{"event":event,"key":asdict(key),"details":details})


def initialize_ledger(root,phase,targets,manifest_digest):
    expected_units(phase,targets)
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if (root/"reservations.jsonl").exists(): raise FileExistsError("Existing reservations cannot acquire a new policy")
    policy={"phase":phase,"targets":targets,"manifest_digest":manifest_digest}
    with (root/"policy.json").open("x") as f:
        f.write(canonical(policy)+"\n"); f.flush(); os.fsync(f.fileno())
