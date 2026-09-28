"""V27 resource and promotion decisions, importable without torch."""
from __future__ import annotations
import math
import numpy as np
from .v5_resolution import paired_summary

def resource_decision(measurements, available_ram_mib, spec):
    if set(measurements) != {"GLOBAL", "INSTANCE"}:
        raise ValueError("Both resource arms required")
    for row in measurements.values():
        if any(not math.isfinite(float(row[k])) or float(row[k]) <= 0
               for k in ("peak_rss_mib", "train_step_p95_seconds", "validation_forward_p95_seconds")):
            raise ValueError("Positive finite resource measurements required")
    peak = max(r["peak_rss_mib"] for r in measurements.values())
    step = max(r["train_step_p95_seconds"] for r in measurements.values())
    forward = max(r["validation_forward_p95_seconds"] for r in measurements.values())
    projected = 40 / 4 * 1.5 * 250 * (
        (math.ceil(2204 / 256) + math.ceil(1764 / 256)) * step
        + math.ceil(441 / 256) * forward)
    checks = {
        "worker_rss": peak <= spec["peak_worker_rss_mib_max"],
        "available_ram": math.isfinite(available_ram_mib) and
            spec["workers_max"] * peak + spec["available_ram_margin_mib"] <= available_ram_mib,
        "development_wall": projected <= spec["development_projected_wall_seconds_max"],
    }
    return {"passed": all(checks.values()), "checks": checks,
            "projected_development_seconds": projected,
            "peak_worker_rss_mib": peak, "available_ram_mib": available_ram_mib}

def select_finalist(records):
    expected = {(t,a) for t in ("tap_iron","tap_time_len") for a in ("GLOBAL","INSTANCE")}
    rows = {(r["target"],r["arm"]):r for r in records}
    if len(records) != 4 or set(rows) != expected:
        raise ValueError("Complete paired development records required")
    eligible=[]
    for i,target in enumerate(("tap_iron","tap_time_len")):
        values=[]
        for arm in ("GLOBAL","INSTANCE"):
            gains=rows[target,arm]["seed_gains"]
            if set(gains) != {"42","3407"}:
                raise ValueError("Both development seeds required")
            g=np.asarray([gains["42"],gains["3407"]],dtype=float)
            if not np.isfinite(g).all():
                raise ValueError("Finite development gains required")
            values.append(g)
        control,candidate=values
        mean=float(candidate.mean())
        if (candidate>0).all() and mean>=.01 and mean>float(control.mean()):
            eligible.append((-mean,i,target))
    return min(eligible)[2] if eligible else None

def promotion_decision(seed_gains, development_scores):
    if set(seed_gains) != {"42","3407","7777","12011"}:
        raise ValueError("Exactly four split seeds required")
    gains=np.asarray([seed_gains[str(s)] for s in (42,3407,7777,12011)],dtype=float)
    scores=np.asarray(development_scores,dtype=float)
    if scores.shape!=(2,) or not np.isfinite(gains).all() or not np.isfinite(scores).all():
        raise ValueError("Finite gains and two development scores required")
    stats=paired_summary(gains)
    local_gate=bool(scores.mean()>=96.25)
    return {"promoted":bool((gains>0).all() and stats["lcb95"]>0 and local_gate),
            "lcb95":stats["lcb95"],"paired_summary":stats,"local_working_gate_met":local_gate}
