"""Torch-free resource admission for frozen V28_EDGE_KAN."""
import math


def resource_decision(measurements, available_ram_mib):
    if set(measurements) != {'EDGE', 'SHARED'}:
        raise ValueError('Both resource arms required')
    for row in measurements.values():
        if any(not math.isfinite(float(row[k])) or row[k] <= 0 for k in (
                'peak_rss_mib', 'train_step_p95_seconds', 'validation_forward_p95_seconds')):
            raise ValueError('Positive finite measurements required')
    peak = max(row['peak_rss_mib'] for row in measurements.values())
    step = max(row['train_step_p95_seconds'] for row in measurements.values())
    forward = max(row['validation_forward_p95_seconds'] for row in measurements.values())
    projected = 40 / 4 * 1.5 * 240 * (16 * step + 2 * forward)
    checks = {'worker_rss': peak <= 1024,
        'available_ram': math.isfinite(available_ram_mib) and 4 * peak + 1024 <= available_ram_mib,
        'development_wall': projected <= 7200}
    return {'passed': all(checks.values()), 'checks': checks,
        'projected_development_seconds': projected, 'peak_worker_rss_mib': peak,
        'available_ram_mib': available_ram_mib}

def select_finalist(records):
    import numpy as np
    targets, arms = ('tap_iron', 'tap_time_len'), ('SHARED', 'EDGE')
    rows = {(r['target'], r['arm']): r for r in records}
    if len(records) != 4 or set(rows) != {(t,a) for t in targets for a in arms}:
        raise ValueError('Complete paired development required')
    eligible = []
    for i, target in enumerate(targets):
        values = []
        for arm in arms:
            gains = rows[target, arm]['seed_gains']
            if set(gains) != {'42', '3407'}:
                raise ValueError('Both development seeds required')
            g = np.array([gains['42'], gains['3407']], dtype=float)
            if not np.isfinite(g).all():
                raise ValueError('Finite gains required')
            values.append(g)
        control, candidate = values
        mean = float(candidate.mean())
        if (candidate > 0).all() and mean >= .01 and mean > float(control.mean()):
            eligible.append((-mean, i, target))
    return min(eligible)[2] if eligible else None


def promotion_decision(seed_gains, development_scores):
    from .v27_protocol import promotion_decision as four_seed_decision
    return four_seed_decision(seed_gains, development_scores)


def require_audited_development(directory, identity):
    import json
    from .v7_periodic import file_hash
    summary = json.loads((directory/'summary.json').read_text())
    audit = json.loads((directory/'audit-r1.json').read_text())
    if (audit.get('status') != 'passed' or audit.get('fits') != 40
            or summary.get('fits') != 40 or summary.get('failed_fits') != 0):
        raise ValueError('Complete audited development required')
    if audit.get('summary_sha256') != file_hash(directory/'summary.json'):
        raise ValueError('Audit summary hash mismatch')
    if any(audit.get(k) != v for k,v in identity.items()):
        raise ValueError('Audit frozen identity hash mismatch')
    target = select_finalist(summary['records'])
    if target is None or target != summary['selected_for_confirmation']:
        raise ValueError('Recomputed eligible target required')
    return target