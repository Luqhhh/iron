"""Fixed-.2 isolated-column scores, causal controls and four-seed gate.

Only independently audited, complete OOF caller artifacts may feed decisions.
No fit/release or average across split seeds is performed here.
"""
import numpy as np

from .data import TARGETS
from .dnnr_model import ARMS
from .dnnr_phase_protocol import DEV, CONFIRM, ELIGIBLE, CONTROLS, validate_eligible, required_arms

LCB_MULTIPLIER = 2.3533634348018264


def wmape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if (y.ndim != 1 or not len(y) or p.shape != y.shape or not np.isfinite(y).all()
            or not np.isfinite(p).all() or np.abs(y).sum() <= 0):
        raise ValueError('Aligned finite nonempty vectors and positive denominator required')
    return float(np.abs(y-p).sum()/np.abs(y).sum())


def score_seed(y, folds, spout, current, parent, members, target, seed):
    if (target not in TARGETS or seed not in (*DEV, *CONFIRM) or set(y) != set(TARGETS)
            or set(current) != set(TARGETS) or set(parent) != set(TARGETS)
            or not members or not set(members) <= set(ARMS)):
        raise ValueError('Complete score definition required')
    n = len(y[target]); folds, spout = np.asarray(folds), np.asarray(spout)
    if (folds.shape != (n,) or not np.issubdtype(folds.dtype, np.integer)
            or set(folds) != set(range(5)) or spout.shape != (n,) or not np.isfinite(spout).all()):
        raise ValueError('Complete aligned fold/spout coverage required')
    base = {t: wmape(y[t], current[t]) for t in TARGETS}
    old = {t: wmape(y[t], parent[t]) for t in TARGETS}
    if any(len(y[t]) != n for t in TARGETS):
        raise ValueError('Target row mismatch')
    other = next(t for t in TARGETS if t != target)
    def detail(p):
        return dict(wmape=wmape(y[target],p),
            by_fold={str(f): wmape(np.asarray(y[target])[folds == f],p[folds == f]) for f in range(5)},
            by_spout={str(int(s)): wmape(np.asarray(y[target])[spout == s],p[spout == s]) for s in sorted(set(spout))})
    records, metrics = [], {'CURRENT_DE3': detail(np.asarray(current[target],float))}
    for arm in ARMS:
        if arm not in members: continue
        raw = np.asarray(members[arm],float)
        if raw.shape != (n,) or not np.isfinite(raw).all():
            raise ValueError('Complete member OOF required; never average split seeds')
        endpoint = .8*np.asarray(current[target])+ .2*raw
        d = detail(endpoint); metrics[arm] = d
        score = 100-50*(d['wmape']+base[other])
        records.append(dict(target=target, arm=arm, seed=seed, rows=n, blend_weight=.2,
            base_score=100-50*sum(base.values()), candidate_score=score,
            gain=50*(base[target]-d['wmape']), target_wmape=d['wmape'],
            standalone_wmape=wmape(y[target],raw), base_target_wmape=base[target],
            parent_target_gain=50*(old[target]-d['wmape']),
            candidate_minus_parent_package=score-(100-50*sum(old.values())),
            other_target_unchanged=True,
            folds_descriptive=[dict(fold=f, gain=50*(metrics['CURRENT_DE3']['by_fold'][str(f)]-d['by_fold'][str(f)])) for f in range(5)]))
    return records, metrics


def _index(records, phase, eligible=None):
    seeds = DEV if phase == 'development' else CONFIRM
    targets = TARGETS if phase == 'development' else tuple(t for t in TARGETS if any(p[0] == t for p in validate_eligible(eligible)))
    expected = {(t,a,s) for t in targets for a in required_arms(t,phase,eligible) for s in seeds}
    index = {}
    for row in records:
        key = row['target'],row['arm'],row['seed']
        if (key not in expected or key in index or row['blend_weight'] != .2
                or row['other_target_unchanged'] is not True
                or not np.isfinite([row['gain'],row['candidate_score']]).all()):
            raise ValueError('Unexpected/duplicate/invalid gate input')
        index[key] = row
    if set(index) != expected:
        raise ValueError('Incomplete gate coverage')
    return index


def decide_development(records):
    index = _index(records,'development')
    decisions, selected = {}, []
    for t in TARGETS:
        decisions[t] = {}
        for a in ELIGIBLE:
            values = [index[t,a,s] for s in DEV]
            gains = np.array([r['gain'] for r in values])
            contrast = gains-np.array([index[t,CONTROLS[a],s]['gain'] for s in DEV])
            checks = dict(both_complete_seeds_positive=bool(min(gains)>0),
                mean_gain_at_least_0_01=bool(gains.mean()>=.01),
                positive_control_advantage=bool(contrast.mean()>0),
                local_package_at_least_96_25=bool(np.mean([r['candidate_score'] for r in values])>=96.25))
            decisions[t][a] = dict(eligible=all(checks.values()),checks=checks,
                failure_reasons=[k for k,v in checks.items() if not v],
                mean_gain=float(gains.mean()),mean_control_advantage=float(contrast.mean()))
            if all(checks.values()): selected.append([t,a])
    return dict(phase='development',decisions=decisions,eligible_pairs=selected,
        release_authorized=False,requires_independent_artifact_audit=True)


def decide_confirmation(development, confirmation):
    selected = decide_development(development)['eligible_pairs']
    index = _index(development,'development')
    index.update(_index(confirmation,'confirmation',selected))
    decisions, promoted = {}, []
    for t,a in validate_eligible(selected):
        gains = np.array([index[t,a,s]['gain'] for s in (*DEV,*CONFIRM)])
        contrast = gains-np.array([index[t,CONTROLS[a],s]['gain'] for s in (*DEV,*CONFIRM)])
        lower = float(gains.mean()-LCB_MULTIPLIER*gains.std(ddof=1)/2)
        checks = dict(all_four_seeds_positive=bool(min(gains)>0),paired_seed_lcb95_positive=bool(lower>0),
                      positive_four_seed_control_advantage=bool(contrast.mean()>0))
        decisions.setdefault(t,{})[a] = dict(promoted=all(checks.values()),checks=checks,
            failure_reasons=[k for k,v in checks.items() if not v],mean_gain=float(gains.mean()),
            seed_lcb95=lower,mean_control_advantage=float(contrast.mean()))
        if all(checks.values()): promoted.append([t,a])
    return dict(phase='confirmation',decisions=decisions,promoted_pairs=promoted,
        release_authorized=False,requires_independent_artifact_audit=True,
        interpretation='Same rows under four splits; folds descriptive, no platform forecast.')
