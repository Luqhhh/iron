"""Zero-fit intake of two fixed retained questions at the measured Q100 parent."""
from pathlib import Path
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'local/runs/q100-retained-questions-20261004/review-r1'
TREE = ROOT/'local/runs/mean3-retained-components-20261003/review-r2'
BASE = ROOT/'local/runs/mean3-base-control-review-20261003/review-r1'
Q100 = ROOT/'local/runs/ema-mean3-q100-20261004/release-r1'
DE3 = ROOT/'local/runs/independent-ensemble-checkpoints-20260929/development-DE3'
ORDER = ('HARDTREE_CURRENT_A20', 'BASE_MEAN3_CURRENT')
SEEDS = (42, 3407)
REFERENCE = 'DE3_EMA_MEAN3_Q100'


def helpers():
    from bf_tap_r2 import ema_independent_batches as m
    return m


def freeze():
    m = helpers()
    if OUT.exists():
        raise FileExistsError('Review directory already consumed')
    best = m.read(ROOT/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'], best['score'], best['zip_sha256']) != (
        'DE3_IRON_EMA_MEAN3_Q100', 96.3979,
        '528b8bf91102bea7ce120a71c560f6b021382132fb425a8f95cbfe79f9e3712c'
    ):
        raise ValueError('Current reference changed')
    files = {}
    paths = [Path(__file__), ROOT/'docs/q100_retained_questions/PREREGISTRATION.md',
             ROOT/best['package'], ROOT/best['platform_feedback_record'], ROOT/'uv.lock',
             ROOT/'pyproject.toml', ROOT/'configs/candidate_tiers.yaml']
    for directory in (TREE, BASE, Q100):
        manifest = m.read(directory/'manifest.json')
        audit = m.read(directory/'independent-audit.json')
        report_name = 'local-report.json' if directory == Q100 else 'report.json'
        report_key = 'local_report_sha256' if directory == Q100 else 'report_sha256'
        report = m.read(directory/report_name)
        if (audit['status'] != 'passed' or audit['manifest_sha256'] != m.sha(directory/'manifest.json')
                or audit[report_key] != m.sha(directory/report_name)):
            raise ValueError('Original audit binding differs')
        m.merge_legacy_files(files, manifest['files'], ROOT)
        paths += [directory/n for n in ('manifest.json', 'independent-audit.json', report_name)]
        if directory != Q100:
            terminal = m.read(directory/'terminal-reconciliation.json')
            if terminal['status'] != 'passed' or any(terminal[k] != 0 for k in (
                'actual_freeze_exit_code', 'actual_evaluation_exit_code', 'actual_independent_audit_exit_code'
            )):
                raise ValueError('Original actual terminal differs')
            paths.append(directory/'terminal-reconciliation.json')
        hashes = report['output_sha256' if directory == BASE else 'oof_sha256']
        for seed in SEEDS:
            p = directory/f'oof-s{seed}.npz'
            if m.sha(p) != hashes[str(seed)]:
                raise ValueError('Original array identity differs')
            paths.append(p)
    audit = m.read(DE3/'audit.json')
    if (audit['status'] != 'passed' or audit['manifest_sha256'] != m.sha(DE3/'manifest.json')
            or audit['summary_sha256'] != m.sha(DE3/'summary.json')):
        raise ValueError('Original DE3 audit differs')
    paths += [DE3/n for n in ('manifest.json', 'summary.json', 'audit.json')]
    for seed in SEEDS:
        for fold in range(5):
            for directory in (DE3/f'reference-s{seed}-f{fold}', DE3/f'tap_iron-DE3-s{seed}-f{fold}'):
                completion = m.read(directory/'complete.json')
                paths.append(directory/'complete.json')
                m.merge_legacy_files(files, {str(directory/n): h for n, h in completion['hashes'].items()}, ROOT)
    paths += list((ROOT/'src').rglob('*.py'))
    for p in paths:
        m.merge_legacy_files(files, {str(p): m.sha(p)}, ROOT)
    m.verify(files)
    OUT.mkdir(parents=True, exist_ok=False)
    m.write(OUT/'manifest.json', dict(files=files, created_ns=time.time_ns(), reference=best,
        candidate_order=list(ORDER), split_seeds=list(SEEDS), retrospective=True,
        new_fits=0, new_model_predict_calls=0, new_confirmation_seeds=0,
        packages=0, platform_slots_allocated=0, formal_promoted=False))
    print(json.dumps(dict(status='frozen', files=len(files))))


def inputs(seed):
    import numpy as np
    m = helpers()
    a = m.reference_arrays(seed)
    with np.load(BASE/f'oof-s{seed}.npz', allow_pickle=False) as b, \
            np.load(TREE/f'oof-s{seed}.npz', allow_pickle=False) as t, \
            np.load(Q100/f'oof-s{seed}.npz', allow_pickle=False) as q:
        for old in (b, t):
            for key in ('ids', 'folds', 'actual'):
                np.testing.assert_array_equal(old[key], a[key])
            np.testing.assert_array_equal(old['iron'], q['iron'])
            np.testing.assert_array_equal(old['mean3'], q['reference'])
        a['base_members'] = b['members'].copy()
        a['tree'] = t['HARDTREE_GLOBAL_A20_MEAN3_member'].copy()
    if (len(a['ids']) != 2754 or len(set(a['ids'])) != 2754
            or set(a['folds']) != set(range(5)) or a['base_members'].shape != (3, 2754)):
        raise ValueError('Complete same-seed OOF required')
    return a


def evaluate():
    import numpy as np
    import yaml
    from bf_tap_r2.candidate_tiers import classify_candidates
    from bf_tap_r2.component_regularization_run import metric_detail
    m = helpers(); manifest = m.read(OUT/'manifest.json'); m.verify(manifest['files'])
    gains = {k: {} for k in ORDER}; metrics = {'tap_time_len': {k: {} for k in (REFERENCE, *ORDER)}}
    hashes = {}
    for seed in SEEDS:
        a = inputs(seed); reference = a['current']; y = a['actual'][:, 1]
        candidates = {
            ORDER[0]: .8*reference+.2*a['tree'],
            ORDER[1]: a['v32']+(a['base_members'].mean(0)-a['v7']),
        }
        for name, p in [(REFERENCE, reference), *candidates.items()]:
            if not np.isfinite(p).all() or (p < 0).any():
                raise ValueError('Invalid prediction; no clipping')
            metrics['tap_time_len'][name][str(seed)] = metric_detail(y, p, a['folds'], a['spouts'])
            if name != REFERENCE:
                gains[name][str(seed)] = float(50*np.sum(np.abs(y-reference)-np.abs(y-p))/np.abs(y).sum())
        p = OUT/f'oof-s{seed}.npz'
        m.save_arrays(p, ids=a['ids'], folds=a['folds'], actual=a['actual'], iron=a['iron'],
            reference=reference, base_members=a['base_members'], tree=a['tree'],
            v32=a['v32'], v7=a['v7'], **candidates)
        hashes[str(seed)] = m.sha(p)
    spec = dict(split_seeds=list(SEEDS), folds=5, candidates={'tap_time_len': list(ORDER)},
        reference_by_target={'tap_time_len': REFERENCE}, tie_preference_by_target={'tap_time_len': list(ORDER)})
    report = dict(status='complete_retrospective_zero_fit_review', gains=gains, metrics=metrics,
        candidate_tiers=classify_candidates(metrics, spec, yaml.safe_load((ROOT/'configs/candidate_tiers.yaml').read_text())),
        confirmation_eligible={k: all(v > 0 for v in g.values()) for k, g in gains.items()},
        formal_promoted=False, manifest_sha256=m.sha(OUT/'manifest.json'), output_sha256=hashes,
        new_fits=0, new_model_predict_calls=0, new_confirmation_seeds=0, new_packages=0, platform_slots_allocated=0)
    m.verify(manifest['files']); m.write(OUT/'report.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'gains', 'confirmation_eligible')}))


def audit():
    import numpy as np
    m = helpers(); manifest = m.read(OUT/'manifest.json'); report = m.read(OUT/'report.json')
    m.verify(manifest['files']); maximum = 0.; gains = {k: {} for k in ORDER}
    for seed in SEEDS:
        source = inputs(seed); p = OUT/f'oof-s{seed}.npz'
        if m.sha(p) != report['output_sha256'][str(seed)]:
            raise ValueError('Saved OOF changed')
        with np.load(p, allow_pickle=False) as a:
            for key in ('ids', 'folds', 'actual', 'iron', 'base_members', 'tree', 'v32', 'v7'):
                np.testing.assert_array_equal(a[key], source[key])
            np.testing.assert_array_equal(a['reference'], source['current'])
            expected = {
                ORDER[0]: [math.fsum((.8*float(b), .2*float(t))) for b, t in zip(a['reference'], a['tree'])],
                ORDER[1]: [math.fsum((float(b), math.fsum(map(float, members))/3, -float(v)))
                           for b, members, v in zip(a['v32'], a['base_members'].T, a['v7'])],
            }
            for name, pred in expected.items():
                maximum = max(maximum, max(abs(x-float(y)) for x, y in zip(pred, a[name])))
                gain = m.scalar_score(a['actual'], a['iron'], pred)-m.scalar_score(a['actual'], a['iron'], a['reference'])
                gains[name][str(seed)] = gain
                maximum = max(maximum, abs(gain-report['gains'][name][str(seed)]))
    eligible = {k: all(v > 0 for v in g.values()) for k, g in gains.items()}
    if (maximum > 1e-11 or eligible != report['confirmation_eligible'] or report['formal_promoted']
            or report['manifest_sha256'] != m.sha(OUT/'manifest.json') or list(gains) != manifest['candidate_order']):
        raise ValueError('Independent arithmetic, decision or scope differs')
    m.write(OUT/'independent-audit.json', dict(status='passed', gains=gains, maximum_difference=maximum,
        manifest_sha256=m.sha(OUT/'manifest.json'), report_sha256=m.sha(OUT/'report.json'),
        new_fits=0, new_model_predict_calls=0, new_packages=0))
    print(json.dumps(dict(status='passed', gains=gains, maximum_difference=maximum)))


if __name__ == '__main__':
    if sys.version_info[:2] != (3, 12) or any(os.environ.get(k) != '1' for k in
        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')):
        raise ValueError('Locked Python3.12 and single numerical threads required')
    {'freeze': freeze, 'evaluate': evaluate, 'audit': audit}[sys.argv[1]]()
