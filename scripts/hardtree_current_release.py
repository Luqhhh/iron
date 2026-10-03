"""One zero-fit hard-tree reserve using the current measured parent."""
from pathlib import Path
import csv
import importlib.util
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/lux1/iron-v39-hard-tree-quality')
OLD = ROOT/'local/runs/hard-tree-time-exploration-20261003/release-r1'
REVIEW = ROOT/'local/runs/q100-retained-questions-20261004/review-r1'
OUT = ROOT/'local/runs/hardtree-current-release-20261004/release-r1'
NAME = 'HARDTREE_DE3_Q100_A20'
PROTOCOL = ROOT/'docs/hardtree_current_release/PREREGISTRATION.md'
spec = importlib.util.spec_from_file_location('original_hard_tree_release', ROOT/'scripts/hard_tree_time_exploration.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
read, write, sha, verify = old.read, old.write, old.sha, old.verify


def freeze():
    if OUT.exists():
        raise FileExistsError('Private release directory already consumed')
    best = read(ROOT/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'], best['score'], best['zip_sha256']) != (
        'DE3_IRON_EMA_MEAN3_Q100', 96.3979,
        '528b8bf91102bea7ce120a71c560f6b021382132fb425a8f95cbfe79f9e3712c'
    ):
        raise ValueError('Latest parent changed')
    terminal = read(OLD.parent/'launch-r1/final-reconciliation.json')
    m, warm, cold, package = [read(OLD/n) for n in ('manifest.json', 'warm.json', 'cold.json', 'package-audit.json')]
    if (terminal['status'] != 'passed' or terminal['actual_supervisor_exec_exit_code'] != 0
            or terminal['actual_stage_exit_codes'] != [0, 0, 0]
            or terminal['manifest_sha256'] != sha(OLD/'manifest.json')
            or warm['manifest_sha256'] != sha(OLD/'manifest.json')
            or cold['status'] != 'passed' or cold['warm_sha256'] != sha(OLD/'warm.json')
            or package['status'] != 'passed' or package['cold_sha256'] != sha(OLD/'cold.json')
            or package['zip_sha256'] != sha(package['package'])):
        raise ValueError('Original full-data cold release not closed')
    r, audit, t = [read(REVIEW/n) for n in ('report.json', 'independent-audit.json', 'terminal-reconciliation.json')]
    if (audit['status'] != 'passed' or audit['report_sha256'] != sha(REVIEW/'report.json')
            or audit['manifest_sha256'] != sha(REVIEW/'manifest.json') or t['status'] != 'passed'
            or any(t[k] != 0 for k in ('actual_freeze_exit_code', 'actual_evaluation_exit_code', 'actual_independent_audit_exit_code'))):
        raise ValueError('Complete current-reference review required')
    files = {}
    for original in (m, read(REVIEW/'manifest.json')):
        for p, h in original['files'].items():
            key = str((ROOT/p).resolve())
            if key in files and files[key] != h:
                raise ValueError('Conflicting historical identity')
            files[key] = h
    paths = [Path(__file__), PROTOCOL, ROOT/'scripts/hard_tree_time_exploration.py',
             ROOT/'scripts/modernnca_time_exploration.py', ROOT/'uv.lock', ROOT/'pyproject.toml',
             ROOT/'复赛_test/result_template.csv', ROOT/best['package'], ROOT/best['platform_feedback_record'],
             OLD.parent/'launch-r1/final-reconciliation.json', Path(package['package'])]
    paths += [OLD/n for n in ('manifest.json', 'warm.json', 'cold.json', 'package-audit.json', 'query.npz',
                             'warm-predictions.npz', 'selector.pt', 'refit.pt', 'selector-complete.json',
                             'refit-complete.json', 'selector-estimator-start.json', 'refit-estimator-start.json')]
    paths += [REVIEW/n for n in ('manifest.json', 'report.json', 'independent-audit.json', 'terminal-reconciliation.json')]
    for p in paths:
        h = sha(p)
        if str(p) in files and files[str(p)] != h:
            raise ValueError('Frozen source changed')
        files[str(p)] = h
    for role in ('selector', 'refit'):
        if sha(OLD/(role+'.pt')) != warm['artifacts'][role]:
            raise ValueError('Original native state changed')
    if sha(OLD/'warm-predictions.npz') != warm['prediction_sha256'] or sha(ROOT/best['package']) != best['zip_sha256']:
        raise ValueError('Prediction or parent identity changed')
    verify(files); OUT.mkdir(parents=True, exist_ok=False)
    write(OUT/'manifest.json', dict(files=files, created_ns=time.time_ns(), candidate=NAME, reference=best,
        parent=str(ROOT/best['package']), old_full_identity=m['identity'], weight=.2,
        local_gains=r['gains']['HARDTREE_CURRENT_A20'], new_fits=0, new_optimizers=0,
        new_confirmation_seeds=0, maximum_packages=1, desktop_writes=0, agent_uploads=0,
        platform_slots_allocated=0, formal_promoted=False))
    print(json.dumps(dict(status='frozen', files=len(files))))


def inference():
    import numpy as np
    import pandas as pd
    import torch
    from bf_tap_r2.data import FEATURES
    from bf_tap_r2.v39_regressor import TreeRegressor, TreePreprocessor
    from bf_tap_r2.submission import deny_training_reads, validate_result
    m = read(OUT/'manifest.json'); verify(m['files'])
    TreeRegressor.initialize = TreeRegressor.train = TreePreprocessor.fit = old.forbidden
    torch.optim.Adam = torch.optim.AdamW = old.forbidden
    sys.addaudithook(deny_training_reads)
    with np.load(OLD/'query.npz', allow_pickle=False) as a:
        query = pd.DataFrame(a['numeric'], columns=FEATURES)
        query['sample_id'], query['spout_no'] = a['ids'], a['spout']
    ids = [r['sample_id'] for r in csv.DictReader((ROOT/'复赛_test/result_template.csv').open())]
    if query.sample_id.tolist() != ids:
        raise ValueError('Template/query order differs')
    parent_bytes = old.helper.zip_bytes(m['parent'])
    parent = validate_result(parent_bytes, ids)
    warm = read(OLD/'warm.json'); maximum = 0.; member = None
    with np.load(OLD/'warm-predictions.npz', allow_pickle=False) as p:
        np.testing.assert_array_equal(p['query_ids'], query.sample_id.to_numpy(str))
        for role in ('selector', 'refit'):
            if sha(OLD/(role+'.pt')) != warm['artifacts'][role]:
                raise ValueError('Native state identity differs')
            model = TreeRegressor.load(OLD/(role+'.pt')); pred = model.predict(query)
            np.testing.assert_array_equal(pred, p[role])
            reverse = model.predict(query.iloc[::-1])[::-1]
            blocks = np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0, len(query), 37)])
            maximum = max(maximum, float(np.max(np.abs(reverse-pred))), float(np.max(np.abs(blocks-pred))))
            if role == 'refit':
                member = pred
    if maximum > 1e-6:
        raise ValueError('Cold ordering/chunk gate failed')
    return m, ids, parent_bytes, parent, member, maximum


def generate():
    import numpy as np
    from bf_tap_r2.submission import package, ZIP_NAME
    m, ids, parent_bytes, _, member, maximum = inference()
    content = old.helper.payload(parent_bytes, ids, member)
    out = OUT/NAME; out.mkdir(exist_ok=False); package(out, content, ids)
    with (OUT/'cold-member.npz').open('xb') as f:
        np.savez(f, ids=np.asarray(ids), member=member)
    write(OUT/'release.json', dict(status='generated', candidate=NAME, pid=os.getpid(),
        manifest_sha256=sha(OUT/'manifest.json'), zip_sha256=sha(out/ZIP_NAME), csv_sha256=sha(out/'result.csv'),
        cold_member_sha256=sha(OUT/'cold-member.npz'), maximum_cold_difference=maximum,
        cold_states=2, peak_rss_mib=old.memory(), new_fits=0, new_packages=1,
        G1='manual_exploration_one_negative_development_split_unmeasured_not_formally_promoted'))
    print(json.dumps(dict(status='generated', zip_sha256=sha(out/ZIP_NAME))))


def audit():
    import numpy as np
    from bf_tap_r2.submission import ZIP_NAME, validate_result
    m, ids, _, parent, member, cold_difference = inference()
    release = read(OUT/'release.json'); out = OUT/NAME
    if release['pid'] == os.getpid() or release['manifest_sha256'] != sha(OUT/'manifest.json'):
        raise ValueError('Independent process identity required')
    if (release['zip_sha256'] != sha(out/ZIP_NAME) or release['csv_sha256'] != sha(out/'result.csv')
            or release['cold_member_sha256'] != sha(OUT/'cold-member.npz')):
        raise ValueError('Generated artifacts changed')
    with np.load(OUT/'cold-member.npz', allow_pickle=False) as p:
        np.testing.assert_array_equal(p['ids'], np.asarray(ids)); np.testing.assert_array_equal(p['member'], member)
    content = old.helper.zip_bytes(out/ZIP_NAME)
    if content != (out/'result.csv').read_bytes():
        raise ValueError('CSV/ZIP readback differs')
    rows = validate_result(content, ids); maximum = 0.
    for row, base, value in zip(rows, parent, member):
        if row['pred_tap_iron'] != base['pred_tap_iron']:
            raise ValueError('Unchanged iron string differs')
        expected = math.fsum((.8*float(base['pred_tap_time_len']), .2*float(value)))
        maximum = max(maximum, abs(expected-float(row['pred_tap_time_len'])))
    if maximum > 1e-11:
        raise ValueError('Independent scalar package arithmetic differs')
    write(OUT/'independent-audit.json', dict(status='passed', pid=os.getpid(), rows=322, unique_ids=322,
        template_order=True, members=['result.csv'], CRC=True, finite_nonnegative=True,
        iron_string_mismatches=0, maximum_scalar_difference=maximum, maximum_cold_difference=cold_difference,
        cold_states=2, manifest_sha256=sha(OUT/'manifest.json'), release_sha256=sha(OUT/'release.json'),
        zip_sha256=sha(out/ZIP_NAME), peak_rss_mib=old.memory(), new_fits=0, new_packages=0,
        formal_promoted=False, platform_slots_allocated=0))
    print(json.dumps(dict(status='passed', maximum_scalar_difference=maximum, maximum_cold_difference=cold_difference)))


if __name__ == '__main__':
    old.setup()
    {'freeze': freeze, 'generate': generate, 'audit': audit}[sys.argv[1]]()
