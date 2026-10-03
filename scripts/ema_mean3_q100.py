"""One frozen, zero-fit strength probe of the measured three-member EMA."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'local/runs/ema-mean3-q100-20261004/release-r1'
MEAN = ROOT/'local/runs/ema-median3-20261003/development-r1'
FULL = ROOT/'local/runs/ema-mean5-20261003/full-cache-intake-r1'
PROBES = ROOT/'local/runs/ema-time-followup-20261001/probes-r2'
OLD = ROOT/'local/runs/strong-component-regularization/development-r2'
CANDIDATE = 'EMA_MEAN3_Q100'
SEEDS = (42, 3407)
INITS = (42, 1042, 2042)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify(files):
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError('Frozen input changed: '+p)


def probe(v32, v7, members):
    import numpy as np
    v32, v7, members = [np.asarray(v, float) for v in (v32, v7, members)]
    if v32.ndim != 1 or v7.shape != v32.shape or members.shape != (3, len(v32)):
        raise ValueError('Exactly three aligned fixed members required')
    if not all(np.isfinite(v).all() for v in (v32, v7, members)):
        raise ValueError('Nonfinite source')
    result = v32 + (members.mean(axis=0)-v7)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid affine prediction; no clipping')
    return result


def payload(parent, prediction):
    if len(parent) != len(prediction):
        raise ValueError('Prediction length mismatch')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['sample_id', 'pred_tap_iron', 'pred_tap_time_len'])
    for row, value in zip(parent, prediction):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError('Invalid prediction')
        writer.writerow([row['sample_id'], row['pred_tap_iron'], format(float(value), '.17g')])
    return stream.getvalue().encode()


def package_rows(path, ids):
    from bf_tap_r2.submission import validate_result
    with zipfile.ZipFile(path) as z:
        if z.namelist() != ['result.csv'] or z.testzip() is not None:
            raise ValueError('ZIP structure/CRC failure')
        data = z.read('result.csv')
    return validate_result(data, ids), data


def prepare(checks):
    check = read(checks)
    if check['status'] != 'passed' or check['source_sha256'] != sha(__file__) or check['junit_sha256'] != sha(check['junit']):
        raise ValueError('Locked exact-source checks required')
    best = read(ROOT/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'], best['score'], best['zip_sha256']) != ('EMA_MEAN3_FULL_Q75', 96.3954,
            '016e7e9cb3f750c74509dbb51961c61fea205d0296bb233056da66005f69a396'):
        raise ValueError('Latest platform reference changed')
    files = {}
    for directory, audit_name in [(MEAN, 'independent-audit.json'), (FULL, 'cold-audit.json')]:
        m, a = read(directory/'manifest.json'), read(directory/audit_name)
        if a['status'] != 'passed' or a['manifest_sha256'] != sha(directory/'manifest.json'):
            raise ValueError('Original identity audit differs')
        for p, h in m['files'].items():
            if p in files and files[p] != h:
                raise ValueError('Conflicting source identities')
            files[p] = h
        for p in [directory/'manifest.json', directory/audit_name]:
            files[str(p)] = sha(p)
    mr = read(MEAN/'report.json')
    if read(MEAN/'independent-audit.json')['report_sha256'] != sha(MEAN/'report.json'):
        raise ValueError('Original local report changed')
    follow = read(PROBES/'manifest.json')
    for p, h in follow['files'].items():
        if p in files and files[p] != h:
            raise ValueError('Original probe source conflicts')
        files[p] = h
    for seed in SEEDS:
        files[str(MEAN/f'oof-s{seed}.npz')] = mr['oof_sha256'][str(seed)]
        for fold in range(5):
            d = OLD/f'reference-s{seed}-f{fold}'
            c = read(d/'complete.json')
            for name, h in c['hashes'].items():
                if str(d/name) not in files or files[str(d/name)] != h:
                    raise ValueError('Reference cache missing frozen original binding')
    packages = {v['name']: v for v in read(PROBES/'release-summary.json')['packages']}
    for name in ['EMA_TIME_Q75', 'EMA_TIME_Q100']:
        files[packages[name]['zip']] = packages[name]['zip_sha256']
    paths = [Path(__file__), ROOT/'tests/test_ema_mean3_q100.py', Path(checks), Path(check['junit']),
        ROOT/'docs/ema_mean3_q100/PREREGISTRATION.md', MEAN/'report.json',
        PROBES/'manifest.json', PROBES/'release-summary.json', PROBES/'independent-audit.json',
        ROOT/best['package'], ROOT/best['platform_feedback_record'], ROOT/'uv.lock', ROOT/'pyproject.toml',
        ROOT/'configs/candidate_tiers.yaml', ROOT/'scripts/local_platform_diagnostic_release.py']
    for p in paths:
        h = sha(p)
        if str(p) in files and files[str(p)] != h:
            raise ValueError('Current source differs from original')
        files[str(p)] = h
    verify(files)
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN/'manifest.json', dict(files=files, created_ns=time.time_ns(), candidate=CANDIDATE,
        reference=best, training_seeds=list(INITS), split_seeds=list(SEEDS), q=1., new_fits=0,
        new_confirmation_seeds=0, maximum_new_packages=1, desktop_writes=0, agent_uploads=0,
        formal_promoted=False, local_gain_veto=False, original_packages=packages))
    print(json.dumps(dict(status='frozen', files=len(files))))


def local_review():
    import numpy as np
    import yaml
    from bf_tap_r2.component_regularization_run import metric_detail
    from bf_tap_r2.candidate_tiers import classify_candidates
    m = read(RUN/'manifest.json'); verify(m['files'])
    gains = {}; metrics = {'tap_time_len': {'EMA_MEAN3_Q75': {}, CANDIDATE: {}}}; hashes = {}; maximum = 0.
    for seed in SEEDS:
        with np.load(MEAN/f'oof-s{seed}.npz', allow_pickle=False) as a:
            ids = a['ids']; folds = a['folds']; v32 = np.full(len(ids), np.nan); v7 = v32.copy()
            if len(ids) != 2754 or len(set(ids)) != 2754 or set(folds) != set(range(5)):
                raise ValueError('Incomplete within-seed OOF')
            for fold in range(5):
                mask = folds == fold
                with np.load(OLD/f'reference-s{seed}-f{fold}/predictions.npz', allow_pickle=False) as old:
                    np.testing.assert_array_equal(old['query_ids'], ids[mask])
                    np.testing.assert_array_equal(old['tap_iron'], a['iron'][mask])
                    v32[mask] = old['tap_time_len']; v7[mask] = old['v7_time']
            q75 = v32+.75*(a['members'][0]-v7)
            mean3 = v32+.75*(a['members'].mean(axis=0)-v7)
            for x, y in [(q75, a['q75']), (mean3, a['mean3'])]:
                maximum = max(maximum, float(np.max(np.abs(x-y))))
            if maximum > 1e-11:
                raise ValueError('Original portfolio reconstruction differs')
            p = probe(v32, v7, a['members']); y = a['actual'][:, 1]; base = a['mean3']
            gains[str(seed)] = float(50*np.sum(np.abs(y-base)-np.abs(y-p))/np.abs(y).sum())
            for name, pred in [('EMA_MEAN3_Q75', base), (CANDIDATE, p)]:
                metrics['tap_time_len'][name][str(seed)] = metric_detail(y, pred, folds, a['spouts'])
            path = RUN/f'oof-s{seed}.npz'
            with path.open('xb') as f:
                np.savez_compressed(f, ids=ids, folds=folds, actual=a['actual'], iron=a['iron'],
                    reference=base, members=a['members'], v32=v32, v7=v7, candidate=p)
            hashes[str(seed)] = sha(path)
    spec = dict(split_seeds=list(SEEDS), folds=5, candidates={'tap_time_len': [CANDIDATE]},
        reference_by_target={'tap_time_len': 'EMA_MEAN3_Q75'}, tie_preference_by_target={'tap_time_len': [CANDIDATE]})
    write(RUN/'local-report.json', dict(status='complete', gains=gains, metrics=metrics,
        candidate_tiers=classify_candidates(metrics, spec, yaml.safe_load((ROOT/'configs/candidate_tiers.yaml').read_text())),
        maximum_reference_difference=maximum, oof_sha256=hashes, manifest_sha256=sha(RUN/'manifest.json'),
        formal_promoted=False, role='pre_registered_platform_strength_exploration', local_gain_veto=False,
        new_fits=0, new_confirmation_seeds=0))
    print(json.dumps(dict(status='local_complete', gains=gains)))


def inference():
    import numpy as np
    import pandas as pd
    from bf_tap_r2.ema_nested_residual import setup, forbid_training, memory
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.submission import deny_training_reads
    from bf_tap_r2.v7_periodic import digest
    from local_platform_diagnostic_release import load_native
    setup()
    # Hash-only verification precedes the label-free inference boundary.
    m = read(RUN/'manifest.json'); verify(m['files'])
    original = read(FULL/'manifest.json'); follow = read(PROBES/'manifest.json')['spec']
    old_release = ROOT/'local/runs/ema-mean3-exploration-release-20261002/release-r1'
    spec = read(old_release/'manifest.json')['spec']
    sys.addaudithook(deny_training_reads)
    query = pd.read_pickle(old_release/'query.pkl')
    if any(t in query for t in ['tap_iron', 'tap_time_len']) or len(query) != 322:
        raise ValueError('Label-free query required')
    if digest(query.to_dict(orient='list')) != original['original_model_partition']['query_frame']:
        raise ValueError('Query features differ')
    ids = query.sample_id.tolist()
    template = list(csv.DictReader((ROOT/'复赛_test/result_template.csv').open()))
    if ids != [r['sample_id'] for r in template]:
        raise ValueError('Template IDs/order differ')
    members = []; maximum = 0.; states = []
    with forbid_training():
        for init in INITS:
            d = Path(original['model_directories'][str(init)]); epoch = None
            for role in ['selection', 'refit']:
                model = ComponentRegressor.load(d/(role+'.pt')); saved = model.saved; trace = saved['trace']
                if (saved['settings'] != dict(spec['training'], random_seed=init)
                        or saved['mechanisms'] != spec['mechanisms'] or saved['arm'] != 'EMA'):
                    raise ValueError('Original scientific recipe differs')
                if trace['fit_ids_digest'] != original['original_model_partition']['inner_fit' if role == 'selection' else 'training']:
                    raise ValueError('Original training partition differs')
                if role == 'selection':
                    epoch = trace['selected_epoch']
                elif trace['selected_epoch'] != epoch or trace['stopped_epoch'] != epoch:
                    raise ValueError('Fresh refit epoch differs')
                states.append(dict(directory=str(d), role=role, sha256=sha(d/(role+'.pt'))))
            p = model.predict(query)[:, 0]
            if init == 42:
                expected = np.load(d/'cold.npy', allow_pickle=False)[:, 0]
            else:
                with np.load(d/'predictions.npz', allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['query_ids'], np.asarray(ids, str)); expected = a['prediction'][:, 0]
            np.testing.assert_array_equal(p, expected); members.append(p)
            for v in [model.predict(query.iloc[::-1])[::-1, 0],
                    np.concatenate([model.predict(query.iloc[i:i+37])[:, 0] for i in range(0, 322, 37)])]:
                maximum = max(maximum, float(np.max(np.abs(v-p))))
        artifact = lambda name: ROOT/follow['artifacts'][name]['path']
        native = load_native(artifact('v7_model')); v7 = native.predict(query)
        np.testing.assert_array_equal(v7, np.load(artifact('v7_prediction'), allow_pickle=False))
        for v in [native.predict(query.iloc[::-1])[::-1],
                np.concatenate([native.predict(query.iloc[i:i+37]) for i in range(0, 322, 37)])]:
            maximum = max(maximum, float(np.max(np.abs(v-v7))))
    if maximum > .0005 or memory() > 1536:
        raise ValueError('Cold numerical or resource gate failed')
    parent, _ = package_rows(ROOT/m['reference']['package'], ids)
    v32rows, _ = package_rows(artifact('v32'), ids)
    v32 = np.asarray([float(r['pred_tap_time_len']) for r in v32rows]); members = np.asarray(members)
    if [r['pred_tap_iron'] for r in parent] != [r['pred_tap_iron'] for r in v32rows]:
        raise ValueError('Unchanged iron field differs')
    q75rows, _ = package_rows(m['original_packages']['EMA_TIME_Q75']['zip'], ids)
    q100rows, _ = package_rows(m['original_packages']['EMA_TIME_Q100']['zip'], ids)
    for q, rows in [(.75, q75rows), (1., q100rows)]:
        np.testing.assert_array_equal(v32+q*(members[0]-v7), [float(r['pred_tap_time_len']) for r in rows])
    old_mean = np.asarray([float(r['pred_tap_time_len']) for r in q75rows])+.75*((members[0]+members[1]+members[2])/3-members[0])
    np.testing.assert_array_equal(old_mean, [float(r['pred_tap_time_len']) for r in parent])
    prediction = probe(v32, v7, members)
    alternate = np.asarray([float(r['pred_tap_time_len']) for r in q100rows])+(members.mean(axis=0)-members[0])
    if float(np.max(np.abs(alternate-prediction))) > 1e-11:
        raise ValueError('Old Q100 plus member replacement differs')
    independent = [float(b)+math.fsum([math.fsum(float(x) for x in row)/3, -float(o)])
        for b, o, row in zip(v32, v7, members.T)]
    diff = max(abs(x-float(y)) for x, y in zip(independent, prediction))
    if diff > 1e-11:
        raise ValueError('Independent probe arithmetic differs')
    audit = dict(refit_cold_predictors=4, EMA_checkpoint_identities=states,
        full_batch_cold_difference=0., maximum_chunk_order_difference=maximum, independent_scalar_difference=diff,
        original_Q75_Q100_and_mean3_reproduced=True, training_label_reads=0, new_fits=0, peak_rss_mib=memory())
    return ids, parent, prediction, audit


def build():
    if read(RUN/'local-report.json')['status'] != 'complete':
        raise ValueError('Full local evidence required')
    from bf_tap_r2.submission import package, ZIP_NAME
    ids, parent, p, cold = inference()
    dest = RUN/CANDIDATE; dest.mkdir()
    package(dest, payload(parent, p), ids)
    write(RUN/'release.json', dict(candidate=CANDIDATE, zip=str(dest/ZIP_NAME), zip_sha256=sha(dest/ZIP_NAME),
        csv_sha256=sha(dest/'result.csv'), cold=cold, G0='built_pending_independent_audit',
        G1='pre_registered_unmeasured_platform_exploration_not_formal_promotion', manifest_sha256=sha(RUN/'manifest.json'),
        new_fits=0, new_packages=1, desktop_writes=0, agent_uploads=0))
    print(json.dumps(dict(status='built', candidate=CANDIDATE)))


def audit():
    import numpy as np
    m = read(RUN/'manifest.json'); r = read(RUN/'release.json'); local = read(RUN/'local-report.json')
    verify(m['files']); gains = {}; maximum = 0.
    # Independent scalar totals from the complete frozen same-seed columns.
    for seed in SEEDS:
        path = RUN/f'oof-s{seed}.npz'
        if sha(path) != local['oof_sha256'][str(seed)]:
            raise ValueError('Local output changed')
        with np.load(path, allow_pickle=False) as a, np.load(MEAN/f'oof-s{seed}.npz', allow_pickle=False) as old:
            for k in ['ids', 'folds', 'actual', 'iron', 'members']:
                np.testing.assert_array_equal(a[k], old[k])
            np.testing.assert_array_equal(a['reference'], old['mean3'])
            for fold in range(5):
                mask = a['folds'] == fold
                with np.load(OLD/f'reference-s{seed}-f{fold}/predictions.npz', allow_pickle=False) as original:
                    np.testing.assert_array_equal(a['ids'][mask], original['query_ids'])
                    np.testing.assert_array_equal(a['v32'][mask], original['tap_time_len'])
                    np.testing.assert_array_equal(a['v7'][mask], original['v7_time'])
            pred = [math.fsum([float(b), math.fsum(float(x) for x in v)/3, -float(o)])
                for b, v, o in zip(a['v32'], a['members'].T, a['v7'])]
            maximum = max(maximum, max(abs(x-float(y)) for x, y in zip(pred, a['candidate'])))
            def score(p):
                return 100-50*math.fsum(math.fsum(abs(float(y)-float(v)) for y, v in zip(a['actual'][:, j], col))/
                    math.fsum(abs(float(v)) for v in a['actual'][:, j]) for j, col in enumerate([a['iron'], p]))
            gains[str(seed)] = score(pred)-score(a['reference'])
            maximum = max(maximum, abs(gains[str(seed)]-local['gains'][str(seed)]))
    if maximum > 1e-11:
        raise ValueError('Independent local score differs')
    ids, parent, prediction, cold = inference()
    rows, data = package_rows(r['zip'], ids)
    if (sha(r['zip']) != r['zip_sha256'] or hashlib.sha256(data).hexdigest() != r['csv_sha256']
            or data != payload(parent, prediction) or data != Path(r['zip']).with_name('result.csv').read_bytes()):
        raise ValueError('Independent cold package reproduction failed')
    if [v['pred_tap_iron'] for v in rows] != [v['pred_tap_iron'] for v in parent]:
        raise ValueError('Iron field string changed')
    write(RUN/'independent-audit.json', dict(status='passed', gains=gains, maximum_local_difference=maximum,
        cold=cold, rows=322, iron_string_mismatches=0, manifest_sha256=sha(RUN/'manifest.json'),
        local_report_sha256=sha(RUN/'local-report.json'), release_sha256=sha(RUN/'release.json'),
        zip_sha256=sha(r['zip']), new_fits=0, new_packages=0, formal_promoted=False))
    print(json.dumps(dict(status='passed', gains=gains, zip_sha256=sha(r['zip']))))


if __name__ == '__main__':
    if sys.version_info[:2] != (3, 12) or any(os.environ.get(k) != '1' for k in
            ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']):
        raise RuntimeError('Locked Python3.12 and pre-import numerical threads=1 required')
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'local', 'build', 'audit'])
    parser.add_argument('--checks'); args = parser.parse_args()
    try:
        if args.stage == 'prepare':
            prepare(args.checks)
        else:
            {'local': local_review, 'build': build, 'audit': audit}[args.stage]()
    except BaseException as error:
        if RUN.exists():
            write(RUN/(args.stage+'-failure.json'), dict(status='failed', error=repr(error), new_fits=0, automatic_retry=False))
        raise
