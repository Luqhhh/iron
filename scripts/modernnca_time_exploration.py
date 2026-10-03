"""Frozen full-data exploration using the unchanged, isolated ModernNCA source."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'local/worktrees/modernnca-source-preparation'
SOURCE_COMMIT = 'f4b0c9d209e6a2d8285cd49314ec01e8f7cde3f7'
RUN = ROOT / 'local/runs/modernnca-time-exploration-20261003/release-r1'
DEV = ROOT / 'local/runs/modernnca-q75-development-20261002/development-r1'
REF = ROOT / 'local/runs/modernnca-q75-readiness-20261001/preparation-r1/reference-r2'
PARENT_SHA = '41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825'
NAME = 'MODERNNCA_TIME_A20'
PROTOCOL = ROOT / 'docs/modernnca_time_exploration/PREREGISTRATION.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def zip_bytes(path):
    with zipfile.ZipFile(path) as z:
        if z.namelist() != ['result.csv'] or z.testzip() is not None:
            raise ValueError('ZIP members/CRC differ')
        return z.read('result.csv')


def payload(parent, ids, member):
    """Preserve untouched CSV text and refuse invalid values, without clipping."""
    import math
    reader = csv.DictReader(io.StringIO(parent.decode('utf-8')))
    columns = ['sample_id', 'pred_tap_iron', 'pred_tap_time_len']
    rows = list(reader)
    if (reader.fieldnames != columns or len(rows) != 322 or len(set(ids)) != 322
            or [r['sample_id'] for r in rows] != list(ids) or len(member) != 322
            or any(set(r) != set(columns) for r in rows)):
        raise ValueError('Parent/template/shape differs')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, columns, lineterminator='\n')
    writer.writeheader()
    for row, value in zip(rows, member):
        old, iron = float(row['pred_tap_time_len']), float(row['pred_tap_iron'])
        new = .8 * old + .2 * float(value)
        if not all(math.isfinite(v) and v >= 0 for v in (old, iron, float(value), new)):
            raise ValueError('Invalid predictions; clipping prohibited')
        writer.writerow({**row, 'pred_tap_time_len': format(new, '.17g')})
    return stream.getvalue().encode('utf-8')


def setup():
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Locked Python 3.12 required')
    if any(os.environ.get(k) != '1' for k in
           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')):
        raise ValueError('Pin numerical threads before imports')
    sys.path.insert(0, str(SOURCE / 'src'))
    import torch
    import numpy as np
    if torch.__version__ != '2.14.0+cpu' or np.__version__ != '2.2.6':
        raise ValueError('Frozen CPU environment changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('Frozen dependency changed: ' + path)


def admitted():
    manifest = read(RUN / 'manifest.json')
    verify(manifest['files'])
    if manifest['settings'] != asdict(__import__('bf_tap_r2.modernnca_model', fromlist=['Settings']).Settings()):
        raise ValueError('Native settings differ')
    return manifest


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if peak > 1536:
        raise ValueError('RSS gate exceeded')
    return peak


def frames():
    import numpy as np
    import pandas as pd
    from bf_tap_r2.data import FEATURES, TARGETS
    from bf_tap_r2.v2_release import load_v2
    receipt = read(REF / 'q75-reference-receipt.json')
    with np.load(REF / receipt['reference_file'], allow_pickle=False) as a:
        frame = pd.DataFrame(a['numeric'], columns=FEATURES)
        frame['sample_id'], frame['spout_no'] = a['ids'], a['spout']
        for j, target in enumerate(TARGETS):
            frame[target] = a['targets'][:, j]
        frame = frame.loc[:, receipt['frame_columns']].astype(receipt['frame_dtypes'])
    query = load_v2(ROOT / '复赛_test', 'test', 322)
    if len(frame) != 2754 or any(t in query for t in TARGETS):
        raise ValueError('Wrong train/query schema')
    return frame, query


def freeze(checks):
    from bf_tap_r2.modernnca_model import Settings, clean, validate_pair
    from bf_tap_r2.modernnca_execution import inner_parts, frame_digest
    from bf_tap_r2.modernnca_ledger import ReservationLedger
    import numpy as np
    if RUN.exists():
        raise FileExistsError('Scientific output already exists')
    best = read(ROOT / 'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'], best['score'], best['zip_sha256']) != ('EMA_TIME_Q75', 96.392, PARENT_SHA):
        raise ValueError('Latest reference differs')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() != SOURCE_COMMIT:
        raise ValueError('Original scientific source commit differs')
    state = read(ROOT / 'EVIDENCE_STATUS.json')['modernnca_q75_development_20261002']
    for name, expected in state['artifacts'].items():
        if sha(DEV / name) != expected:
            raise ValueError('Original development evidence changed')
    receipt = read(REF / 'q75-reference-receipt.json')
    if sha(REF / receipt['reference_file']) != receipt['reference_sha256']:
        raise ValueError('Original training/reference identity differs')
    tests = read(checks)
    if tests['exit_code'] != 0 or tests['python'] != '3.12' or tests['runner_sha256'] != sha(__file__):
        raise ValueError('Passing locked checks for this runner required')
    if sha(tests['junit']) != tests['junit_sha256']:
        raise ValueError('Test receipt differs')
    parent = Path(best['desktop'])
    if sha(parent) != PARENT_SHA:
        raise ValueError('Parent identity differs')
    paths = list((SOURCE / 'src').rglob('*.py'))
    paths += [SOURCE / p for p in ('uv.lock', 'pyproject.toml', 'configs/modernnca/SPEC.yaml')]
    paths += [ROOT / 'uv.lock', ROOT / 'pyproject.toml', ROOT / 'configs/protection.yaml',
              ROOT / 'local/authorizations/optimization-standing-20261002-r1.json',
              PROTOCOL, Path(__file__), Path(checks), Path(tests['junit']), parent,
              REF / 'q75-reference-receipt.json', REF / receipt['reference_file']]
    paths += [ROOT / f'复赛_{s}/{s}_{k}.csv' for s in ('train', 'test') for k in ('samples', 'features')]
    paths += [ROOT / '复赛_test/result_template.csv']
    paths += [DEV / name for name in state['artifacts']]
    files = {str(p): sha(p) for p in paths}
    # Also bind the original development code and all reference provenance.
    original = read(ROOT / 'local/runs/modernnca-q75-readiness-20261001/preparation-r1/development-manifest.json')
    files.update(original['sources'])
    files.update(receipt['model_source_hashes'])
    files.update({str(ROOT / p): h for p, h in receipt['input_hashes'].items()})
    files.update({str(Path(receipt['original_source_worktree']) / p): h for p, h in receipt['source_hashes'].items()})
    verify(files)
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN / 'frozen-files.json', files)
    write(RUN / 'access-before-round2-read.json', dict(scope='authorized_round2_only',
          protected_preliminary_targets=False, frozen_files_sha256=sha(RUN / 'frozen-files.json')))
    training, query = frames()
    validate_pair(clean(training), clean(query))
    fitting, calibration, info = inner_parts(training)
    ids = query.sample_id.to_numpy(str)
    template = list(csv.DictReader((ROOT / '复赛_test/result_template.csv').open()))
    if [r['sample_id'] for r in template] != ids.tolist():
        raise ValueError('Official template order differs')
    parent_bytes = zip_bytes(parent)
    payload(parent_bytes, ids, np.ones(322))  # Schema-only admission, no package.
    with (RUN / 'query.json').open('x') as stream:
        clean(query).to_json(stream, orient='table', double_precision=15)
    with (RUN / 'parent.csv').open('xb') as stream:
        stream.write(parent_bytes)
    ledger = ReservationLedger.create(RUN / 'ledger', dict(estimator=2, optimizer=2))
    files.update({str(RUN / n): sha(RUN / n) for n in ('query.json', 'parent.csv')})
    write(RUN / 'manifest.json', dict(candidate=NAME, settings=asdict(Settings()),
          identity=dict(source_directory=str(RUN), split_seed=-1, trial_id=NAME),
          formal_promoted=False, reference_score_user_reported=96.392, target=96.45,
          files=files, ledger_policy_sha256=ledger.policy_sha256,
          source_commit=SOURCE_COMMIT, runner_commit=subprocess.check_output(
              ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          training_sha256=frame_digest(training), query_sha256=frame_digest(query),
          fitting_sha256=frame_digest(fitting), calibration_sha256=frame_digest(calibration),
          inner_fold_hash=info['inner_fold_hash'], group_hash=info['group_hash'],
          rows=dict(training=len(training), fitting=len(fitting), calibration=len(calibration), query=len(query)),
          counts=dict(full_training_procedures=1, estimator=2, optimizer=2, packages=1,
                      new_cv=0, confirmation_seeds=0, desktop_writes=0, agent_uploads=0)))
    print(json.dumps(dict(status='frozen', files=len(files), rows=read(RUN / 'manifest.json')['rows'])))


def fit():
    import numpy as np
    import torch
    from bf_tap_r2.modernnca_model import NeighborRegressor, Settings, clean
    from bf_tap_r2.modernnca_execution import inner_parts, frame_digest
    from bf_tap_r2.modernnca_ledger import ReservationLedger
    manifest = admitted()
    training, query = frames()
    fitting, calibration, _ = inner_parts(training)
    if any(frame_digest(f) != manifest[n + '_sha256'] for n, f in
           [('training', training), ('query', query), ('fitting', fitting), ('calibration', calibration)]):
        raise ValueError('Release partition changed')
    ledger = ReservationLedger.open(RUN / 'ledger', manifest['ledger_policy_sha256'])
    real_adam = torch.optim.AdamW
    calls = []
    def counted(*a, **kw):
        if len(calls) >= 2:
            raise ValueError('Actual optimizer constructor budget exceeded')
        calls.append(len(calls) + 1)
        write(RUN / f'optimizer-constructor-{len(calls)}.json', dict(attempt=len(calls), time_ns=time.time_ns()))
        return real_adam(*a, **kw)
    torch.optim.AdamW = counted
    selected, artifacts, predictions, costs = None, {}, {}, {}
    for role, part in [('selector', fitting), ('refit', training)]:
        identity = dict(**manifest['identity'], role=role, fit_sha256=frame_digest(part))
        with ledger.event('estimator', (NAME, role), identity) as er:
            with ledger.event('optimizer', (NAME, role), identity) as opt:
                model = NeighborRegressor('LEARNED_ENCODER', Settings()).initialize(clean(part), part.tap_time_len.to_numpy())
                cal = (clean(calibration), calibration.tap_time_len.to_numpy()) if role == 'selector' else None
                epoch = model.train(Settings().max_epochs if role == 'selector' else selected, cal)
                artifacts[role] = model.save(RUN / (role + '.npz'))
                predictions[role] = model.predict(clean(query))
                if role == 'selector':
                    selected = epoch
                    predictions['calibration'] = model.predict(clean(calibration))
                costs[role] = dict(selected_epoch=epoch, actual_epochs=model.actual_epochs_,
                                   optimizer_steps=model.optimizer_steps_, model_sha256=artifacts[role])
                opt.update(costs[role]); er.update(costs[role])
                memory()
    if len(calls) != 2:
        raise ValueError('Optimizer count mismatch')
    with (RUN / 'warm-predictions.npz').open('xb') as stream:
        np.savez(stream, query_ids=query.sample_id.to_numpy(str), **predictions)
    admitted()
    write(RUN / 'warm.json', dict(artifacts=artifacts, costs=costs, optimizer_constructors=len(calls),
          predictions_sha256=sha(RUN / 'warm-predictions.npz'), manifest_sha256=sha(RUN / 'manifest.json'),
          peak_rss_mib=memory(), ledger=ledger.inspect()))
    print(json.dumps(dict(status='warm_complete', selected_epoch=selected, peak_rss_mib=memory())))


def forbidden(*a, **kw):
    raise RuntimeError('Independent audit attempted model initialization/training')


def cold():
    import numpy as np
    import torch
    from bf_tap_r2.modernnca_model import NeighborRegressor, clean
    from bf_tap_r2.modernnca_execution import inner_parts
    from bf_tap_r2.modernnca_audit import audit_partition, read_state, numpy_predict
    from bf_tap_r2.modernnca_ledger import ReservationLedger
    manifest = admitted(); warm = read(RUN / 'warm.json')
    if warm['manifest_sha256'] != sha(RUN / 'manifest.json') or warm['predictions_sha256'] != sha(RUN / 'warm-predictions.npz'):
        raise ValueError('Warm identity changed')
    NeighborRegressor.initialize = NeighborRegressor.train = forbidden
    torch.optim.AdamW = forbidden
    training, query = frames(); fitting, calibration, _ = inner_parts(training)
    maximum, reports = 0., {}
    with np.load(RUN / 'warm-predictions.npz', allow_pickle=False) as p:
        if p['query_ids'].tolist() != query.sample_id.tolist():
            raise ValueError('Warm query IDs changed')
        for role, part in [('selector', fitting), ('refit', training)]:
            cal = (clean(calibration), calibration.tap_time_len.to_numpy()) if role == 'selector' else None
            state_path = RUN / (role + '.npz'); anchor = warm['artifacts'][role]
            metadata, state, _, arrays = read_state(state_path, anchor)
            if metadata['settings'] != manifest['settings'] or metadata['arm'] != 'LEARNED_ENCODER':
                raise ValueError('Wrong recipe/arm')
            values, reports[role] = audit_partition(state_path, anchor, clean(part), part.tap_time_len.to_numpy(), clean(query), cal)
            maximum = max(maximum, float(np.max(np.abs(values - p[role]))))
            if role == 'selector':
                cp = numpy_predict(clean(calibration), metadata, state, arrays['bank_x'], arrays['bank_y'])
                maximum = max(maximum, float(np.max(np.abs(cp - p['calibration']))))
            for k in ('actual_epochs', 'selected_epoch', 'optimizer_steps'):
                if warm['costs'][role][k] != metadata[k]:
                    raise ValueError('Native counts changed')
    if maximum > 1e-8 or reports['selector']['selected_epoch'] != reports['refit']['actual_epochs']:
        raise ValueError('Cold predictions/selected epoch differ')
    counts = ReservationLedger.open(RUN / 'ledger', manifest['ledger_policy_sha256']).inspect()
    if counts != dict(started=dict(estimator=2, optimizer=2), completed=dict(estimator=2, optimizer=2),
                      failed=dict(estimator=0, optimizer=0), incomplete=dict(estimator=0, optimizer=0)):
        raise ValueError('Budget not closed')
    write(RUN / 'cold.json', dict(status='passed', maximum_difference=maximum, models=reports,
          ledger=counts, warm_sha256=sha(RUN / 'warm.json'), peak_rss_mib=memory()))
    print(json.dumps(dict(status='cold_passed', maximum_difference=maximum)))


def package_cold():
    import numpy as np
    import pandas as pd
    import torch
    from bf_tap_r2.modernnca_audit import read_state, numpy_predict
    from bf_tap_r2.modernnca_model import NeighborRegressor
    from bf_tap_r2.submission import package, validate_result, ZIP_NAME, deny_training_reads
    NeighborRegressor.initialize = NeighborRegressor.train = forbidden
    torch.optim.AdamW = forbidden
    sys.addaudithook(deny_training_reads)
    manifest = read(RUN / 'manifest.json'); warm = read(RUN / 'warm.json'); audit = read(RUN / 'cold.json')
    if (audit['status'] != 'passed' or audit['warm_sha256'] != sha(RUN / 'warm.json')
            or warm['manifest_sha256'] != sha(RUN / 'manifest.json')
            or warm['predictions_sha256'] != sha(RUN / 'warm-predictions.npz')):
        raise ValueError('Independent native audit required')
    verify({p: h for p, h in manifest['files'].items() if Path(p).is_relative_to(SOURCE / 'src')
            or Path(p) in (Path(__file__), PROTOCOL, RUN / 'query.json', RUN / 'parent.csv')})
    # This second process reads no training inputs; the bank is a persisted part of the model.
    query = pd.read_json(RUN / 'query.json', orient='table')
    metadata, state, _, arrays = read_state(RUN / 'refit.npz', warm['artifacts']['refit'])
    member = numpy_predict(query, metadata, state, arrays['bank_x'], arrays['bank_y'])
    with np.load(RUN / 'warm-predictions.npz', allow_pickle=False) as p:
        difference = float(np.max(np.abs(member - p['refit'])))
        if difference > 1e-8 or p['query_ids'].tolist() != query.sample_id.tolist():
            raise ValueError('Label-free query roundtrip changed prediction')
        # Serialized package uses the warm vector; independently cold-check its arithmetic.
        content = payload((RUN / 'parent.csv').read_bytes(), query.sample_id.tolist(), p['refit'])
    parent_rows = validate_result((RUN / 'parent.csv').read_bytes(), query.sample_id)
    rows = validate_result(content, query.sample_id)
    arithmetic = max(abs(float(r['pred_tap_time_len']) - (.8*float(old['pred_tap_time_len']) + .2*float(v)))
                     for r, old, v in zip(rows, parent_rows, member))
    if arithmetic > 1e-8 or any(r['pred_tap_iron'] != old['pred_tap_iron'] for r, old in zip(rows, parent_rows)):
        raise ValueError('Independent package arithmetic/iron strings differ')
    output = RUN / NAME; output.mkdir(exist_ok=False)
    package(output, content, query.sample_id)
    if zip_bytes(output / ZIP_NAME) != content:
        raise ValueError('Package readback differs')
    write(RUN / 'package-audit.json', dict(status='passed', candidate=NAME, rows=322, unique_ids=322,
          template_order=True, members=['result.csv'], CRC=True, finite_nonnegative=True,
          iron_string_mismatches=0, maximum_cold_difference=difference,
          maximum_arithmetic_difference=arithmetic, package=str(output / ZIP_NAME),
          zip_sha256=sha(output / ZIP_NAME), csv_sha256=sha(output / 'result.csv'),
          parent_sha256=PARENT_SHA, cold_audit_sha256=sha(RUN / 'cold.json'), peak_rss_mib=memory(),
          G1='manual_platform_exploration_original_development_gate_failed_not_formally_promoted'))
    print(json.dumps(dict(status='package_passed', zip_sha256=sha(output / ZIP_NAME))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'fit', 'cold', 'package'])
    parser.add_argument('--checks')
    args = parser.parse_args(); setup()
    if args.stage == 'freeze':
        freeze(args.checks)
    else:
        {'fit': fit, 'cold': cold, 'package': package_cold}[args.stage]()
