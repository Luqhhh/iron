"""Explicitly authorized V7 isolated time release; never uploads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import yaml

from .data import FEATURES, SUBMISSION_COLUMNS, TARGETS
from .submission import ZIP_NAME, deny_training_reads, package, validate_result
from .v2_release import load_v2
from .v5_library import fold_vector, load_v5_training_frame
from .v5_package import payload_with_parent_other_column
from .v5_spec import load_v5_spec
from .v7_periodic import PeriodicRegressor, digest, file_hash, write_new


def parent_payload(path, expected_sha, ids):
    if file_hash(path) != expected_sha:
        raise ValueError('Parent ZIP identity mismatch')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['result.csv'] or archive.testzip() is not None:
            raise ValueError('Invalid parent archive')
        payload = archive.read('result.csv')
    validate_result(payload, ids)
    return payload


def blended_payload(parent, ids, member, alpha):
    rows = validate_result(parent, ids)
    values = np.asarray(member, dtype=float)
    if values.shape != (len(ids),) or not np.isfinite(values).all():
        raise ValueError('Invalid time member predictions')
    if not 0 <= alpha <= 1:
        raise ValueError('Invalid release blend weight')
    base = np.asarray([float(r['pred_tap_time_len']) for r in rows])
    raw = (1-alpha)*base + alpha*values
    changed = np.maximum(raw, 0.)
    payload = payload_with_parent_other_column(ids, 'tap_time_len', changed,
                                               [r['pred_tap_iron'] for r in rows])
    return payload, {'negative_rows_clipped': int((raw < 0).sum()), 'raw_minimum': float(raw.min())}


def verify_payload(payload, parent, ids, member, alpha):
    original = validate_result(parent, ids)
    rows = validate_result(payload, ids)
    mismatch = sum(a['pred_tap_iron'] != b['pred_tap_iron'] for a, b in zip(rows, original))
    expected = np.maximum((1-alpha)*np.asarray([float(r['pred_tap_time_len']) for r in original])
                          + alpha*np.asarray(member), 0.)
    actual = np.asarray([float(r['pred_tap_time_len']) for r in rows])
    difference = float(np.max(np.abs(actual-expected)))
    if mismatch or not np.array_equal(actual, expected):
        raise ValueError('Isolated-column readback failed')
    return {'rows': len(rows), 'unique_ids': len({r['sample_id'] for r in rows}),
            'template_order': True, 'iron_string_mismatches': mismatch,
            'blend_maximum_absolute_difference': difference}


def cold_infer(work):
    """Fresh process entry point: no labels or training feature reads allowed."""
    import torch
    torch.set_num_threads(1)
    sys.addaudithook(deny_training_reads)
    work = Path(work)
    with (work/'model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    query = pd.read_pickle(work/'query.pkl')
    if any(t in query for t in TARGETS):
        raise ValueError('Cold query contains labels')
    prediction = model.predict(query)
    reversed_prediction = model.predict(query.iloc[::-1])[::-1]
    cuts = [query.iloc[:1], query.iloc[1:111], query.iloc[111:]]
    chunks = np.concatenate([model.predict(part) for part in cuts if len(part)])
    singleton = model.predict(query.iloc[:1])
    scale = max(1., float(np.max(np.abs(prediction))))
    differences = {'reverse': float(np.max(np.abs(prediction-reversed_prediction))),
                   'chunk': float(np.max(np.abs(prediction-chunks))),
                   'singleton': float(np.max(np.abs(prediction[:1]-singleton)))}
    config = json.loads((work/'inference.json').read_text())
    if max(differences.values())/scale > config['relative_tolerance']:
        raise ValueError('Cold row invariance failed')
    with (work/'cold.npy').open('xb') as stream:
        np.save(stream, prediction)
    if (work/'parent.csv').exists():
        payload, _ = blended_payload((work/'parent.csv').read_bytes(), query.sample_id.tolist(),
                                      prediction, config['alpha'])
        with (work/'cold-result.csv').open('xb') as stream:
            stream.write(payload)
    write_new(work/'cold.json', {'training_reads_prohibited': True, 'relative_scale': scale,
                                'query_differences': differences,
                                'relative_tolerance': config['relative_tolerance'],
                                'model_sha256': file_hash(work/'model.pkl'),
                                'query_sha256': file_hash(work/'query.pkl')})


def save_and_cold(model, query, prediction, work, tolerance, parent=None, alpha=None):
    work.mkdir(exist_ok=False)
    # The optimizer is not required for inference. The fitted preprocessing,
    # model weights and target scaling are retained in the original estimator.
    del model.optimizer_
    with (work/'model.pkl').open('xb') as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    clean_query = query.drop(columns=[t for t in TARGETS if t in query])
    with (work/'query.pkl').open('xb') as stream:
        pickle.dump(clean_query, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with (work/'warm.npy').open('xb') as stream:
        np.save(stream, prediction)
    write_new(work/'inference.json', {'relative_tolerance': tolerance, 'alpha': alpha})
    if parent is not None:
        with (work/'parent.csv').open('xb') as stream:
            stream.write(parent)
    subprocess.run([sys.executable, '-m', 'bf_tap_r2.v7_release', 'infer', '--work', str(work)], check=True)
    cold = np.load(work/'cold.npy', allow_pickle=False)
    if not np.array_equal(prediction, cold):
        raise ValueError('Cold-process predictions differ')
    return {'maximum_absolute_difference': float(np.max(np.abs(prediction-cold))),
            'bit_identical': True, **json.loads((work/'cold.json').read_text())}


def append_event(out, event):
    with (out/'fit_ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(event, allow_nan=False)+'\n')


def run(root, release_spec=Path('configs/round2_v7/RELEASE.yaml')):
    root = Path(root).resolve()
    release_path = (root/release_spec).resolve()
    if not release_path.is_relative_to(root/'configs/round2_v7'):
        raise ValueError('V7 release specification required')
    release = yaml.safe_load(release_path.read_text())
    spec_path = root/release['model_spec']; spec = yaml.safe_load(spec_path.read_text())
    out = (root/release['output']).resolve(); desktop = Path(release['desktop'])
    if not out.is_relative_to(root/'local/runs/round2-v7-periodic-networks'):
        raise ValueError('Private V7 release output required')
    if desktop.exists():
        raise FileExistsError('Desktop destination already exists; refusing overwrite')
    out.mkdir(parents=True, exist_ok=False)
    try:
        dev = root/release['development_run']; confirm = root/release['confirmation_run']
        dev_manifest = json.loads((dev/'manifest.json').read_text())
        confirm_summary = json.loads((confirm/'summary.json').read_text())
        audit_path = confirm/release['confirmation_audit']; audit = json.loads(audit_path.read_text())
        if (audit['status'] != 'PASS' or not confirm_summary['seed_gate']['admitted']
                or audit['manual_blend_arithmetic']['lcb95'] <= 0
                or not all(g > 0 for g in audit['manual_blend_arithmetic']['seed_gains'].values())
                or set(audit['manual_blend_arithmetic']['alphas'].values()) != {release['blend_weight']}):
            raise ValueError('Frozen V7 confirmation or release weight mismatch')
        if file_hash(spec_path) != dev_manifest['spec_sha256']:
            raise ValueError('Historical V7 specification changed')
        for name, sha in dev_manifest['dependency_code_hashes'].items():
            if file_hash(root/'src/bf_tap_r2'/name) != sha:
                raise ValueError('Historical V7 source changed')
        for relative, sha in audit['prediction_hashes'].items():
            if file_hash(root/relative) != sha:
                raise ValueError('Historical V7 predictions changed')
        versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
        if versions != spec['runtime_versions']:
            raise ValueError('Frozen V7 runtime mismatch')
        train = load_v5_training_frame(root); test = load_v2(root/'复赛_test', 'test', 322)
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(train, index=True).values.tobytes()).hexdigest()
        if data_hash != dev_manifest['data_digest'] or len(train) != release['full_training']['rows']:
            raise ValueError('Training identity mismatch')
        ids = test.sample_id.tolist()
        parent_zip = Path(release['parent']['zip'])
        parent = parent_payload(parent_zip, release['parent']['sha256'], ids)
        files = [release_path, spec_path, parent_zip, root/'uv.lock', audit_path, confirm/'summary.json',
                 *sorted((root/'src/bf_tap_r2').glob('*.py'))]
        files += [root/f'复赛_{stage}/{stage}_{kind}.csv' for stage in ['train', 'test'] for kind in ['samples', 'features']]
        files += [root/'复赛_test/result_template.csv']
        hashes = {str(p): file_hash(p) for p in files}
        write_new(out/'manifest.json', {'candidate': release['candidate'], 'authorization': release['authorization'],
                  'files': hashes, 'data_digest': data_hash, 'versions': versions,
                  'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()})
        recipe, settings = spec['recipes'][release['recipe']], spec['training']
        seed, fold = release['replay']['split_seed'], release['replay']['fold']
        folds = fold_vector(root, train, seed, load_v5_spec(root))
        if digest(folds.tolist()) != dev_manifest['fold_digests'][str(seed)]:
            raise ValueError('Replay fold identity mismatch')
        mask = folds == fold
        append_event(out, {'event': 'started', 'phase': 'replay', 'seed': seed, 'fold': fold})
        replay_train = train.loc[~mask].reset_index(drop=True)
        model = PeriodicRegressor(recipe, settings).fit(replay_train, replay_train.tap_time_len.values)
        replay = model.predict(train.loc[mask])
        expected = np.load(dev/f'tap_time_len-{release["recipe"]}-s{seed}-f{fold}.npy')
        if not np.array_equal(replay, expected):
            raise ValueError('Official-fold release replay is not bit-identical')
        replay_cold = save_and_cold(model, train.loc[mask], replay, out/'replay',
                                    release['validation']['row_order_and_chunk_relative_tolerance'])
        append_event(out, {'event': 'complete', 'phase': 'replay', 'metadata': model.metadata_, 'cold': replay_cold})
        print(json.dumps({'replay': 'passed', 'maximum_absolute_difference': 0.0}), flush=True)
        append_event(out, {'event': 'started', 'phase': 'full_data', 'rows': len(train)})
        full = PeriodicRegressor(recipe, settings).fit(train.reset_index(drop=True), train.tap_time_len.values)
        member = full.predict(test)
        cold = save_and_cold(full, test, member, out/'full',
                             release['validation']['row_order_and_chunk_relative_tolerance'], parent, release['blend_weight'])
        append_event(out, {'event': 'complete', 'phase': 'full_data', 'metadata': full.metadata_, 'cold': cold})
        payload, clipping = blended_payload(parent, ids, member, release['blend_weight'])
        if payload != (out/'full/cold-result.csv').read_bytes():
            raise ValueError('Cold-process submission CSV bytes differ')
        package_dir = out/release['candidate']; package_dir.mkdir(exist_ok=False)
        package(package_dir, payload, ids)
        with zipfile.ZipFile(package_dir/ZIP_NAME) as archive:
            if archive.namelist() != ['result.csv'] or archive.testzip() is not None:
                raise ValueError('Invalid release archive')
            archived = archive.read('result.csv')
        if archived != payload:
            raise ValueError('Release archive content differs')
        readback = verify_payload(archived, parent, ids, member, release['blend_weight'])
        if hashes != {str(p): file_hash(p) for p in files}:
            raise ValueError('Source or input identity changed during release')
        verification = {'G0': 'PASS', 'candidate': release['candidate'], 'readback': readback, 'clipping': clipping,
                         'replay_cold': replay_cold, 'full_cold': cold, 'full_fit_metadata': full.metadata_,
                         'zip_sha256': file_hash(package_dir/ZIP_NAME), 'result_sha256': file_hash(package_dir/'result.csv'),
                         'parent_zip_sha256': release['parent']['sha256'], 'release_alpha': release['blend_weight'],
                         'local_working_gate_exception': 'explicit_user_authorization_this_candidate_only',
                         'G1': {'platform_score': None, 'local_confirmation': audit['manual_blend_arithmetic']},
                         'uploads': 0}
        write_new(out/'verification.json', verification)
        desktop.mkdir(parents=True, exist_ok=False)
        with (package_dir/ZIP_NAME).open('rb') as source, (desktop/ZIP_NAME).open('xb') as destination:
            shutil.copyfileobj(source, destination)
        if file_hash(desktop/ZIP_NAME) != verification['zip_sha256']:
            raise ValueError('Desktop copy hash mismatch')
        with zipfile.ZipFile(desktop/ZIP_NAME) as archive:
            if archive.namelist() != ['result.csv'] or archive.read('result.csv') != payload:
                raise ValueError('Desktop archive verification failed')
        with (desktop/'README.txt').open('x', encoding='utf-8') as stream:
            stream.write(f"V7_TIME_PLR001_A50\n提交文件：{ZIP_NAME}\n仅替换时长列：0.5 × A35 + 0.5 × V7。铁量保持 A35 原字符串。\n322行与模板顺序一致，冷进程预测完全一致。\nZIP SHA-256: {verification['zip_sha256']}\n用户自行上传；尚无平台成绩，不保证达到96.4。\n")
        write_new(out/'delivery.json', {'desktop': str(desktop), 'zip': str(desktop/ZIP_NAME),
                  'zip_sha256': verification['zip_sha256'], 'copy_verified': True, 'uploads': 0})
        print(json.dumps({'status': 'delivered', 'zip': str(desktop/ZIP_NAME), 'sha256': verification['zip_sha256']}), flush=True)
    except Exception as exc:
        append_event(out, {'event': 'failed', 'error': repr(exc)})
        write_new(out/'FAILED.json', {'error': repr(exc), 'evidence_preserved': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['release', 'infer'])
    parser.add_argument('--work', type=Path)
    parser.add_argument('--spec', type=Path, default=Path('configs/round2_v7/RELEASE.yaml'))
    args = parser.parse_args()
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        if os.environ.get(name) != '1':
            parser.error(f'Set {name}=1')
    if args.mode == 'infer':
        if args.work is None:
            parser.error('--work required for inference')
        cold_infer(args.work)
    else:
        run(Path.cwd(), args.spec)


if __name__ == '__main__':
    main()
