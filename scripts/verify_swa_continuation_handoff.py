"""Verify portable Q75 references without fitting, labels or pickle loading."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def child(root, name):
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts and bool(rel.parts),
            'Unsafe payload path')
    path = Path(root) / str(rel)
    require(path.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink(),
            'Payload path escapes bundle')
    return path


def read(root, name):
    return json.loads(child(root, name).read_text())


def verify_bundle(root, expected_manifest_sha256):
    root = Path(root).resolve()
    require(sha(root / 'HANDOFF_MANIFEST.json') == expected_manifest_sha256,
            'External handoff manifest identity differs')
    manifest = read(root, 'HANDOFF_MANIFEST.json')
    actual_files = {str(p.relative_to(root)).replace('\\', '/')
                    for p in root.rglob('*') if p.is_file()}
    require(actual_files == set(manifest['payload_files']) | {'HANDOFF_MANIFEST.json'},
            'Unexpected or missing payload files')
    for name, item in manifest['payload_files'].items():
        path = child(root, name)
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Changed payload: ' + name)
    spec = read(root, 'CONFIRMATION_SPEC.json')
    require(spec['identity'] == 'TABM_TIME_SWA_CONFIRM_CONTINUATION_20261001'
            and spec['retained_development_seeds'] == [42, 3407]
            and spec['confirmation_seeds'] == [271828, 314159]
            and spec['original_confirmation_seeds'] == [7777, 12011], 'Wrong confirmation scope')
    require(spec['alpha'] == .2 and spec['confirmation_budget']['optimizer_initializations'] == 40
            and spec['confirmation_budget']['new_reference_fits'] == 0
            and spec['release_authorized'] is False, 'Wrong fixed recipe or budget')
    require(sha(child(root, 'ORIGINAL_SWA_SPEC.json')) == spec['original_spec_sha256'],
            'Original SWA protocol changed')
    original_spec = read(root, 'ORIGINAL_SWA_SPEC.json')
    require(original_spec['confirmation_seeds'] == [7777, 12011]
            and spec['averaging'] == original_spec['averaging'], 'Original averaging recipe changed')
    for name, digest in spec['source_hashes'].items():
        require(sha(child(root, 'source_snapshot/' + name)) == digest,
                'Frozen scientific source differs: ' + name)
    binding = read(root, 'refs/original-binding.json')
    require(sha(child(root, 'refs/original-binding.json')) == spec['reference_export_binding_sha256'],
            'Original Q75 export identity differs')
    require(binding['status'] == 'passed_zero_fit_existing_Q75_export'
            and binding['verified_split_seeds'] == spec['four_seed_order']
            and binding['new_fits'] == binding['label_values_read'] == 0
            and binding['usable_by_original_SWA_confirmation'] is False,
            'Reference coverage or original waiting record changed')
    require(binding['platform_package_sha256'] == spec['reference']['zip_sha256'],
            'Current Q75 package identity differs')
    mapping = manifest['original_path_to_payload']

    def original(path):
        require(path in mapping, 'Missing original evidence: ' + path)
        require(mapping[path] in manifest['payload_files'], 'Unlisted original evidence')
        return child(root, mapping[path])

    for path, digest in binding['frozen_original_evidence'].items():
        require(sha(original(path)) == digest, 'Original evidence changed: ' + path)
    require(sha(original(binding['ids']['path'])) == binding['ids']['sha256'],
            'Exported ID identity differs')
    ids = np.load(original(binding['ids']['path']), allow_pickle=False)
    require(ids.shape == (2754,) and len(set(ids.tolist())) == 2754
            and np.array_equal(ids, np.sort(ids)), 'Invalid explicit reference ID order')
    columns = {}
    for seed in spec['four_seed_order']:
        saved = binding['columns'][str(seed)]
        for item in saved.values():
            require(sha(original(item['path'])) == item['sha256'], 'Exported reference changed')
        time = np.load(original(saved['time']['path']), allow_pickle=False)
        folds = np.load(original(saved['folds']['path']), allow_pickle=False)
        require(time.shape == folds.shape == ids.shape and np.isfinite(time).all()
                and (time >= 0).all() and np.issubdtype(folds.dtype, np.integer)
                and set(folds.tolist()) == set(range(5)), 'Invalid complete same-seed column')
        is_confirmation = seed not in (42, 3407)
        source = binding['original_confirmation_source' if is_confirmation else 'original_development_source']
        audit = json.loads(original(source + '/audit.json').read_text())
        terminal = json.loads(original(source + '/terminal-verification-r1.json').read_text())
        require(audit['status'] == terminal['status'] == 'passed'
                and terminal.get('actual_controller_exit_code', terminal.get('controller_exit_code')) == 0,
                'Original actual successful audit/terminal missing')
        for name in ('manifest', 'summary', 'audit'):
            require(terminal[name + '_sha256'] == sha(original(source + '/' + name + '.json')),
                    'Original terminal ancestry differs')
        for name in ('manifest', 'summary'):
            require(audit[name + '_sha256'] == sha(original(source + '/' + name + '.json')),
                    'Original audit ancestry differs')
        for fold in range(5):
            unit_name = f's{seed}-f{fold}' if is_confirmation else f'SHORT_SPAN-s{seed}-f{fold}'
            unit = source + '/' + unit_name
            receipt_name = 'warm-complete.json' if is_confirmation else 'complete.json'
            receipt = json.loads(original(unit + '/' + receipt_name).read_text())
            require(receipt['manifest_sha256'] == sha(original(source + '/manifest.json')),
                    'Original unit manifest differs')
            npz_path = original(unit + '/predictions.npz')
            if is_confirmation:
                cold = json.loads(original(unit + '/cold-complete.json').read_text())
                require(cold['status'] == 'passed' and cold['new_fits'] == 0
                        and cold['warm_receipt_sha256'] == sha(original(unit + '/' + receipt_name))
                        and receipt['predictions_sha256'] == sha(npz_path), 'Original fold cold chain differs')
                require(receipt['reference_receipt_sha256'] == sha(original(unit + '/reference/complete.json')),
                        'Original same-seed reference receipt differs')
            else:
                for name, digest in receipt['hashes'].items():
                    require(sha(original(unit + '/' + name)) == digest, 'Original development unit changed')
            with np.load(npz_path, allow_pickle=False) as a:
                qids = a['query_ids'].astype(str)
                # Align by ID explicitly; original fold/query orders remain untouched.
                require(set(qids.tolist()) == set(ids[folds == fold].tolist()), 'Different fold population')
                ix = np.searchsorted(ids, qids)
                require(np.array_equal(ids[ix], qids) and np.array_equal(time[ix], a['q75']),
                        'Exported Q75 differs from original same-seed query')
                if is_confirmation:
                    require(np.array_equal(a['q75'], a['tap_time_len'] + .75 * (a['old_ema'] - a['v7_time'])),
                            'Original Q75 arithmetic differs')
                    require(np.array_equal(a['iron'], a['tap_iron']), 'Original Q75 iron differs')
                else:
                    require(np.array_equal(a['candidate'], a['q75'] + .75 * (a['ema'] - a['old_ema'])),
                            'Original development arithmetic differs')
        columns[seed] = {'time': time, 'folds': folds}
    return {'spec': spec, 'ids': ids, 'columns': columns, 'manifest': manifest}


def reindex_reference(verified, seed, native_ids, native_folds=None):
    require(seed in verified['columns'], 'Unregistered reference seed')
    ids = np.asarray(native_ids, dtype=str)
    exported_ids = verified['ids']
    require(ids.shape == exported_ids.shape and len(set(ids.tolist())) == len(ids)
            and np.array_equal(np.sort(ids), exported_ids), 'Native row population differs')
    ix = np.searchsorted(exported_ids, ids)
    column = verified['columns'][seed]
    folds = column['folds'][ix]
    if native_folds is not None:
        require(np.array_equal(folds, np.asarray(native_folds)), 'Native split seed/folds differ')
    return column['time'][ix].copy(), folds.copy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    checked = verify_bundle(args.directory, args.expected_manifest_sha256)
    print(json.dumps({'status': 'passed_portable_Q75_reference_identity_and_ancestry',
                      'seeds': checked['spec']['four_seed_order'], 'rows_per_seed': len(checked['ids']),
                      'new_fits': 0, 'parsed_label_values': 0,
                      'scientific_execution_started': False}, ensure_ascii=False, indent=2))
