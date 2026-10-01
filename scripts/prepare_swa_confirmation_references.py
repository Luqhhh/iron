"""Export existing audited Q75 columns without fits, label reads or seed changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (42, 3407, 271828, 314159)
ORIGINAL_SWA_CONFIRMATION_SEEDS = (7777, 12011)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(base, hashes):
    for name, expected in hashes.items():
        require(sha(Path(base) / name) == expected, 'Frozen file changed: ' + name)


def close_seed(parts, expected_ids=None):
    """Close exactly five disjoint folds into one explicit ID order."""
    require(len(parts) == 5, 'Five complete folds required')
    ids = np.concatenate([part['ids'] for part in parts])
    require(len(ids) == 2754 and len(set(ids.tolist())) == 2754,
            'Incomplete or duplicated same-seed OOF IDs')
    order = np.sort(ids) if expected_ids is None else np.asarray(expected_ids)
    require(np.array_equal(np.sort(ids), np.sort(order)), 'Different seed populations')
    positions = {name: index for index, name in enumerate(order.tolist())}
    predictions = np.full(2754, np.nan)
    folds = np.full(2754, -1, dtype=np.int64)
    for fold, part in enumerate(parts):
        values = part['q75']
        require(values.shape == part['ids'].shape and np.isfinite(values).all()
                and (values >= 0).all(), 'Invalid Q75 column')
        indices = [positions[name] for name in part['ids'].tolist()]
        predictions[indices] = values
        folds[indices] = fold
    require(np.isfinite(predictions).all() and set(folds.tolist()) == set(range(5)),
            'Same-seed OOF coverage failed')
    return order, predictions, folds


def checked_context(main, source, state, confirmation):
    manifest = record(source / 'manifest.json')
    audit = record(source / 'audit.json')
    terminal = record(source / 'terminal-verification-r1.json')
    require(sha(source / 'manifest.json') == state['manifest_sha256'], 'State manifest identity differs')
    require(audit['status'] == terminal['status'] == 'passed', 'Successful original audit required')
    require(terminal.get('actual_controller_exit_code', terminal.get('controller_exit_code')) == 0,
            'Actual successful terminal required')
    for name in ('manifest', 'summary', 'audit'):
        require(terminal[name + '_sha256'] == sha(source / (name + '.json')),
                'Terminal identity differs: ' + name)
    for name in ('manifest', 'summary'):
        require(audit[name + '_sha256'] == sha(source / (name + '.json')),
                'Original audit identity differs: ' + name)
    if confirmation:
        evidence = state['terminal_evidence']
        for key in ('audit', 'process_terminal', 'terminal_verification'):
            item = evidence[key]
            require(sha(main / item['path']) == item['sha256'], 'Published terminal anchor differs')
    else:
        require(sha(source / 'audit.json') == state['audit_sha256'], 'Published audit anchor differs')
        require(sha(source / 'terminal-verification-r1.json') == state['terminal_verification_sha256'],
                'Published terminal anchor differs')
    verify_files(manifest['workspace'], manifest['sources'])
    verify_files(main, manifest['spec']['inputs'])
    if confirmation:
        verify_files('/', manifest['model_sources'])
    return manifest


def checked_unit(source, manifest, seed, fold, confirmation):
    unit = source / (f's{seed}-f{fold}' if confirmation else f'SHORT_SPAN-s{seed}-f{fold}')
    require(not (unit / 'failure.json').exists(), 'Failed unit cannot supply an admitted reference')
    receipt_name = 'warm-complete.json' if confirmation else 'complete.json'
    complete = record(unit / receipt_name)
    require(complete['manifest_sha256'] == sha(source / 'manifest.json'), 'Unit manifest differs')
    if confirmation:
        cold = record(unit / 'cold-complete.json')
        require(cold['status'] == 'passed' and cold['new_fits'] == 0
                and cold['warm_receipt_sha256'] == sha(unit / receipt_name), 'Unit cold provenance differs')
        require(sha(unit / 'predictions.npz') == complete['predictions_sha256'], 'Unit predictions changed')
        require(sha(unit / 'reference/complete.json') == complete['reference_receipt_sha256'],
                'Matching original reference receipt changed')
    else:
        verify_files(unit, complete['hashes'])
    with np.load(unit / 'predictions.npz', allow_pickle=False) as saved:
        ids = saved['query_ids'].astype(str)
        q75 = saved['q75'].copy()
        if confirmation:
            expected = saved['tap_time_len'] + .75 * (saved['old_ema'] - saved['v7_time'])
            require(np.array_equal(q75, expected), 'Original Q75 recipe differs')
            require(np.array_equal(saved['iron'], saved['tap_iron']), 'Original Q75 iron differs')
        else:
            # The original development audit binds these OLD_EMA/Q75 inputs.
            require(np.array_equal(saved['candidate'], q75 + .75 * (saved['ema'] - saved['old_ema'])),
                    'Original development composition differs')
    require(ids.ndim == 1 and ids.shape == q75.shape, 'Query identity shape differs')
    paths = [unit / receipt_name, unit / 'predictions.npz']
    if confirmation:
        paths += [unit / 'cold-complete.json', unit / 'reference/complete.json']
    else:
        paths += [unit / name for name in complete['hashes']]
    return {'ids': ids, 'q75': q75}, {str(path): sha(path) for path in paths}


def prepare(main, output):
    main = Path(main).resolve()
    output = Path(output).resolve()
    require(not output.exists(), 'Existing reference export cannot be overwritten')
    require(output.is_relative_to(Path(__file__).resolve().parents[1] / 'local'),
            'Export must stay in this worktree private local directory')
    state = record(main / 'EVIDENCE_STATUS.json')
    current = state['round2_current_platform_best']
    require(current['candidate'] == 'EMA_TIME_Q75'
            and current['zip_sha256'] == '41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825',
            'Current platform reference changed; reassess before another phase')
    development = main / 'local/runs/ema-average-span-20261001/development-r1'
    confirmation = main / 'local/runs/ema-span-confirmation-20261001/confirmation-r1'
    contexts = {
        False: checked_context(main, development, state['ema_average_span_20261001'], False),
        True: checked_context(main, confirmation, state['ema_span_confirmation_20261001'], True),
    }
    columns = {}
    frozen = {}
    population = None
    for seed in SEEDS:
        is_confirmation = seed not in (42, 3407)
        source = confirmation if is_confirmation else development
        parts = []
        for fold in range(5):
            part, hashes = checked_unit(source, contexts[is_confirmation], seed, fold, is_confirmation)
            parts.append(part)
            frozen.update(hashes)
        ids, prediction, folds = close_seed(parts, population)
        if population is None:
            population = ids
        columns[seed] = (prediction, folds)
    for source in (development, confirmation):
        for name in ('manifest.json', 'summary.json', 'audit.json', 'terminal-verification-r1.json'):
            frozen[str(source / name)] = sha(source / name)
    # All validation occurs before consuming the new export directory.
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'sample_ids.npy').open('xb') as stream:
        np.save(stream, population, allow_pickle=False)
    exported = {}
    for seed, (prediction, folds) in columns.items():
        files = {}
        for name, value in [('time', prediction), ('folds', folds)]:
            path = output / f's{seed}-{name}.npy'
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), value)
            files[name] = {'path': str(path), 'sha256': sha(path)}
        exported[str(seed)] = files
    binding = {
        'status': 'passed_zero_fit_existing_Q75_export',
        'candidate': 'EMA_TIME_Q75',
        'platform_score': current['score'],
        'platform_score_source': current['source'],
        'platform_package_sha256': current['zip_sha256'],
        'verified_split_seeds': list(SEEDS),
        'rows_per_seed': 2754,
        'folds_per_seed': 5,
        'row_order': 'sample_id_lexicographic_explicit_reindex_required_for_native_frame',
        'ids': {'path': str(output / 'sample_ids.npy'), 'sha256': sha(output / 'sample_ids.npy')},
        'columns': exported,
        'frozen_original_evidence': frozen,
        'original_development_source': str(development),
        'original_confirmation_source': str(confirmation),
        'original_SWA_required_missing_seeds': list(ORIGINAL_SWA_CONFIRMATION_SEEDS),
        'usable_by_original_SWA_confirmation': False,
        'scientific_phase_started': False,
        'new_fits': 0,
        'label_values_read': 0,
        'new_reference_fits': 0,
        'new_packages': 0,
        'original_freezes_and_failed_decisions_changed': False,
        'exporter_sha256': sha(__file__),
    }
    with (output / 'binding.json').open('x') as stream:
        json.dump(binding, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    verify_files('/', frozen)
    print(json.dumps({'status': binding['status'], 'binding': str(output / 'binding.json'),
                      'binding_sha256': sha(output / 'binding.json'),
                      'seeds': list(SEEDS), 'rows_per_seed': 2754,
                      'original_SWA_required_missing_seeds': list(ORIGINAL_SWA_CONFIRMATION_SEEDS),
                      'new_fits': 0, 'label_values_read': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', default='/home/lux1/iron')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.main, args.output)
