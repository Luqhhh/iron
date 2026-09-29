#!/usr/bin/env python3
"""Audit a frozen three-seed ensemble from authorized training OOF predictions.

No training, model selection, weight optimization or platform access is performed.
CSV columns are FULL target predictions after fixed component replacement, NOT
raw V12/V7 component outputs. Provenance/leakage safety requires a separate audit.
Only the non-clipped score regime is covered by the 50*WMAPE contribution formula.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

MEMBERS = ('seed_42_prediction', 'seed_104729_prediction', 'seed_130363_prediction')
REQUIRED = ('target', 'split_seed', 'fold', 'sample_id', 'actual',
            'current_prediction', *MEMBERS)
TARGETS = {'tap_iron', 'tap_time_len'}


def _finite(value: str, field: str, line: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Line {line}: invalid numeric value in {field}') from exc
    if not math.isfinite(number):
        raise ValueError(f'Line {line}: nonfinite value in {field}')
    if number < 0:
        raise ValueError(f'Line {line}: negative value in {field}')
    return number


def decompose(actual: list[float], current: list[float],
              members: list[list[float]]) -> dict[str, Any]:
    """Exact absolute-error decomposition; does not imply improvement over current."""
    n = len(actual)
    if not n or not members or len(current) != n or any(len(p) != n for p in members):
        raise ValueError('Empty or inconsistent prediction dimensions')
    if any(not math.isfinite(x) or x < 0 for p in [actual, current, *members] for x in p):
        raise ValueError('All labels/predictions must be finite and nonnegative')
    denominator = math.fsum(actual)
    if denominator <= 0:
        raise ValueError('WMAPE denominator must be positive')
    size = len(members)
    averaged = [math.fsum(p[i] for p in members) / size for i in range(n)]
    errors = [math.fsum(abs(y - a) for y, a in zip(actual, p)) / denominator
              for p in members]
    base_error = math.fsum(abs(y-a) for y, a in zip(actual, current)) / denominator
    ensemble_error = math.fsum(abs(y-a) for y, a in zip(actual, averaged)) / denominator
    mean_member_error = math.fsum(errors) / size
    mean_member_gain = 50 * (base_error - mean_member_error)
    jensen_gain = 50 * (mean_member_error - ensemble_error)
    net_gain = 50 * (base_error - ensemble_error)
    identity_error = abs(net_gain - (mean_member_gain + jensen_gain))
    if jensen_gain < -1e-10 or identity_error > 1e-10:
        raise ArithmeticError('Absolute-error decomposition failed')
    return {
        'rows': n, 'member_count': size, 'wmape_current': base_error,
        'wmape_ensemble': ensemble_error,
        'member_isolated_score_gains': [50 * (base_error-e) for e in errors],
        'mean_member_quality_gain': mean_member_gain,
        'absolute_error_cancellation_gain': jensen_gain,
        'net_isolated_score_gain': net_gain,
        'decomposition_max_error': identity_error,
        'mean_absolute_prediction_change': math.fsum(
            abs(p-b) for p, b in zip(averaged, current)) / n,
    }


def audit(path: Path, expected_seeds: tuple[str, ...] = ('42', '3407'),
          expected_folds: int = 5) -> dict[str, Any]:
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError('Expected split seeds must be nonempty and unique')
    if expected_folds < 2:
        raise ValueError('At least two outer folds are required')
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError('Duplicate CSV column names')
        missing = set(REQUIRED) - set(fields)
        if missing:
            raise ValueError(f'Missing CSV columns: {sorted(missing)}')
        for line, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f'Line {line}: too many CSV fields')
            if any(raw.get(key) is None for key in REQUIRED):
                raise ValueError(f'Line {line}: missing CSV value')
            target, seed, sample = (raw[k].strip() for k in ('target', 'split_seed', 'sample_id'))
            if target not in TARGETS or seed not in expected_seeds or not sample:
                raise ValueError(f'Line {line}: invalid target, split seed or sample_id')
            try:
                fold = int(raw['fold'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Line {line}: invalid fold') from exc
            if not 0 <= fold < expected_folds:
                raise ValueError(f'Line {line}: fold outside expected range')
            key = target, seed, sample
            if key in seen:
                raise ValueError(f'Line {line}: duplicate OOF sample {key}')
            seen.add(key)
            row = {'sample_id': sample, 'fold': fold}
            for field in ('actual', 'current_prediction', *MEMBERS):
                row[field] = _finite(raw[field], field, line)
            if abs(row['current_prediction'] - row[MEMBERS[0]]) > 1e-10:
                raise ValueError(f'Line {line}: seed42 full prediction does not replay current')
            groups[target, seed].append(row)
    if not groups:
        raise ValueError('No data rows')
    output: dict[str, Any] = {
        'status': 'arithmetic_audit_only', 'provenance_verified': False,
        'training_runs': 0, 'weight_fits': 0, 'platform_actions': 0,
        'notes': [
            'Uses training OOF input supplied by the caller; membership/source hashes are not verified.',
            'Averages TRAINING seeds within each outer split/fold; never averages outer split seeds.',
            'Averaged-member score is not the score of averaged predictions.',
            'A nonnegative cancellation term does not guarantee a positive gain against incumbent.',
            'No promotion/confirmation decision or platform forecast is made.',
        ],
        'targets': {},
    }
    for target in sorted({key[0] for key in groups}):
        reference_labels: dict[str, float] | None = None
        per_seed = {}
        for seed in expected_seeds:
            rows = sorted(groups.get((target, seed), []), key=lambda r: r['sample_id'])
            if {r['fold'] for r in rows} != set(range(expected_folds)):
                raise ValueError(f'{target}/{seed}: incomplete outer-fold coverage')
            labels = {r['sample_id']: r['actual'] for r in rows}
            if reference_labels is None:
                reference_labels = labels
            elif reference_labels != labels:
                raise ValueError(f'{target}/{seed}: sample set or labels differ across outer splits')
            per_seed[seed] = decompose(
                [r['actual'] for r in rows], [r['current_prediction'] for r in rows],
                [[r[key] for r in rows] for key in MEMBERS])
        output['targets'][target] = {
            'split_results': per_seed,
            'mean_split_gain': math.fsum(v['net_isolated_score_gain'] for v in per_seed.values())
                               / len(per_seed),
            'positive_split_count': sum(v['net_isolated_score_gain'] > 0 for v in per_seed.values()),
            'split_count': len(per_seed),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON report; existing files are never overwritten')
    parser.add_argument('--split-seeds', default='42,3407')
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    try:
        report = audit(args.input, tuple(s.strip() for s in args.split_seeds.split(',')), args.folds)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    except (ValueError, ArithmeticError, OSError) as exc:
        parser.exit(2, f'Audit failed: {exc}\n')
    print(f'Wrote arithmetic report: {args.output}')


if __name__ == '__main__':
    main()
