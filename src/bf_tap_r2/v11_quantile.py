"""Training-only quantile coordinates for frozen from-scratch TabM recipes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import yaml

from .candidate_tiers import classify_candidates
from .data import FEATURES, TARGETS
from .v3_6_networks import NumericPreprocessor
from .v5_library import fold_vector, load_v5_training_frame
from .v5_resolution import nested_blend, package_score, wmape
from .v5_spec import load_v5_spec
from .v7_periodic import digest, file_hash, references, score_detail, write_new
from .v10_periodic_loss import LossRegressor


class QuantilePreprocessor(NumericPreprocessor):
    def __init__(self, distribution, policy):
        super().__init__(structure='raw_tabm')
        if distribution not in ('normal', 'uniform'):
            raise ValueError('Unregistered quantile distribution')
        self.distribution, self.policy = distribution, dict(policy)

    def fit(self, frame):
        super().fit(frame)
        x = frame[list(FEATURES)].to_numpy(dtype=np.float64)
        noisy = x + np.random.RandomState(self.policy['random_seed']).normal(
            0., self.policy['noise_std_raw_units'], x.shape)
        self.transformer_ = QuantileTransformer(
            n_quantiles=max(min(len(x)//30, 1000), 10), output_distribution=self.distribution,
            subsample=None, random_state=self.policy['random_seed'])
        self.transformer_.fit(noisy)
        self.noisy_fit_digest_ = hashlib.sha256(noisy.tobytes()).hexdigest()
        self.quantile_digest_ = hashlib.sha256(self.transformer_.quantiles_.tobytes()).hexdigest()
        return self

    def transform_tabm(self, frame):
        x = self.transformer_.transform(frame[list(FEATURES)].to_numpy(dtype=np.float64))
        if self.distribution == 'uniform':
            x = np.sqrt(12.)*(x-.5)
        return x.astype(np.float32), self._cat_codes(frame)[:, None]


class QuantileRegressor(LossRegressor):
    def __init__(self, recipe, settings, policy):
        super().__init__({**recipe, 'loss': 'mse'}, settings)
        self.policy = dict(policy)

    def _initialize(self, frame, y):
        # Preserve exact V7 model/RNG/optimizer initialization. This initial
        # raw scaler uses only the same training rows and is then replaced.
        super()._initialize(frame, y)
        distribution = self.recipe['distribution']
        trace = {'distribution': distribution, 'rows': len(frame), 'ids_digest': digest(frame.sample_id.tolist())}
        if distribution != 'standard':
            self.preprocessor_ = QuantilePreprocessor(distribution, self.policy).fit(frame)
            trace.update(noisy_fit_digest=self.preprocessor_.noisy_fit_digest_,
                         quantile_digest=self.preprocessor_.quantile_digest_,
                         n_quantiles=self.preprocessor_.transformer_.n_quantiles_)
        self.preprocessing_trace_.append(trace)

    def fit(self, frame, y):
        self.preprocessing_trace_ = []
        super().fit(frame, y)
        self.metadata_['preprocessing_trace'] = self.preprocessing_trace_
        return self


def evaluate_fold(frame, folds, target, recipe, settings, policy, fold):
    train = frame.loc[folds != fold].reset_index(drop=True)
    query = frame.loc[folds == fold].drop(columns=list(TARGETS)).reset_index(drop=True)
    model = QuantileRegressor(recipe, settings, policy).fit(train, train[target].values)
    return model.predict(query), model.metadata_


def load_oof(root, directory, frame, folds, target, recipe):
    directory = root/directory
    manifest = json.loads((directory/'manifest.json').read_text())
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    if data_hash != manifest['data_digest']:
        raise ValueError('Reference data identity mismatch')
    events = [json.loads(line) for line in (directory/'fit_ledger.jsonl').read_text().splitlines()]
    complete = [e for e in events if e['event'] == 'complete']
    bykey = {e['key']: e for e in complete}
    if len(bykey) != len(complete):
        raise ValueError('Duplicate reference fit records')
    result, hashes = {}, {str((directory/'manifest.json').relative_to(root)): file_hash(directory/'manifest.json'),
                          str((directory/'fit_ledger.jsonl').relative_to(root)): file_hash(directory/'fit_ledger.jsonl')}
    for seed, fv in folds.items():
        if digest(fv.tolist()) != manifest['fold_digests'][str(seed)]:
            raise ValueError('Reference fold identity mismatch')
        values = np.full(len(frame), np.nan)
        for fold in range(5):
            key = f'{target}-{recipe}-s{seed}-f{fold}'; event = bykey[key]
            path = directory/(key+'.npy'); sha = file_hash(path); mask = fv == fold
            if sha != event['prediction_sha256']:
                raise ValueError('Reference prediction hash mismatch')
            if event['metadata']['fit_ids_digest'] != digest(frame.loc[~mask, 'sample_id'].tolist()):
                raise ValueError('Reference training rows mismatch')
            p = np.load(path, allow_pickle=False)
            if p.shape != (int(mask.sum()),) or not np.isfinite(p).all():
                raise ValueError('Reference prediction coverage mismatch')
            values[mask] = p; hashes[str(path.relative_to(root))] = sha
        if not np.isfinite(values).all():
            raise ValueError('Incomplete reference OOF')
        result[seed] = values
    return result, hashes


def load_references(root, frame, folds, spec):
    audit_path = root/spec['primary_reference_audit']
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'passed':
        raise ValueError('Verified primary reference required')
    hashes = dict(audit['file_sha256'])
    for relative, sha in hashes.items():
        if file_hash(root/relative) != sha:
            raise ValueError('Primary reference changed')
    hashes[str(audit_path.relative_to(root))] = file_hash(audit_path)
    base, q20 = references(root, frame, spec)
    controls = {}
    for target in TARGETS:
        controls[target] = {}
        for recipe in ['tabm_raw', 'tabm_plr001']:
            p, h = load_oof(root, spec['v7_reference']['prediction_directory'], frame, folds, target, recipe)
            controls[target][recipe] = p; hashes.update(h)
    v9, h = load_oof(root, spec['v9_reference']['prediction_directory'], frame, folds, 'tap_iron', spec['v9_reference']['recipe'])
    hashes.update(h)
    current = {}
    for seed in folds:
        iron_alpha = spec['v9_reference']['blend_weight']; time_alpha = spec['v7_reference']['blend_weight']
        current[seed] = {'tap_iron': (1-iron_alpha)*base[seed]['tap_iron']+iron_alpha*v9[seed],
                         'tap_time_len': (1-time_alpha)*base[seed]['tap_time_len']+time_alpha*controls['tap_time_len']['tabm_plr001'][seed]}
    return base, q20, current, controls, hashes


def choose_confirmation(records, spec):
    eligible = [r for r in records if all(g > 0 for label in ['A35', 'CURRENT']
                                         for g in r['comparisons'][label]['seed_gains'].values())]
    if not eligible:
        return None
    best = min(eligible, key=lambda r: (-r['comparisons']['CURRENT']['seed_summary']['mean'],
                spec['tie_preference_by_target'][r['target']].index(r['recipe']), r['target']))
    return {'target': best['target'], 'recipe': best['recipe']}


def run(root, output):
    root = Path(root).resolve(); spec_path = root/'configs/round2_v11/SPEC.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    versions = {p: importlib.metadata.version(p) for p in spec['runtime_versions']}
    if versions != spec['runtime_versions']:
        raise ValueError('V11 runtime mismatch')
    out = (root/output).resolve()
    if not out.is_relative_to(root/'local/runs/round2-v11-quantile-representation'):
        raise ValueError('Private V11 output required')
    out.mkdir(parents=True, exist_ok=False)
    frame = load_v5_training_frame(root)
    folds = {s: fold_vector(root, frame, s, load_v5_spec(root)) for s in spec['split_seeds']}
    base, q20, current, controls, hashes = load_references(root, frame, folds, spec)
    write_new(out/'manifest.json', {'spec_sha256': file_hash(spec_path), 'versions': versions,
        'data_digest': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        'fold_digests': {s: digest(f.tolist()) for s, f in folds.items()}, 'reference_hashes': hashes,
        'source_hashes': {p.name: file_hash(p) for p in sorted((root/'src/bf_tap_r2').glob('*.py'))}})
    c = spec['control']; seed, fold = c['split_seed'], c['fold']
    pred, metadata = evaluate_fold(frame, folds[seed], c['target'],
        {'backbone': 'tabm', 'frequency': .01, 'distribution': 'standard'}, spec['training'], spec['quantile_policy'], fold)
    expected = controls[c['target']][c['recipe']][seed][folds[seed] == fold]
    with (out/'control.npy').open('xb') as stream:
        np.save(stream, pred)
    diff = float(np.max(np.abs(pred-expected)))
    write_new(out/'control.json', {'metadata': metadata, 'sha256': file_hash(out/'control.npy'), 'max_absolute_difference': diff})
    if not np.array_equal(pred, expected):
        raise ValueError('Standard-coordinate control failed; candidates not fitted')
    print(json.dumps({'control': 'passed', 'max_absolute_difference': diff}), flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=spec['budget']['workers']) as executor:
        jobs = {executor.submit(evaluate_fold, frame, folds[s], target, recipe, spec['training'], spec['quantile_policy'], f): (target, name, s, f)
                for target in spec['targets'] for name, recipe in spec['recipes'].items() for s in folds for f in range(5)}
        for future in as_completed(jobs):
            target, name, seed, fold = jobs[future]; key = f'{target}-{name}-s{seed}-f{fold}'
            try:
                p, meta = future.result(); path = out/(key+'.npy')
                with path.open('xb') as stream:
                    np.save(stream, p)
                event = {'event': 'complete', 'key': key, 'metadata': meta, 'prediction_sha256': file_hash(path)}
            except Exception as exc:
                failures += 1; event = {'event': 'failed', 'key': key, 'error': repr(exc)}
            with (out/'fit_ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
            print(json.dumps({k: v for k, v in event.items() if k != 'metadata'}), flush=True)
    if failures:
        raise RuntimeError(f'V11 failed fits retained: {failures}')
    records, metrics = [], {}
    for target in spec['targets']:
        y = frame[target].values
        metrics[target] = {spec['reference_by_target'][target]: {str(s): score_detail(y, current[s][target], folds[s], frame.spout_no.values) for s in folds}}
        for name, recipe in spec['recipes'].items():
            pred = {s: np.full(len(frame), np.nan) for s in folds}
            for s in folds:
                for f in range(5):
                    pred[s][folds[s] == f] = np.load(out/f'{target}-{name}-s{s}-f{f}.npy')
                if not np.isfinite(pred[s]).all():
                    raise ValueError('Incomplete V11 OOF')
            comparisons = {label: nested_blend(y, folds, {s: ref[s][target] for s in folds}, pred, spec['blend_grid'])
                           for label, ref in [('A35', base), ('Q20', q20), ('CURRENT', current)]}
            metrics[target][name], scores = {}, {}
            other = next(t for t in TARGETS if t != target)
            for s in folds:
                alpha = comparisons['CURRENT']['alphas'][s]
                blend = (1-alpha)*current[s][target]+alpha*pred[s]
                metrics[target][name][str(s)] = score_detail(y, blend, folds[s], frame.spout_no.values)
                # The other target remains A35, never an automatic joint blend.
                scores[s] = package_score(wmape(y, blend), wmape(frame[other].values, base[s][other]))
            control = 'tabm_raw' if recipe['frequency'] is None else 'tabm_plr001'
            records.append({'target': target, 'recipe': name, 'comparisons': comparisons,
                'single_wmape': {s: wmape(y, pred[s]) for s in folds},
                'standard_control_wmape': {s: wmape(y, controls[target][control][s]) for s in folds},
                'development_package_scores': scores})
    result = {'status': 'development_complete', 'records': records,
        'candidate_tiers': classify_candidates(metrics, spec, yaml.safe_load((root/'configs/candidate_tiers.yaml').read_text())),
        'selected_confirmation': choose_confirmation(records, spec), 'packages': 0, 'uploads': 0, 'release_authorized': False}
    write_new(out/'summary.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        if os.environ.get(name) != '1':
            parser.error(f'Set {name}=1')
    run(Path.cwd(), args.output)


if __name__ == '__main__':
    main()
