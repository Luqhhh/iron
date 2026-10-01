"""Independent artifact and arithmetic audit without importing the calculator."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out):
    out = Path(out)
    report = json.loads((out/'report.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    spec = manifest['spec']
    root = Path(spec['main_root'])
    for name, expected in spec['inputs'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)
    for name, expected in manifest['execution_files'].items():
        if sha(name) != expected:
            raise ValueError('Execution source changed: '+name)
    for name, expected in report['artifact_hashes'].items():
        if sha(out/name) != expected:
            raise ValueError('Output changed: '+name)
    actual = pd.read_csv(root/'复赛_train/train_samples.csv', dtype={'sample_id': str})
    features = pd.read_csv(root/'复赛_train/train_features.csv', dtype={'sample_id': str}).set_index('sample_id').loc[actual.sample_id]
    checked, maximum = 0, 0.
    top_sets = {}

    def close(observed, expected):
        nonlocal checked, maximum
        x, y = np.asarray(observed, float), np.asarray(expected, float)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Audit shape/nonfinite mismatch')
        difference = float(np.max(np.abs(x-y))) if x.size else 0.
        maximum = max(maximum, difference)
        checked += x.size
        if difference > 1e-9:
            raise ValueError(f'Independent numeric mismatch: {difference}')

    for seed in spec['seeds']:
        with np.load(out/f'error-map-{seed}.npz', allow_pickle=False) as saved:
            a = {k: saved[k].copy() for k in saved.files}
        rows = pd.read_csv(out/f'error-map-{seed}.csv', dtype={'sample_id': str})
        record = report['seeds'][str(seed)]
        assert rows.sample_id.tolist() == actual.sample_id.tolist() == a['query_ids'].tolist()
        assert len(rows) == 2754 and rows.sample_id.is_unique and (rows.split_seed == seed).all()
        y = actual[['tap_iron', 'tap_time_len']].to_numpy(float)
        close(a['y'], y)
        close(rows.spout_no, actual.spout_no)
        folds = pd.read_csv(root/f'local/runs/round2-v2/comparison-r1/folds-{seed}.csv').set_index('sample_id').loc[actual.sample_id].fold.to_numpy()
        close(a['folds'], folds)
        close(rows.fold, folds)
        if set(folds) != set(range(5)):
            raise ValueError('Incomplete folds')
        # Every exact numerical duplicate group remains in a single outer fold.
        group = pd.util.hash_pandas_object(features, index=False)
        if pd.DataFrame({'group': group.to_numpy(), 'fold': folds}).groupby('group').fold.nunique().max() != 1:
            raise ValueError('Duplicate-group isolation failed')
        ref = a['reference']
        # Reconstruct both reference columns directly from original saved fold outputs.
        for f in range(5):
            held = folds == f
            d = root/spec['component_development']
            with np.load(d/f'reference-s{seed}-f{f}/predictions.npz', allow_pickle=False) as original:
                assert original['query_ids'].tolist() == actual.loc[held, 'sample_id'].tolist()
                iron, b, v, vi = (original[k].copy() for k in ('tap_iron','tap_time_len','v7_time','v12_iron'))
            with np.load(d/f'tap_time_len-EMA-s{seed}-f{f}/predictions.npz', allow_pickle=False) as original:
                e = original['prediction'][:, 0]
            with np.load(d/f'tap_time_len-SAM-s{seed}-f{f}/predictions.npz', allow_pickle=False) as original:
                sam = original['prediction'][:, 0]
            with np.load(d/f'tap_iron-EMA-s{seed}-f{f}/predictions.npz', allow_pickle=False) as original:
                ei = original['prediction'][:, 0]
            close(ref[held], np.column_stack([iron, b+.75*(e-v)]))
            close(a['V32'][held], np.column_stack([iron, b]))
            close(a['EMA_TIME_Q50'][held, 1], b+.5*(e-v))
            close(a['EMA_TIME_Q100'][held, 1], b+e-v)
            close(a['SAM_TIME_Q50'][held, 1], b+.5*(sam-v))
            close(a['EMA_IRON_EMA_TIME'][held, 0], iron+.5*(ei-vi))
        with np.load(root/spec['previous_output']/f'paired-oof-{seed}.npz', allow_pickle=False) as previous:
            close(a['PTARL_TIME_Q20'][:, 1], .8*previous['V32']+.2*previous['PTARL_RAW'])
            close(a['EMA_PTARL_ENDPOINT_HALF'][:, 1], .5*ref[:, 1]+.5*a['PTARL_TIME_Q20'][:, 1])
            close(a['EMA_PLUS_PTARL_INCREMENT_Q20'][:, 1], ref[:, 1]+.2*(previous['PTARL_RAW']-previous['V32']))
        de3 = pd.read_csv(root/spec['de3_oof'])
        selected = de3.loc[(de3.target == 'tap_iron') & (de3.split_seed == seed)].set_index('sample_id').loc[actual.sample_id]
        close(a['DE3_IRON_Q75_RESERVE'][:, 0], selected[['seed_42_prediction','seed_104729_prediction','seed_130363_prediction']].to_numpy().mean(axis=1))
        den = np.array([math.fsum(y[:, k]) for k in range(2)])
        close(record['denominators'], den)
        loss = 50*np.abs(y-ref)/den
        close(record['score'], 100-loss.sum())
        regions = {'ALL': np.ones(len(rows), bool)}
        for s in (1, 2):
            regions[f'spout={s}'] = actual.spout_no.to_numpy() == s
        for feature in spec['conditional_features']:
            x = features[feature].to_numpy(float)
            buckets = np.empty(len(x), int)
            for f in range(5):
                cuts = np.quantile(x[folds != f], [.25, .5, .75])
                close(record['input_thresholds'][feature][str(f)], cuts)
                buckets[folds == f] = np.searchsorted(cuts, x[folds == f], side='right')
            for q in range(4):
                m = buckets == q
                regions[f'{feature}:Q{q+1}'] = m
                for s in (1, 2):
                    regions[f'{feature}:Q{q+1}:spout={s}'] = m & (actual.spout_no.to_numpy() == s)
        residual = y-ref
        close(record['target_residual_correlation']['signed'], np.corrcoef(residual.T)[0, 1])
        close(record['target_residual_correlation']['absolute'], np.corrcoef(np.abs(residual).T)[0, 1])
        for name, mask in regions.items():
            r = record['regions'][name]
            assert r['rows'] == int(mask.sum())
            if mask.any():
                close(r['mean'], residual[mask].mean(axis=0)); close(r['median'], np.median(residual[mask], axis=0))
                close(r['mae'], np.abs(residual[mask]).mean(axis=0))
                close(r['positive_fraction'], (residual[mask] > 0).mean(axis=0))
                close(r['wmape'], np.abs(residual[mask]).sum(axis=0)/y[mask].sum(axis=0))
                close(r['score_loss_global_denominator'], 50*(np.abs(residual[mask]).sum(axis=0)/den).sum())
        for k, label in enumerate(('iron', 'time', 'combined')):
            values = loss[:, k] if k < 2 else loss.sum(axis=1)
            close(rows[f'score_loss_{label}'], values)
            order = np.argsort(-values, kind='stable')
            top = pd.read_csv(out/f'top20-{seed}-{label}.csv')
            assert top.sample_id.tolist() == actual.sample_id.iloc[order[:20]].tolist()
            top_sets[seed, label] = set(top.sample_id)
            for bucket, n in [('top20',20), ('top1pct', math.ceil(.01*len(y))), ('top5pct',math.ceil(.05*len(y)))]:
                close(record['concentration'][label][bucket]['share'], values[order[:n]].sum()/values.sum())
            curve = pd.read_csv(out/f'concentration-{seed}-{label}.csv')
            close(curve.cumulative_share, np.cumsum(values[order])/values.sum())
            if k < 2:
                for column, expected in [('true',y[:, k]), ('prediction',ref[:, k]), ('error',-residual[:, k]),
                                          ('residual',residual[:, k]), ('relative_error',np.abs(residual[:, k])/y[:, k])]:
                    close(rows[f'{column}_{label}'], expected)
                cuts = np.quantile(ref[:, k], [.25,.5,.75]); close(record['prediction_bins'][label]['cuts'], cuts)
                buckets = np.searchsorted(cuts, ref[:, k], side='right')
                for q in range(4):
                    close(record['prediction_bins'][label]['groups'][f'Q{q+1}']['median'], np.median(residual[buckets == q], axis=0))
        for name in spec['candidates']:
            p = a[name]
            if not np.isfinite(p).all() or (p < 0).any():
                raise ValueError('Invalid candidate')
            if name in ('EMA_IRON_EMA_TIME','DE3_IRON_Q75_RESERVE'):
                close(p[:, 1], ref[:, 1])
            else:
                close(p[:, 0], ref[:, 0])
            reduction = np.abs(y-ref)-np.abs(y-p)
            gains = 50*(reduction/den).sum(axis=1)
            for k, label in enumerate(('iron','time')):
                close(rows[f'{name}_prediction_{label}'], p[:, k])
                close(rows[f'{name}_error_reduction_{label}'], reduction[:, k])
            close(rows[f'{name}_score_gain'], gains)
            c = record['candidates'][name]
            close(c['gain_global_denominator'], math.fsum(gains))
            close(c['positive_score_reduction'], np.maximum(gains, 0).sum())
            close(c['negative_score_reduction'], np.minimum(gains, 0).sum())
            for region, mask in regions.items():
                if mask.any():
                    d = c['regions'][region]
                    close(d['gain_global_denominator'], gains[mask].sum())
                    close(d['gain_region_denominator'], 50*(reduction[mask].sum(axis=0)/y[mask].sum(axis=0)).sum())
            for f in range(5):
                m = folds == f
                close(c['fold_gains'][f], 50*(reduction[m].sum(axis=0)/y[m].sum(axis=0)).sum())
            for direction, order in [('positive',np.argsort(-gains, kind='stable')[:20]), ('negative',np.argsort(gains,kind='stable')[:20])]:
                top = pd.read_csv(out/f'candidate-top20-{seed}-{name}-{direction}.csv')
                assert top.sample_id.tolist() == actual.sample_id.iloc[order].tolist()
                close(top.score_gain, gains[order])
    for label in ('iron','time','combined'):
        assert report['top20_overlap_count'][label] == len(top_sets[42,label] & top_sets[3407,label])
    payload = dict(G0='passed', report_sha256=sha(out/'report.json'), manifest_sha256=sha(out/'manifest.json'),
        auditor_sha256=sha(__file__), numerical_cells=checked, maximum_arithmetic_difference=maximum,
        new_fits=0, G1='descriptive_only_not_formal_promotion')
    with (out/'independent-audit.json').open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(payload))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
