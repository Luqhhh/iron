"""Independent scalar scoring and within-split averaging; reads private arrays only."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def scalar_score(actual, iron, time):
    return 100-50*math.fsum(math.fsum(abs(float(row[j])-float(p)) for row, p in zip(actual, prediction))/
        math.fsum(abs(float(row[j])) for row in actual) for j, prediction in enumerate((iron, time)))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    out = Path(parser.parse_args().output)
    manifest = json.loads((out/'manifest.json').read_text()); spec = manifest['spec']
    report = json.loads((out/'report.json').read_text()); differences = []; hashes = {}
    for seed in spec['split_seeds']:
        path = out/f'oof-s{seed}.npz'
        with np.load(path, allow_pickle=False) as saved:
            if len(saved['query_ids']) != 2754 or len(set(saved['query_ids'])) != 2754 or set(saved['folds']) != set(range(5)):
                raise ValueError('Complete unique same-split OOF required')
            baseline = scalar_score(saved['actual'], saved['iron'], saved['q75'])
            record = report['records'][str(seed)]
            scores = {}
            for arm in record['gains_vs_q75']:
                value = saved[arm]
                if not np.isfinite(value).all() or (value < 0).any(): raise ValueError('Invalid affine prediction')
                if spec['phase'] == 'initialization':
                    if arm.endswith('MEAN3'):
                        prefix = arm.split('_')[0]
                        component = np.mean([saved[f'{prefix}_{i}_component'] for i in (42, 1042, 2042)], axis=0)
                    else: component = saved[arm+'_component']
                else: component = saved['EMA_UPDATES_component']
                np.testing.assert_array_equal(value, saved['q75']+.75*(component-saved['old_ema']))
                scores[arm] = scalar_score(saved['actual'], saved['iron'], value)
                differences.append(abs(scores[arm]-baseline-record['gains_vs_q75'][arm]))
            if spec['phase'] == 'initialization':
                np.testing.assert_array_equal(saved['EMA_42'], saved['q75'])
                for init in (42, 1042, 2042):
                    differences.append(abs(scores[f'EMA_{init}']-scores[f'BASE_{init}']-record['ema_minus_paired_base'][str(init)]['fusion_gain']))
                    differences.append(abs(scores['EMA_MEAN3']-scores[f'EMA_{init}']-record['mean_ema_minus_each_single'][str(init)]))
                    y = saved['actual'][:, 1]
                    component_gain = 50*(math.fsum(abs(float(t)-float(p)) for t, p in zip(y, saved[f'BASE_{init}_component']))-
                        math.fsum(abs(float(t)-float(p)) for t, p in zip(y, saved[f'EMA_{init}_component'])))/math.fsum(abs(float(t)) for t in y)
                    differences.append(abs(component_gain-record['ema_minus_paired_base'][str(init)]['component_gain']))
                differences.append(abs(scores['EMA_MEAN3']-scores['BASE_MEAN3']-record['equal_mean_ema_minus_equal_mean_base']))
        hashes[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if max(differences) > 1e-10 or hashes != report['vector_sha256']:
        raise ValueError('Independent scalar scores/vector identities mismatch')
    receipt = dict(status='passed', maximum_score_difference=max(differences), vector_sha256=hashes,
        report_sha256=hashlib.sha256((out/'report.json').read_bytes()).hexdigest(), new_fits=0, official_data_reads=0)
    with (out/'independent-score.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
