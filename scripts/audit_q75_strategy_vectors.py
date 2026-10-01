"""Independent scalar WMAPE audit: no training code or official CSV reads."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def scalar_score(actual, iron, time):
    prediction = (iron, time)
    return 100-50*math.fsum(math.fsum(abs(float(row[j])-float(p)) for row, p in zip(actual, prediction[j]))/
        math.fsum(abs(float(row[j])) for row in actual) for j in (0, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--kind', choices=['combination', 'selector'], required=True)
    args = parser.parse_args()
    out = Path(args.output)
    name = 'combination-report.json' if args.kind == 'combination' else 'report.json'
    report = json.loads((out/name).read_text())
    spec = json.loads((out/'manifest.json').read_text())['spec']
    differences = []
    identities = {}
    for seed in spec['split_seeds']:
        path = out/(f'combination-s{seed}.npz' if args.kind == 'combination' else f'oof-s{seed}.npz')
        with np.load(path, allow_pickle=False) as saved:
            ids = saved['query_ids'].tolist()
            if len(ids) != 2754 or len(set(ids)) != len(ids) or set(saved['folds']) != set(range(5)):
                raise ValueError('Full unique within-seed five-fold coverage required')
            baseline = scalar_score(saved['actual'], saved['iron'], saved['q75'])
            arms = list(spec['combination_probes']) if args.kind == 'combination' else spec['arms']
            scores = {}
            for arm in arms:
                scores[arm] = scalar_score(saved['actual'], saved['iron'], saved[arm])
                expected = (report['records'][arm][str(seed)]['gain'] if args.kind == 'combination'
                    else report['gains_vs_original_q75'][str(seed)][arm])
                differences.append(abs(scores[arm]-baseline-expected))
                if args.kind == 'combination':
                    differences.append(abs(scores[arm]-report['records'][arm][str(seed)]['candidate_score']))
            if args.kind == 'selector':
                differences.append(abs(scores['FUSION_MAE']-scores['COMPONENT_MAE']-
                    report['fusion_minus_matched_control'][str(seed)]))
        identities[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if max(differences) > 1e-10:
        raise ValueError('Independent scalar scoring mismatch')
    receipt = dict(status='passed', kind=args.kind, maximum_score_difference=max(differences),
        vector_sha256=identities, report_sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(),
        new_fits=0, official_data_reads=0)
    with (out/'independent-score.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
