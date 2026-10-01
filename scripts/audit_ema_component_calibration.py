"""Independent saved-column and scalar audit; no CSV reads or fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_score(y, iron, time):
    return 100-50*math.fsum(math.fsum(abs(float(row[j])-float(p)) for row, p in zip(y, prediction))/
        math.fsum(abs(float(row[j])) for row in y) for j, prediction in enumerate((iron, time)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    out = Path(p.parse_args().output); manifest = json.loads((out/'manifest.json').read_text())
    report = json.loads((out/'report.json').read_text()); differences = []; states = 0; fits = 0
    for seed in manifest['spec']['split_seeds']:
        path = out/f'oof-s{seed}.npz'
        if sha(path) != report['vector_sha256'][str(seed)]: raise ValueError('OOF identity differs')
        with np.load(path, allow_pickle=False) as saved:
            if len(saved['query_ids']) != 2754 or len(set(saved['query_ids'])) != 2754 or set(saved['folds']) != set(range(5)):
                raise ValueError('Complete unique same-split OOF required')
            baseline = scalar_score(saved['actual'], saved['iron'], saved['q75'])
            control = scalar_score(saved['actual'], saved['iron'], saved['UNCORRECTED'])
            record = report['records'][str(seed)]
            differences.append(abs(control-baseline-record['control_gain']))
            for family in ('GLOBAL', 'PRESSURE'):
                value = scalar_score(saved['actual'], saved['iron'], saved[family])
                differences.extend((abs(value-baseline-record['gains'][family]), abs(value-control-record['matched_gains'][family])))
            for fold in range(5):
                unit = out/f's{seed}-f{fold}'; receipt = json.loads((unit/'complete.json').read_text())
                if receipt['new_model_fits'] or receipt['optimizer_runs']: raise ValueError('Extra model fit')
                fits += receipt['correction_fit_calls']
                for name, expected in receipt['witnesses'].items():
                    cold = json.loads((unit/name/'cold-audit.json').read_text())
                    rss = json.loads((unit/name/'cold-rss.json').read_text())
                    if (sha(unit/name/'complete.json') != expected or cold['receipt_sha256'] != expected
                            or cold['status'] != 'passed' or rss['status'] != 'passed' or rss['receipt_sha256'] != expected
                            or rss['peak_rss_mib'] > manifest['spec']['max_worker_rss_mib']):
                        raise ValueError('Actual independent cold coverage differs')
                    witness = json.loads((unit/name/'complete.json').read_text())
                    for filename, digest in witness['hashes'].items():
                        if sha(unit/name/filename) != digest: raise ValueError('Cold witness artifact changed')
                    states += 1
                if sha(unit/'predictions.npz') != receipt['predictions_sha256']: raise ValueError('Unit predictions changed')
                with np.load(unit/'predictions.npz', allow_pickle=False) as values:
                    held = saved['folds'] == fold
                    np.testing.assert_array_equal(values['query_ids'], saved['query_ids'][held])
                    np.testing.assert_array_equal(values['UNCORRECTED'], values['q75']+.75*(values['ema_F']-values['old_ema']))
                    for family in ('GLOBAL', 'PRESSURE'):
                        head = receipt['fitted'][family]
                        bins = np.searchsorted(head['cuts'], values['pressure'], side='right')
                        corrected = values['ema_F']+head['gamma']*np.asarray(head['offsets'])[bins]
                        np.testing.assert_array_equal(values[family], values['q75']+.75*(corrected-values['old_ema']))
                        np.testing.assert_array_equal(saved[family][held], values[family])
                        if not np.isfinite(values[family]).all() or (values[family] < 0).any(): raise ValueError('Invalid column')
    if max(differences) > 1e-10 or states != 20 or fits != 80: raise ValueError('Scalar/state/head-fit budget differs')
    receipt = dict(status='passed', maximum_scalar_difference=max(differences), cold_witnesses=states,
        correction_fit_calls=fits, new_model_fits=0, optimizer_runs=0, official_CSV_reads=0,
        report_sha256=sha(out/'report.json'), manifest_sha256=sha(out/'manifest.json'))
    with (out/'independent-score.json').open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
