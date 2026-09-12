"""Restore a certified final QRF for new label-free feature handoffs; never fit."""
import argparse
from pathlib import Path
import time
import numpy as np
from worker import payload, restore, sha, write, zero_fit


def inference(model_path, bundle_sha256, input_path, input_sha256, output):
    output = Path(output)
    if output.exists() or Path(str(output)+'.json').exists():
        raise FileExistsError(output)
    if sha(input_path) != input_sha256:
        raise ValueError('feature handoff changed')
    started = time.perf_counter()
    with zero_fit() as counts:
        model, pre, md = restore(Path(model_path), bundle_sha256)
        arrays, info = payload(Path(input_path))
        if set(arrays) != {'ids', 'numeric', 'spout', 'reference_ns'} or info['training']:
            raise ValueError('only label-free evaluation features permitted')
        if (info['cutoff_ns'] != md['input']['cutoff_ns'] or
                info['raw_schema_sha256'] != md['input']['raw_schema_sha256'] or
                set(info['ids']) & set(model.ids) or
                (arrays['reference_ns'] < info['cutoff_ns']).any()):
            raise ValueError('model cutoff/schema/training ID overlap changed')
        x, diagnostic = pre.transform(arrays['numeric'], arrays['spout'], info['ids'], info['numeric_columns'])
        median, mean, neighbors = model.predict(x, model.training_months)
        checks = {}
        sequences = {'reverse': np.arange(len(x))[::-1],
                     'subset': np.arange(0, len(x), max(1, len(x)//7)),
                     'single': np.array([len(x)//2])}
        for name, indices in sequences.items():
            transformed, _ = pre.transform(arrays['numeric'][indices], arrays['spout'][indices],
                                           arrays['ids'][indices].tolist(), info['numeric_columns'])
            if not np.array_equal(transformed, x[indices]):
                raise ValueError('preprocessing varies with order/subset')
            q, m, _ = model.predict(transformed)
            if not np.array_equal(q, median[indices]) or not np.array_equal(m, mean[indices]):
                raise ValueError('QRF varies with order/subset')
            checks[name] = True
        chunks = [model.predict(x[i:i+127])[:2] for i in range(0, len(x), 127)]
        if not np.array_equal(np.concatenate([v[0] for v in chunks]), median) or not np.array_equal(np.concatenate([v[1] for v in chunks]), mean):
            raise ValueError('QRF varies with chunks')
        checks['chunks'] = True
        if not np.isfinite([median, mean]).all() or (np.asarray([median, mean]) < 0).any():
            raise ValueError('finite nonnegative QRF outputs required')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        np.savez(handle, ids=arrays['ids'], median=median, mean=mean)
    write(Path(str(output)+'.json'), dict(status='PASS', input=info,
          model_bundle_sha256=bundle_sha256, output_sha256=sha(output),
          zero_fit=counts, checks=checks, environment=md['environment'],
          sources=md['sources'], protocol=md['protocol'], support=md['support'],
          preprocessor_diagnostic=diagnostic, neighbors=neighbors,
          elapsed_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    inference(args.model, args.bundle_sha256, args.input, args.input_sha256, args.output)
