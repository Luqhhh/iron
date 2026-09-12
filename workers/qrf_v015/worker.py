"""Private, hashed file interface for fixed QRF fitting and zero-fit cold inference."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import platform
from pathlib import Path
import resource
import time
from unittest.mock import patch
import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.ensemble import RandomForestRegressor
from preprocessing import Preprocessor
from qrf_model import QRF, PARAMETERS, PROTOCOL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__, threadpoolctl=threadpoolctl.__version__)


def source_identity():
    here = Path(__file__).resolve().parent
    return {name: sha(here/name) for name in ('worker.py','qrf_model.py','preprocessing.py','pyproject.toml','uv.lock')}


def payload(path, expected=None):
    info = json.loads(Path(str(path)+'.json').read_text())
    if sha(path) != info['sha256'] or (expected and info != expected):
        raise ValueError('worker input hash/identity changed')
    with np.load(path, allow_pickle=False) as f:
        arrays = {name: f[name] for name in f.files}
    if arrays['ids'].tolist() != info['ids']:
        raise ValueError('worker metadata IDs misaligned')
    columns = info['numeric_columns']
    from preprocessing import validate
    validate(arrays['numeric'], arrays['spout'], arrays['ids'].tolist(), columns)
    return arrays, info


@contextmanager
def zero_fit():
    counter = {'forest_fit_attempts': 0, 'preprocessor_fit_attempts': 0}
    def reject_forest(*a, **k):
        counter['forest_fit_attempts'] += 1
        raise ValueError('inference/cold audit forbids forest fitting')
    def reject_preprocessor(*a, **k):
        counter['preprocessor_fit_attempts'] += 1
        raise ValueError('inference/cold audit forbids preprocessing fitting')
    with patch.object(QRF,'fit',reject_forest), patch.object(RandomForestRegressor,'fit',reject_forest), patch.object(Preprocessor,'fit',reject_preprocessor):
        yield counter


def fit(root, month):
    manifest = json.loads((root/'manifest.json').read_text())
    p0 = json.loads((root/'p0/complete.json').read_text())
    if p0['status'] != 'PASS' or manifest['worker_environment'] != environment() or manifest['worker_sources'] != source_identity():
        raise ValueError('P0/environment/source identity differs')
    if manifest['registration']['forest_parameters'] != PARAMETERS or month not in range(6,12):
        raise ValueError('unregistered model parameters/cutoff')
    counts = count_fits(root)
    if counts['forest_attempted'] >= 6 or counts['preprocessor_attempted'] >= 6:
        raise ValueError('QRF development fit budget exhausted')
    folder = root/'models'/str(month)
    folder.mkdir(parents=True, exist_ok=False)
    arrays, info = payload(root/'features'/str(month)/'train.npz', p0['training_inputs'][str(month)])
    cutoff = info['cutoff_ns']
    if (arrays['reference_ns'] >= cutoff).any() or (arrays['available_ns'] > cutoff).any() or not np.isfinite(arrays['y']).all() or (arrays['y'] < 0).any():
        raise ValueError('future/unavailable/non-finite training targets')
    if set(arrays) != {'ids','numeric','spout','reference_ns','available_ns','y','training_months'}:
        raise ValueError('unexpected training payload')
    start = time.perf_counter()
    common = dict(input=info, manifest_sha256=sha(root/'manifest.json'), p0_sha256=sha(root/'p0/complete.json'),
                  sources=source_identity(), environment=environment(), parameters=PARAMETERS, protocol=PROTOCOL)
    write(folder/'preprocessor_intent.json', common)
    pre = Preprocessor().fit(arrays['numeric'],arrays['spout'],info['ids'],info['numeric_columns'])
    write(folder/'preprocessor.json', pre.metadata())
    x, diagnostic = pre.transform(arrays['numeric'],arrays['spout'],info['ids'],info['numeric_columns'])
    write(folder/'forest_intent.json', common)
    model = QRF().fit(x, arrays['y'], info['ids'])
    model.training_months = arrays['training_months']
    joblib.dump(model, folder/'forest.joblib', compress=3)
    write(folder/'bundle.json', dict(**common, forest_sha256=sha(folder/'forest.joblib'),
          preprocessor_sha256=sha(folder/'preprocessor.json'), support=[float(model.y.min()),float(model.y.max())],
          training_diagnostic=diagnostic, transformed_schema_sha256=hashlib.sha256(json.dumps(pre.transformed_columns).encode()).hexdigest(),
          fit_seconds=time.perf_counter()-start, model_bytes=(folder/'forest.joblib').stat().st_size,
          peak_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(month=month, **count_fits(root))), flush=True)


def count_fits(root):
    models = root/'models'
    return dict(forest_attempted=len(list(models.glob('*/forest_intent.json'))),
                forest_completed=len(list(models.glob('*/bundle.json'))),
                preprocessor_attempted=len(list(models.glob('*/preprocessor_intent.json'))),
                preprocessor_completed=len(list(models.glob('*/preprocessor.json'))))


def restore(folder, expected_bundle_sha256):
    if sha(folder/'bundle.json') != expected_bundle_sha256:
        raise ValueError('untrusted or changed bundle metadata')
    md = json.loads((folder/'bundle.json').read_text())
    if md['environment'] != environment() or md['sources'] != source_identity() or md['parameters'] != PARAMETERS or md['protocol'] != PROTOCOL:
        raise ValueError('worker environment/source/model protocol changed')
    if sha(folder/'forest.joblib') != md['forest_sha256'] or sha(folder/'preprocessor.json') != md['preprocessor_sha256']:
        raise ValueError('untrusted serialized model hash')
    # Only self-generated private run artifacts with pre-registered hashes reach joblib.
    model = joblib.load(folder/'forest.joblib')
    pre = Preprocessor.restore(json.loads((folder/'preprocessor.json').read_text()))
    if model.ids != pre.training_ids or len(model.forest.estimators_) != 256:
        raise ValueError('saved training IDs/tree budget changed')
    return model, pre, md


def predict(root, month, output, cold=False):
    complete = json.loads((root/'models_complete.json').read_text())
    folder = root/'models'/str(month)
    start = time.perf_counter()
    with zero_fit() as counter:
        model, pre, md = restore(folder,complete[str(month)])
        load_seconds = time.perf_counter()-start
        arrays, info = payload(root/'features'/str(month)/'evaluation.npz')
        if set(arrays) != {'ids','numeric','spout','reference_ns'} or set(info['ids']) & set(model.ids) or (arrays['reference_ns'] < info['cutoff_ns']).any():
            raise ValueError('evaluation labels/current history/future model rejected')
        x, diagnostic = pre.transform(arrays['numeric'],arrays['spout'],info['ids'],info['numeric_columns'])
        start = time.perf_counter()
        median, mean, neighbors = model.predict(x, model.training_months)
        prediction_seconds = time.perf_counter()-start
        if cold:
            saved = np.load(root/'worker_predictions'/f'{month}.npz',allow_pickle=False)
            for name, value in [('median',median),('mean',mean)]:
                if not np.array_equal(saved[name],value): raise ValueError('cold process differs')
            sequences = [np.arange(len(x))[::-1], np.array([0,len(x)//2,len(x)-1]),np.array([len(x)//2])]
            for indices in sequences:
                q, m, _ = model.predict(x[indices])
                if not np.array_equal(q,median[indices]) or not np.array_equal(m,mean[indices]):
                    raise ValueError('reverse/subset/single inference differs')
            chunks = [model.predict(x[i:i+127])[:2] for i in range(0,len(x),127)]
            if not np.array_equal(np.concatenate([v[0] for v in chunks]),median) or not np.array_equal(np.concatenate([v[1] for v in chunks]),mean):
                raise ValueError('chunked inference differs')
        output.parent.mkdir(parents=True,exist_ok=True)
        if not cold:
            if output.exists(): raise ValueError('prediction output already exists')
            np.savez(output,ids=arrays['ids'],median=median,mean=mean)
        write(Path(str(output)+'.json'), dict(month=month, rows=len(x), load_seconds=load_seconds,
             prediction_seconds=prediction_seconds, peak_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             support=md['support'], lower_boundary=int(np.count_nonzero(median==md['support'][0])),
             upper_boundary=int(np.count_nonzero(median==md['support'][1])), preprocessor_diagnostic=diagnostic,
             neighbors=neighbors, input=info, zero_fit=counter, exact_cold_checks=cold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['environment','fit','predict','cold'])
    parser.add_argument('--root',type=Path)
    parser.add_argument('--month',type=int)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.command == 'environment': print(json.dumps(environment()))
    elif args.command == 'fit': fit(args.root.resolve(),args.month)
    else: predict(args.root.resolve(),args.month,args.output.resolve(),args.command=='cold')
