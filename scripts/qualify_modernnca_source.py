"""Limited author-core qualification on artificial arrays, without model fits.

This deliberately does not instantiate PLR embeddings or the TALENT adapter.
It checks distance/label aggregation, inference geometry and an autograd
direction against independent NumPy arithmetic. No competition data is read.
"""
import argparse
import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MODEL_SHA256 = '02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f'


def load_author(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError('Pinned unmodified author model required')
    def unused_embedding(*args, **kwargs):
        raise AssertionError('This limited qualification cannot instantiate embeddings')
    def imports(name, *args, **kwargs):
        if name == 'TALENT.model.lib.tabr.utils':
            return SimpleNamespace(make_module=unused_embedding)
        return builtins.__import__(name, *args, **kwargs)
    namespace = {'__builtins__': dict(vars(builtins), __import__=imports)}
    exec(compile(data, str(path), 'exec'), namespace)
    return namespace['ModernNCA']


def numpy_embedding(x, state, blocks):
    def linear(v, key):
        return v @ state[key + '.weight'].T + state[key + '.bias']
    def norm(v, key):
        return (v - state[key + '.running_mean']) / np.sqrt(state[key + '.running_var'] + 1e-5) * state[key + '.weight'] + state[key + '.bias']
    value = linear(x, 'encoder')
    for index in range(blocks):
        key = f'post_encoder.{index}.block'
        value = linear(np.maximum(linear(norm(value, key + '.0'), key + '.1'), 0), key + '.4')
    return norm(value, f'post_encoder.{blocks}') if blocks else value


def numpy_predict(query, candidates, labels, state, blocks, temperature, diagonal=False):
    q = numpy_embedding(query, state, blocks)
    c = numpy_embedding(candidates, state, blocks)
    # Explicit unsquared Euclidean distances, independent of torch.cdist.
    distances = np.empty((len(q), len(c)))
    for i in range(len(q)):
        for j in range(len(c)):
            distances[i, j] = np.sqrt(sum((q[i, k] - c[j, k]) ** 2 for k in range(q.shape[1])))
    logits = -distances / temperature
    if diagonal:
        np.fill_diagonal(logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ labels


def qualify(source):
    torch.set_num_threads(1)
    Author = load_author(source)
    rng = np.random.default_rng(59001)
    candidates = rng.normal(size=(37, 5))
    query = rng.normal(size=(7, 5))
    labels = 750 + 30 * rng.normal(size=37)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64)
    witnesses = []
    for blocks in (0, 1):
        torch.manual_seed(59001)
        model = Author(d_in=5, d_num=5, d_out=1, dim=3, dropout=.1,
            d_block=7, n_blocks=blocks, num_embeddings=None, temperature=.7, sample_rate=.5).double().eval()
        state = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
        with torch.no_grad():
            prediction = model(tensor(query), None, tensor(candidates), tensor(labels), False).numpy()
            reverse = model(tensor(query[::-1].copy()), None, tensor(candidates), tensor(labels), False).numpy()[::-1]
            singleton = np.array([model(tensor(row[None]), None, tensor(candidates), tensor(labels), False).item() for row in query])
            ignored_labels = model(tensor(query), torch.full((len(query),), float('nan')), tensor(candidates), tensor(labels), False).numpy()
        independent = numpy_predict(query, candidates, labels, state, blocks, .7)
        differences = dict(independent=float(np.max(np.abs(prediction-independent))),
            reversed=float(np.max(np.abs(prediction-reverse))), singleton=float(np.max(np.abs(prediction-singleton))),
            inference_labels=float(np.max(np.abs(prediction-ignored_labels))))
        if max(differences.values()) > 1e-8:
            raise ValueError('Limited independent inference witness failed')
        if np.any(prediction < labels.min()) or np.any(prediction > labels.max()):
            raise ValueError('Expected convex label aggregation differs')
        witnesses.append(dict(blocks=blocks, differences=differences, label_hull=True))
        if blocks == 0:
            target = tensor(750+30*rng.normal(size=len(query)))
            loss = ((model(tensor(query), None, tensor(candidates), tensor(labels), False)-target)**2).mean()
            loss.backward()
            actual = model.encoder.weight.grad[0, 0].item()
            losses = []
            for direction in (-1, 1):
                changed = {k: v.copy() for k, v in state.items()}
                changed['encoder.weight'][0, 0] += direction * 1e-6
                pred = numpy_predict(query, candidates, labels, changed, 0, .7)
                losses.append(float(np.mean((pred-target.numpy())**2)))
            finite = (losses[1]-losses[0])/2e-6
            if not np.isclose(actual, finite, atol=1e-5, rtol=1e-5):
                raise ValueError('Independent finite-difference gradient differs')
            witnesses[-1]['gradient'] = dict(autograd=actual, independent_finite_difference=finite)
            # The adapter must first exclude current batch indices from the pool.
            # The author core then adds other batch rows and masks each own label.
            batch = candidates[:4]; batch_y = tensor(labels[:4]).requires_grad_()
            torch.manual_seed(59002)
            selected = torch.randperm(len(candidates)-4)[:int((len(candidates)-4)*.5)].numpy()
            torch.manual_seed(59002)
            result = model(tensor(batch), batch_y, tensor(candidates[4:]), tensor(labels[4:]), True)
            expected = numpy_predict(batch, np.vstack([batch,candidates[4:][selected]]),
                np.concatenate([labels[:4],labels[4:][selected]]), state, 0, .7, diagonal=True)
            if np.max(np.abs(result.detach().numpy()-expected)) > 1e-8:
                raise ValueError('Sampled training label aggregation differs')
            jacobian = np.stack([torch.autograd.grad(result[i], batch_y, retain_graph=True)[0].numpy() for i in range(4)])
            if np.any(np.diag(jacobian) != 0) or not np.any(jacobian-np.diag(np.diag(jacobian))):
                raise ValueError('Own-label exclusion or other-neighbor use differs')
            witnesses[-1]['training_own_label_jacobian'] = np.diag(jacobian).tolist()
            witnesses[-1]['sampled_training_difference'] = float(np.max(np.abs(result.detach().numpy()-expected)))
    return dict(status='limited_author_core_passed',model_sha256=MODEL_SHA256,witnesses=witnesses,
        scope='float64_no_embedding_core_only',unqualified=['PLR embeddings','TALENT adapter','training protocol','cold saved model','resource admission','official G1'],
        official_label_reads=0,optimizer_calls=0,new_official_fits=0,packages=0,uploads=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=qualify(args.source)
    with args.output.open('x') as stream:
        json.dump(report,stream,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':main()
