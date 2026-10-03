"""Audit the one existing synthetic fit after the target-layout audit correction."""
import ast
import json
import os
from pathlib import Path

import numpy as np
import torch

from bf_tap_r2 import de3_independent_batches as run
from bf_tap_r2.ema_independent_batches_model import member_orders


def main():
    run.setup()
    base = run.MAIN/'local/runs/de3-independent-batches-20261004'
    d = base/'engineering-fit-r1'
    prior = base/'engineering-cold-failure-r1'
    warm = run.read(d/'warm.json')
    run.verify(warm['files'])
    if (warm['pid'] == os.getpid() or warm['scope'] != 'synthetic_only'
            or warm['source_sha256'] != run.sha(prior/'de3_independent_batches.py')
            or warm['model_sha256'] != run.sha(run.MODEL)
            or run.sha(prior/'test_de3_independent_batches.py') != run.sha(run.ROOT/'tests/test_de3_independent_batches.py')):
        raise ValueError('Original fit/model/test identity differs')
    trees = [ast.parse(p.read_text()) for p in (prior/'de3_independent_batches.py', Path(run.__file__))]
    for tree in trees:
        matched = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'audit_models']
        if len(matched) != 1:
            raise ValueError('Exact one audit-only function correction required')
        matched[0].body = [ast.Pass()]
    if ast.dump(trees[0]) != ast.dump(trees[1]):
        raise ValueError('Code outside audit_models changed since the existing fit')
    import pickle
    with (d/'training.pkl').open('rb') as f:
        training = pickle.load(f)
    with (d/'query.pkl').open('rb') as f:
        query = pickle.load(f)
    if len(training) != 2204 or not all(k.startswith('synthetic-joint-') for k in training.sample_id):
        raise ValueError('Frozen synthetic rows required')
    traces = {}
    for role in ('selection','refit'):
        trace = torch.load(d/(role+'.pt'), map_location='cpu', weights_only=True)['trace']
        traces[role] = trace
        rng = np.random.default_rng(warm['settings']['random_seed'])
        expected = [run.digest(member_orders(rng,trace['fit_rows'],16).tolist()) for _ in range(trace['stopped_epoch'])]
        if trace['member_order_digests'] != expected or trace['stopped_epoch'] > 2:
            raise ValueError('Synthetic coverage or epoch budget differs')
    run.verify_counts(warm['counts'], traces)
    with np.load(d/'predictions.npz', allow_pickle=False) as p:
        np.testing.assert_array_equal(p['ids'], query.sample_id.to_numpy(str))
        result = run.audit_models(d, training, query, p['prediction'], warm['settings'], {})
    run.write(d/'cold-r2.json', dict(**result, pid=os.getpid(), warm_sha256=run.sha(d/'warm.json'),
        source_sha256=run.sha(run.__file__), original_source_sha256=warm['source_sha256'],
        audit_only_ast_change_verified=True, model_sha256=run.sha(run.MODEL),
        auditor_sha256=run.sha(__file__), new_fits=0, new_optimizer_constructors=0,
        original_synthetic_optimizer_constructors=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
