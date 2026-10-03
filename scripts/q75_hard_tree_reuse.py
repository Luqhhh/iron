"""Zero-fit replay of full V39 evidence; fixed blends against the current Q75."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path('/home/lux1/iron-v39-hard-tree-quality')
OLD = OLD_ROOT / 'local/runs/round2-v39/development-r1'
REF = ROOT / 'local/runs/modernnca-q75-readiness-20261001/preparation-r1/reference-r2'
OUT = ROOT / 'local/runs/q75-hard-tree-reuse-20261003/review-r1'
TARGETS = ('tap_iron', 'tap_time_len')
ARMS = ('GLOBAL', 'INSTANCE')
SEEDS = (42, 3407)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def verify(files):
    for p, h in files.items():
        if sha(p) != h:
            raise ValueError('Dependency changed: ' + p)


def aligned_unit(ids, folds, fold, query_ids, values):
    import numpy as np
    ids, folds, query_ids, values = map(np.asarray, (ids, folds, query_ids, values))
    if len(set(ids)) != len(ids) or folds.shape != ids.shape or set(folds) != set(range(5)):
        raise ValueError('Complete five-fold unique identity required')
    mask = folds == fold
    if (not np.array_equal(ids[mask], query_ids) or values.shape != (int(mask.sum()),)
            or not np.isfinite(values).all()):
        raise ValueError('Query identity/coverage/order or predictions differ')
    return mask


def freeze():
    state = read(ROOT / 'EVIDENCE_STATUS.json')
    best = state['round2_current_platform_best']
    if (best['candidate'], best['score'], best['zip_sha256']) != (
            'EMA_TIME_Q75', 96.392, '41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825'):
        raise ValueError('Reference changed')
    old_state = state['round2_v39_hard_tree_quality']
    anchors = {'manifest.json': old_state['development']['manifest_sha256'],
               'summary.json': '64c8077209fa15b8f91bb6d47d54e69831c6e6dca8e370b97de0867511e43fe2',
               'audit.json': 'ad3daf91cced60ea6c6e9db382517b9fdbf718716f11e5a6b62bcd4ed4231823'}
    files = {str(OLD / n): h for n, h in anchors.items()}
    manifest, audit = read(OLD / 'manifest.json'), read(OLD / 'audit.json')
    if (audit['status'] != 'passed' or audit['cold_models'] != 80 or audit['units'] != 60
            or audit['summary_sha256'] != anchors['summary.json']
            or audit['manifest_sha256'] != anchors['manifest.json']):
        raise ValueError('Original complete cold audit required')
    identity = dict(manifest); identity.pop('identity')
    if digest(identity) != manifest['identity']:
        raise ValueError('Original manifest identity differs')
    for seed in SEEDS:
        for fold in range(5):
            keys = [f'reference-{role}-s{seed}-f{fold}' for role in ('outer', 'calibration')]
            keys += [f'{t}-{a}-s{seed}-f{fold}' for t in TARGETS for a in ARMS]
            for key in keys:
                unit = OLD / key; c = read(unit / 'complete.json')
                if c['identity'] != digest(dict(manifest_identity=manifest['identity'], unit=key)):
                    raise ValueError('Original unit identity differs')
                files[str(unit / 'complete.json')] = sha(unit / 'complete.json')
                files.update({str(unit / n): h for n, h in c['hashes'].items()})
    # Model source identity is reused, not fitted or rewritten.
    for name in ('v39_regressor.py', 'v39_run.py', 'v39_audit.py', 'v37_hard_tree.py', 'v36_hard_tree.py'):
        rel = 'src/bf_tap_r2/' + name
        if rel in manifest['source_hashes']:
            files[str(OLD_ROOT / rel)] = manifest['source_hashes'][rel]
    files[str(OLD_ROOT / 'configs/round2_v39/SPEC.yaml')] = manifest['spec_sha256']
    ref = read(REF / 'q75-reference-receipt.json')
    files[str(REF / ref['reference_file'])] = ref['reference_sha256']
    files[str(REF / 'q75-reference-receipt.json')] = sha(REF / 'q75-reference-receipt.json')
    files.update({str(ROOT / p): h for p, h in ref['input_hashes'].items()})
    files.update(ref['model_source_hashes'])
    for p in [Path(__file__), ROOT / 'docs/q75_hard_tree_reuse/PREREGISTRATION.md', ROOT / 'uv.lock']:
        files[str(p)] = sha(p)
    verify(files)
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / 'manifest.json', dict(files=files, old_manifest=manifest['identity'], reference='EMA_TIME_Q75',
          seeds=list(SEEDS), targets=list(TARGETS), arms=list(ARMS), weight=.2,
          all_new_fits=0, packages=0, formal_promoted=False, scope='preserved_vector_identity_and_arithmetic_only'))
    print(json.dumps(dict(status='frozen', files=len(files))))


def metrics(y, reference, candidate, member, folds, spout):
    import numpy as np
    wm = lambda a, b: float(np.abs(a-b).sum()/np.abs(a).sum())
    improvement = np.abs(y-reference)-np.abs(y-candidate)
    return dict(reference_wmape=wm(y, reference), candidate_wmape=wm(y, candidate), member_wmape=wm(y, member),
                gain=50*(wm(y, reference)-wm(y, candidate)),
                residual_correlation=float(np.corrcoef(y-reference, y-member)[0, 1]),
                improvement_absolute_sum=float(improvement[improvement > 0].sum()),
                harm_absolute_sum=float(-improvement[improvement < 0].sum()),
                improved_rows=int((improvement > 0).sum()),
                fold_gains={str(f):50*(wm(y[folds==f],reference[folds==f])-wm(y[folds==f],candidate[folds==f])) for f in range(5)},
                spout_gains={str(s):50*(wm(y[spout==s],reference[spout==s])-wm(y[spout==s],candidate[spout==s])) for s in sorted(set(spout))})


def evaluate():
    import numpy as np
    manifest = read(OUT / 'manifest.json'); verify(manifest['files'])
    ref = read(REF / 'q75-reference-receipt.json')
    old = read(OLD / 'manifest.json'); summary = read(OLD / 'summary.json')
    records = {}; replay_max = 0.
    with np.load(REF / ref['reference_file'], allow_pickle=False) as bank:
        ids, truth, spout = bank['ids'], bank['targets'], bank['spout']
        if len(ids) != 2754:
            raise ValueError('Full official row coverage required')
        for seed in SEEDS:
            folds = bank[f'fold-{seed}']; q75 = bank[f'current-{seed}']
            if digest(folds.tolist()) != old['fold_hashes'][str(seed)]:
                raise ValueError('Same numeric seed has a different fold identity')
            reference = np.full((len(ids), 2), np.nan)
            members = {f'{t}-{a}':np.full(len(ids), np.nan) for t in TARGETS for a in ARMS}
            old_blends = {k:np.full(len(ids), np.nan) for k in members}
            for fold in range(5):
                mask = folds == fold
                with np.load(OLD/f'reference-outer-s{seed}-f{fold}/predictions.npz', allow_pickle=False) as r:
                    for j, t in enumerate(TARGETS):
                        aligned_unit(ids,folds,fold,r['query_ids'],r[t]);reference[mask,j] = r[t]
                for j,t in enumerate(TARGETS):
                    for arm in ARMS:
                        key=f'{t}-{arm}'; unit=OLD/f'{key}-s{seed}-f{fold}'; meta=read(unit/'metadata.json')
                        if meta['refit']['fit_ids_digest'] != digest(ids[~mask].tolist()):
                            raise ValueError('Outer training IDs differ')
                        with np.load(unit/'predictions.npz', allow_pickle=False) as p:
                            aligned_unit(ids,folds,fold,p['query_ids'],p['prediction'])
                            members[key][mask]=p['prediction']
                        w=meta['weight'];old_blends[key][mask]=(1-w)*reference[mask,j]+w*members[key][mask]
            if not np.isfinite(reference).all() or any(not np.isfinite(v).all() for v in members.values()):
                raise ValueError('Incomplete OOF; do not evaluate partial folds')
            vectors=dict(ids=ids,folds=folds,truth=truth,spout=spout,q75=q75,old_reference=reference)
            for j,t in enumerate(TARGETS):
                for arm in ARMS:
                    key=f'{t}-{arm}'; y=truth[:,j];member=members[key]
                    old_result=metrics(y,reference[:,j],old_blends[key],member,folds,spout)
                    expected=next(r for r in summary['records'] if r['target']==t and r['recipe']==arm)['seed_results'][str(seed)]
                    replay_max=max(replay_max,abs(old_result['gain']-expected['gain']),abs(old_result['member_wmape']-expected['standalone_wmape']))
                    candidate=.8*q75[:,j]+.2*member
                    row=metrics(y,q75[:,j],candidate,member,folds,spout)
                    row.update(old_replayed_gain=old_result['gain'],old_member_wmape=old_result['member_wmape'],
                               max_reference_difference=float(np.max(np.abs(reference[:,j]-q75[:,j]))))
                    records.setdefault(key,{})[str(seed)]=row
                    vectors[key+'-member']=member;vectors[key+'-candidate']=candidate;vectors[key+'-old_blend']=old_blends[key]
            with (OUT/f'seed-{seed}.npz').open('xb') as f:np.savez_compressed(f,**vectors)
    if replay_max > 1e-10:
        raise ValueError('Original published result did not replay')
    write(OUT/'report.json',dict(records=records,old_replay_maximum_difference=replay_max,
          two_seed_positive={k:all(r['gain']>0 for r in v.values()) for k,v in records.items()},
          mean_gain={k:float(np.mean([r['gain'] for r in v.values()])) for k,v in records.items()},
          new_fits=0,new_confirmation_seeds=0,packages=0,formal_promoted=False,platform_gain=None,
          array_hashes={str(s):sha(OUT/f'seed-{s}.npz') for s in SEEDS}))
    print(json.dumps({k:{s:v['gain'] for s,v in r.items()} for k,r in records.items()},indent=2))


def audit():
    import math
    import numpy as np
    manifest=read(OUT/'manifest.json'); verify(manifest['files']); report=read(OUT/'report.json'); maximum=0.;count=0
    for seed in SEEDS:
        if sha(OUT/f'seed-{seed}.npz') != report['array_hashes'][str(seed)]:
            raise ValueError('Evaluated OOF vector changed')
        with np.load(OUT/f'seed-{seed}.npz',allow_pickle=False) as v:
            for j,t in enumerate(TARGETS):
                y=v['truth'][:,j];base=v['q75'][:,j];old=v['old_reference'][:,j]
                denominator=math.fsum(abs(float(x)) for x in y)
                wm=lambda p:math.fsum(abs(float(a)-float(b)) for a,b in zip(y,p))/denominator
                for arm in ARMS:
                    key=f'{t}-{arm}';row=report['records'][key][str(seed)];member=v[key+'-member'];candidate=v[key+'-candidate']
                    formula=[.8*float(a)+.2*float(b) for a,b in zip(base,member)]
                    assert max(abs(a-b) for a,b in zip(formula,candidate))<1e-10
                    scalar=dict(reference_wmape=wm(base),candidate_wmape=wm(candidate),member_wmape=wm(member),
                                gain=50*(wm(base)-wm(candidate)),old_replayed_gain=50*(wm(old)-wm(v[key+'-old_blend'])))
                    for k,x in scalar.items():maximum=max(maximum,abs(x-row[k]));count+=1
    if maximum>1e-10:raise ValueError('Independent scalar replay differs')
    write(OUT/'independent-audit.json',dict(status='passed',fields_checked=count,maximum_difference=maximum,
          report_sha256=sha(OUT/'report.json'),manifest_sha256=sha(OUT/'manifest.json'),new_fits=0,
          cold_scope='original audited model hashes retained; no new cold model execution'))
    print(json.dumps(dict(status='passed',fields=count,maximum_difference=maximum)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','evaluate','audit']);args=p.parse_args()
    if sys.version_info[:2]!=(3,12) or any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise ValueError('Locked Python3.12 and pre-import thread pinning required')
    {'freeze':freeze,'evaluate':evaluate,'audit':audit}[args.stage]()
