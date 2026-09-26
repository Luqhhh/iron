"""Zero-fit independent arithmetic and identity audit for V12 development."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
import yaml

from bf_tap_r2.candidate_tiers import classify_candidates
from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v5_library import load_v5_training_frame, fold_vector
from bf_tap_r2.v5_spec import load_v5_spec
from bf_tap_r2.v7_periodic import digest, file_hash, write_new
from bf_tap_r2.v11_quantile import load_references


def err(y, p):
    return float(np.abs(y-p).sum()/np.abs(y).sum())


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def details(y, p, fv, spout):
    return {'wmape': err(y,p),
            'by_fold': {str(f): err(y[fv==f],p[fv==f]) for f in range(5)},
            'by_spout': {str(s):err(y[spout==s],p[spout==s]) for s in np.unique(spout)}}


def audit(root, out, destination):
    spec_path=root/'configs/round2_v12/SPEC.yaml'
    spec=yaml.safe_load(spec_path.read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert file_hash(spec_path)==manifest['spec_sha256']
    assert {p:importlib.metadata.version(p) for p in spec['runtime_versions']}==manifest['versions']==spec['runtime_versions']
    for name,sha in manifest['source_hashes'].items():
        assert file_hash(root/'src/bf_tap_r2'/name)==sha
    for path,sha in manifest['reference_hashes'].items():
        assert file_hash(root/path)==sha
    frame=load_v5_training_frame(root)
    assert hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()==manifest['data_digest']
    folds={s:fold_vector(root,frame,s,load_v5_spec(root)) for s in spec['split_seeds']}
    for s,fv in folds.items():
        assert digest(fv.tolist())==manifest['fold_digests'][str(s)]
    base,q20,current,controls,hashes=load_references(root,frame,folds,spec)
    assert hashes==manifest['reference_hashes']
    control=json.loads((out/'control.json').read_text()); c=spec['control']
    assert control['sha256']==file_hash(out/'control.npy')
    np.testing.assert_array_equal(np.load(out/'control.npy'), controls[c['target']][c['recipe']][c['split_seed']][folds[c['split_seed']]==c['fold']])
    assert control['max_absolute_difference']==0
    events=[json.loads(line) for line in (out/'fit_ledger.jsonl').read_text().splitlines()]
    expected={f'joint-{name}-s{s}-f{f}' for name in spec['recipes'] for s in folds for f in range(5)}
    assert len(events)==len(expected)==spec['budget']['development_outer_fits']
    assert {e['key'] for e in events}==expected and all(e['event']=='complete' for e in events)
    bykey={e['key']:e for e in events}; predictions={}; checked={}
    for name in spec['recipes']:
        predictions[name]={s:np.full((len(frame),2),np.nan) for s in folds}
        for s,fv in folds.items():
            for f in range(5):
                key=f'joint-{name}-s{s}-f{f}'; event=bykey[key]; meta=event['metadata']; path=out/(key+'.npy')
                assert file_hash(path)==event['prediction_sha256']; checked[key]=file_hash(path)
                p=np.load(path,allow_pickle=False); mask=fv==f
                assert p.shape==(int(mask.sum()),2) and np.isfinite(p).all()
                predictions[name][s][mask]=p
                outer=frame.loc[~mask].reset_index(drop=True)
                inner=outer.loc[group_safe_inner_folds(outer,seed=spec['training']['inner_seed'])['fold']!=0]
                assert meta['fit_ids_digest']==digest(outer.sample_id.tolist())
                assert meta['inner_ids_digest']==digest(inner.sample_id.tolist())
                assert meta['fit_rows']==len(outer) and meta['inner_fit_rows']==len(inner)
                assert meta['n_outputs']==2 and meta['optimizer_runs']==2
                assert 1<=meta['selected_epoch']<=meta['selection_stopped_epoch']<=spec['training']['max_epochs']
                assert meta['budget_limited']==(meta['selection_stopped_epoch']>=spec['training']['max_epochs'])
                for label,part in [('inner',inner),('outer',outer)]:
                    check(meta[label+'_feature_means'],part[list(FEATURES)].mean().values)
                    check(meta[label+'_target_mean'],part[list(TARGETS)].mean().values)
                    check(meta[label+'_target_std'],part[list(TARGETS)].std(ddof=0).values)
    assert len(summary['records'])==len(spec['recipes'])*2
    assert {(r['target'],r['recipe']) for r in summary['records']}=={(target,name) for target in TARGETS for name in spec['recipes']}
    metrics={target:{spec['reference_by_target'][target]:{str(s):details(frame[target].values,current[s][target],fv,frame.spout_no.values) for s,fv in folds.items()}} for target in TARGETS}
    eligible=[]
    for row in summary['records']:
        target,name=row['target'],row['recipe']; y=frame[target].values
        p={s:predictions[name][s][:,list(TARGETS).index(target)] for s in folds}
        metrics[target][name]={}; ok=True
        for label,ref in [('A35',base),('Q20',q20),('CURRENT',current)]:
            gains=[]; comp=row['comparisons'][label]; allcells=[]
            assert len(comp['cells'])==10 and {(c['seed'],c['fold']) for c in comp['cells']}=={(s,f) for s in folds for f in range(5)}
            for held,fv in folds.items():
                choices=[np.mean([err(y,(1-a)*ref[s][target]+a*p[s]) for s in folds if s!=held]) for a in spec['blend_grid']]
                a=spec['blend_grid'][int(np.argmin(choices))]; assert comp['alphas'][str(held)]==a
                blend=(1-a)*ref[held][target]+a*p[held]
                cells=[50*(err(y[fv==f],ref[held][target][fv==f])-err(y[fv==f],blend[fv==f])) for f in range(5)]
                recorded=sorted((c for c in comp['cells'] if c['seed']==held),key=lambda c:c['fold'])
                check([c['delta_score'] for c in recorded],cells)
                g=float(np.mean(cells)); check(comp['seed_gains'][str(held)],g); gains.append(g); allcells.extend(cells)
                if label in ('A35','CURRENT') and g<=0: ok=False
                if label=='CURRENT':
                    metrics[target][name][str(held)]=details(y,blend,fv,frame.spout_no.values)
                    other=next(x for x in TARGETS if x!=target)
                    check(row['development_package_scores'][str(held)],100-50*(err(y,blend)+err(frame[other].values,base[held][other])))
                check(row['single_wmape'][str(held)],err(y,p[held]))
            check(comp['seed_summary']['mean'],np.mean(gains))
            check(comp['seed_summary']['lcb95'],np.mean(gains)-t.ppf(.95,len(gains)-1)*np.std(gains,ddof=1)/np.sqrt(len(gains)))
            assert comp['fold_summary']['positive']==sum(g>0 for g in allcells)
        if ok: eligible.append(row)
    selected=None
    if eligible:
        best=min(eligible,key=lambda r:(-r['comparisons']['CURRENT']['seed_summary']['mean'],spec['tie_preference_by_target'][r['target']].index(r['recipe']),r['target']))
        selected={k:best[k] for k in ('target','recipe')}
    assert summary['selected_confirmation']==selected
    tiers=classify_candidates(metrics,spec,yaml.safe_load((root/'configs/candidate_tiers.yaml').read_text()))
    assert tiers==summary['candidate_tiers']
    assert summary['packages']==summary['uploads']==0 and not summary['release_authorized']
    result={'status':'passed','prediction_count':len(checked),'hashes':checked,'selected_confirmation':selected,'model_fits':0,
            'manifest_sha256':file_hash(out/'manifest.json'),'summary_sha256':file_hash(out/'summary.json'),
            'audit_script_sha256':file_hash(__file__),'epoch_cap_hits':sum(e['metadata']['budget_limited'] for e in events)}
    write_new(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=Path.cwd()
    assert args.run.resolve().is_relative_to(root/'local/runs/round2-v12-joint-tabm')
    assert args.output.resolve().is_relative_to(root/'local')
    audit(root,args.run,args.output)
