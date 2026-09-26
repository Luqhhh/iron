"""Read-only complete V13 audit; recompute preprocessing and score arithmetic."""
from pathlib import Path
import hashlib,importlib.metadata,json
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t
import torch
from rtdl_num_embeddings import compute_bins
from bf_tap_r2.data import FEATURES,TARGETS
from bf_tap_r2.candidate_tiers import classify_candidates
from bf_tap_r2.v5_library import load_v5_training_frame,fold_vector
from bf_tap_r2.v5_spec import load_v5_spec
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v7_periodic import digest,file_hash,write_new
from bf_tap_r2.v11_quantile import load_references
root=Path.cwd();out=root/'local/runs/round2-v13-ple-repair/development-r1'
spec=yaml.safe_load((root/'configs/round2_v13/SPEC.yaml').read_text());m=json.loads((out/'manifest.json').read_text());summary=json.loads((out/'summary.json').read_text())
assert summary['status']=='development_complete'
expected_pool={(target,recipe) for target in spec['targets'] for recipe in spec['recipes']}
assert len(summary['records'])==len(expected_pool)
assert {(r['target'],r['recipe']) for r in summary['records']}==expected_pool
assert file_hash(root/'configs/round2_v13/SPEC.yaml')==m['spec_sha256']
assert {p:importlib.metadata.version(p) for p in spec['runtime_versions']}==m['versions']
for name,sha in m['source_hashes'].items():assert file_hash(root/'src/bf_tap_r2'/name)==sha,name
for relative,sha in m['reference_hashes'].items():assert file_hash(root/relative)==sha,relative
frame=load_v5_training_frame(root)
assert hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()==m['data_digest']
folds={s:fold_vector(root,frame,s,load_v5_spec(root)) for s in spec['split_seeds']}
for s,f in folds.items():assert digest(f.tolist())==m['fold_digests'][str(s)]
base,q20,current,controls,hashes=load_references(root,frame,folds,spec);assert hashes==m['reference_hashes']
control=json.loads((out/'control.json').read_text());assert control['max_absolute_difference']==0
assert file_hash(out/'control.npy')==control['sha256']
np.testing.assert_array_equal(np.load(out/'control.npy'),controls['tap_time_len']['tabm_plr001'][42][folds[42]==0])
e=[json.loads(line) for line in (out/'fit_ledger.jsonl').read_text().splitlines()]
assert len(e)==40 and all(x['event']=='complete' for x in e)
assert len({x['key'] for x in e})==40
bykey={x['key']:x for x in e};checked={};records=[];budgets={};prep_cache={}
def direct_error(actual,p):return float(np.abs(actual-p).sum()/np.abs(actual).sum())
def metric(y,p,fv):
 spout=frame.spout_no.values
 return {'wmape':direct_error(y,p),'by_fold':{str(f):direct_error(y[fv==f],p[fv==f]) for f in range(5)},'by_spout':{str(v):direct_error(y[spout==v],p[spout==v]) for v in np.unique(spout)}}
computed_metrics={t:{spec['reference_by_target'][t]:{str(seed):metric(frame[t].values,current[seed][t],folds[seed]) for seed in folds}} for t in spec['targets']}
for target in spec['targets']:
 for recipe,config in spec['recipes'].items():
  pred={s:np.full(len(frame),np.nan) for s in folds};budgets[target+'-'+recipe]=[]
  for seed,foldvec in folds.items():
   for fold in range(5):
    mask=foldvec==fold;key=f'{target}-{recipe}-s{seed}-f{fold}';path=out/(key+'.npy');event=bykey[key]
    sha=file_hash(path);assert sha==event['prediction_sha256'];p=np.load(path)
    assert p.shape==(int(mask.sum()),) and np.isfinite(p).all();pred[seed][mask]=p;checked[key]=sha
    train=frame.loc[~mask].reset_index(drop=True);inner=group_safe_inner_folds(train,seed=42)['fold']!=0;meta=event['metadata']
    assert meta['fit_rows']==len(train) and meta['inner_fit_rows']==int(inner.sum()) and meta['optimizer_runs']==2
    assert meta['fit_ids_digest']==digest(train.sample_id.tolist()) and meta['inner_ids_digest']==digest(train.loc[inner,'sample_id'].tolist())
    np.testing.assert_allclose(meta['inner_feature_means'],train.loc[inner,list(FEATURES)].mean(),atol=1e-12)
    np.testing.assert_allclose(meta['outer_feature_means'],train[list(FEATURES)].mean(),atol=1e-12)
    assert meta['loss']=='mse' and meta['beta'] is None
    assert 1<=meta['selected_epoch']<=meta['selection_stopped_epoch']<=240
    assert meta['budget_limited']==(meta['selection_stopped_epoch']>=240)
    assert len(meta['bin_trace'])==2
    assert meta['activation']==config['activation']
    if config['activation']:
     assert meta['nonlinear_weights_nonzero']==0 and meta['nonlinear_weight_norm']==0
    else:
     assert meta['nonlinear_weights_nonzero']>0 and meta['nonlinear_weight_norm']>0
    for stage,(trace,part) in enumerate(zip(meta['bin_trace'],[train.loc[inner],train])):
     cache_key=(seed,fold,stage)
     if cache_key not in prep_cache:
      x=part[list(FEATURES)].to_numpy(dtype=np.float64)
      numeric=((x-x.mean(0))/x.std(0)).astype(np.float32)
      bins=compute_bins(torch.as_tensor(numeric),n_bins=spec['ple_policy']['n_bins'])
      prep_cache[cache_key]={'standardized_input_sha256':hashlib.sha256(numeric.tobytes()).hexdigest(),'bins':[b.tolist() for b in bins]}
     prep=prep_cache[cache_key]
     assert trace['rows']==len(part) and trace['ids_digest']==digest(part.sample_id.tolist())
     assert trace['standardized_input_sha256']==prep['standardized_input_sha256'] and trace['bins']==prep['bins']
    budgets[target+'-'+recipe].append({k:meta[k] for k in ['selected_epoch','selection_stopped_epoch','budget_limited']})
  assert all(np.isfinite(x).all() for x in pred.values())
  row=next(r for r in summary['records'] if r['target']==target and r['recipe']==recipe);y=frame[target].values
  error=lambda actual,p:float(np.abs(actual-p).sum()/np.abs(actual).sum())
  checks={}
  for label,ref in [('A35',base),('Q20',q20),('CURRENT',current)]:
   stored=row['comparisons'][label];gains={};alphas={};count=0
   for held in folds:
    scores=[np.mean([error(y,(1-a)*ref[s][target]+a*pred[s]) for s in folds if s!=held]) for a in spec['blend_grid']]
    alpha=spec['blend_grid'][int(np.argmin(scores))];assert alpha==stored['alphas'][str(held)];alphas[held]=alpha
    blend=(1-alpha)*ref[held][target]+alpha*pred[held];g=[]
    for f in range(5):
     mask=folds[held]==f;value=50*(error(y[mask],ref[held][target][mask])-error(y[mask],blend[mask]));cell=next(c for c in stored['cells'] if c['seed']==held and c['fold']==f)
     assert abs(value-cell['delta_score'])<1e-12;g.append(value);count+=int(value>0)
    gains[held]=float(np.mean(g));assert abs(gains[held]-stored['seed_gains'][str(held)])<1e-12
   mean=float(np.mean(list(gains.values())));lcb=mean-float(t.ppf(.95,1))*float(np.std(list(gains.values()),ddof=1))/np.sqrt(2)
   assert abs(mean-stored['seed_summary']['mean'])<1e-12 and abs(lcb-stored['seed_summary']['lcb95'])<1e-12
   checks[label]={'seed_gains':gains,'mean':mean,'lcb95':lcb,'alphas':alphas,'positive_folds':count}
  eligible=all(g>0 for label in ['A35','CURRENT'] for g in checks[label]['seed_gains'].values())
  other=next(x for x in TARGETS if x!=target);packages=[];computed_metrics[target][recipe]={}
  for seed in folds:
   alpha=checks['CURRENT']['alphas'][seed];blend=(1-alpha)*current[seed][target]+alpha*pred[seed]
   computed_metrics[target][recipe][str(seed)]=metric(y,blend,folds[seed])
   score=100-50*(error(y,blend)+error(frame[other].values,base[seed][other]))
   assert abs(score-row['development_package_scores'][str(seed)])<1e-12;packages.append(score)
   assert abs(error(y,pred[seed])-row['single_wmape'][str(seed)])<1e-12
   control_name='tabm_plr001'
   assert abs(error(y,controls[target][control_name][seed])-row['standard_control_wmape'][str(seed)])<1e-12
  records.append({'target':target,'recipe':recipe,'checks':checks,'confirmation_eligible':bool(eligible),'development_package_score':float(np.mean(packages))})
eligible=[r for r in records if r['confirmation_eligible']]
if eligible:
 best=min(eligible,key=lambda r:(-r['checks']['CURRENT']['mean'],spec['tie_preference_by_target'][r['target']].index(r['recipe']),r['target']));selected={'target':best['target'],'recipe':best['recipe']}
else:selected=None
assert selected==summary['selected_confirmation']
tiers=classify_candidates(computed_metrics,spec,yaml.safe_load((root/'configs/candidate_tiers.yaml').read_text()))
for target in spec['targets']:
 for recipe in spec['recipes']:
  a=tiers['decisions'][target][recipe];b=summary['candidate_tiers']['decisions'][target][recipe]
  for key in ['tier','failed_conditions','eligible','improved_folds']:assert a[key]==b[key]
for key in ['formal_selected','exploration_selected','submission_priority']:
 assert [(r['target'],r['candidate']) for r in tiers[key]]==[(r['target'],r['candidate']) for r in summary['candidate_tiers'][key]]
result={'status':'passed','prediction_count':len(checked),'hashes':checked,'training_budget':budgets,'records':records,'selected_confirmation':selected,'model_fits':0,'preprocessing_reconstructions':len(prep_cache),'candidate_tiers_recomputed':True,'audit_code_sha256':file_hash(__file__)}
write_new(out/'audit-r1.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['hashes','training_budget']},indent=2))
