"""Independent arithmetic and identity audit of V15 confirmation, no fits."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import numpy as np
import pandas as pd
import yaml
from scipy.stats import t
from bf_tap_r2.data import FEATURES,TARGETS
from bf_tap_r2.v5_library import load_v5_training_frame,fold_vector
from bf_tap_r2.v5_spec import load_v5_spec
from bf_tap_r2.v3_4_bags import group_safe_inner_folds
from bf_tap_r2.v7_periodic import digest,file_hash,write_new
from bf_tap_r2.v15_confirm import four_seed_references
root=Path.cwd();out=root/'local/runs/round2-v15-task-experts/confirmation-r1'
spec=yaml.safe_load((root/'configs/round2_v15/SPEC.yaml').read_text());conf=yaml.safe_load((root/'configs/round2_v15/CONFIRMATION.yaml').read_text())
dev=root/conf['development_run'];manifest=json.loads((out/'manifest.json').read_text());summary=json.loads((out/'summary.json').read_text());dev_audit=json.loads((dev/'audit-r1.json').read_text())
assert summary['selected']==conf['selected']=={'target':'tap_time_len','recipe':'task_gated_experts'}
assert file_hash(root/'configs/round2_v15/SPEC.yaml')==manifest['spec_sha256']
assert file_hash(root/'configs/round2_v15/CONFIRMATION.yaml')==manifest['confirmation_sha256']
assert file_hash(dev/'summary.json')==manifest['development_summary_sha256'] and file_hash(dev/'audit-r1.json')==manifest['development_audit_sha256']
for name,sha in manifest['source_hashes'].items():assert file_hash(root/'src/bf_tap_r2'/name)==sha
assert {k:importlib.metadata.version(k) for k in spec['runtime_versions']}==manifest['versions']
frame=load_v5_training_frame(root)
assert hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()==manifest['data_digest']
folds={s:fold_vector(root,frame,s,load_v5_spec(root)) for s in spec['split_seeds']+conf['confirmation_seeds']}
for s,f in folds.items():assert digest(f.tolist())==manifest['fold_digests'][str(s)]
ref,hashes=four_seed_references(root,frame,folds,spec,conf);assert hashes==manifest['reference_hashes']
events=[json.loads(line) for line in (out/'fit_ledger.jsonl').read_text().splitlines()]
assert len(events)==10 and all(e['event']=='complete' for e in events)
assert {(e['seed'],e['fold']) for e in events}=={(s,f) for s in conf['confirmation_seeds'] for f in range(5)}
bykey={(e['seed'],e['fold']):e for e in events};pred={s:np.full(len(frame),np.nan) for s in folds};checked={};budgets=[]
for s,foldvec in folds.items():
 for f in range(5):
  mask=foldvec==f
  if s in spec['split_seeds']:
   key=f'joint-task_gated_experts-s{s}-f{f}';path=dev/(key+'.npy');assert file_hash(path)==dev_audit['hashes'][key]
  else:
   event=bykey[(s,f)];path=out/f'seed-{s}-fold-{f}.npy';assert file_hash(path)==event['prediction_sha256']
   training=frame.loc[~mask].reset_index(drop=True);meta=event['metadata'];inner=group_safe_inner_folds(training,seed=42)['fold']!=0
   assert meta['fit_rows']==len(training) and meta['inner_fit_rows']==int(inner.sum()) and meta['optimizer_runs']==2
   assert meta['fit_ids_digest']==digest(training.sample_id.tolist()) and meta['inner_ids_digest']==digest(training.loc[inner,'sample_id'].tolist())
   np.testing.assert_allclose(meta['inner_feature_means'],training.loc[inner,list(FEATURES)].mean(),atol=1e-12)
   np.testing.assert_allclose(meta['outer_feature_means'],training[list(FEATURES)].mean(),atol=1e-12)
   assert meta['n_outputs']==2
   assert 1<=meta['selected_epoch']<=meta['selection_stopped_epoch']<=240
   assert meta['budget_limited']==(meta['selection_stopped_epoch']>=240)
   for label,part in [('inner',training.loc[inner]),('outer',training)]:
    np.testing.assert_allclose(meta[label+'_target_mean'],part[list(TARGETS)].mean().values,rtol=0,atol=1e-12)
    np.testing.assert_allclose(meta[label+'_target_std'],part[list(TARGETS)].std(ddof=0).values,rtol=0,atol=1e-12)
   gate=meta['gate_diagnostics'];assert gate['learned'] is True
   assert np.isfinite(gate['parameter_norm']) and gate['parameter_norm']>=0
   for key in ['mean_weights','min_weights','max_weights']:
    value=np.asarray(gate[key]);assert value.shape==(2,2) and np.isfinite(value).all() and (value>=0).all() and (value<=1).all()
   np.testing.assert_allclose(np.asarray(gate['mean_weights']).sum(-1),1,atol=1e-6)
   assert all(0<=x<=np.log(2)+1e-6 for x in gate['mean_entropy'])
   assert meta['parameter_count']==meta['trainable_parameter_count']>0
   budgets.append({k:meta[k] for k in ['selected_epoch','selection_stopped_epoch','budget_limited']})
  p=np.load(path,allow_pickle=False);assert p.shape==(int(mask.sum()),2) and np.isfinite(p).all();pred[s][mask]=p[:,1];checked[str(path.relative_to(root))]=file_hash(path)
assert all(np.isfinite(p).all() for p in pred.values())
y=frame.tap_time_len.values
error=lambda actual,p:float(np.abs(actual-p).sum()/np.abs(actual).sum())
records={}
for label,base in ref.items():
 stored=summary['comparisons'][label];gains={};alphas={};count=0
 for held in folds:
  scores=[np.mean([error(y,(1-a)*base[s]+a*pred[s]) for s in folds if s!=held]) for a in spec['blend_grid']]
  alpha=spec['blend_grid'][int(np.argmin(scores))];assert alpha==stored['alphas'][str(held)];alphas[held]=alpha
  blend=(1-alpha)*base[held]+alpha*pred[held];g=[]
  for f in range(5):
   mask=folds[held]==f;value=50*(error(y[mask],base[held][mask])-error(y[mask],blend[mask]));cell=next(c for c in stored['cells'] if c['seed']==held and c['fold']==f)
   assert abs(value-cell['delta_score'])<1e-12;g.append(value);count+=int(value>0)
  gains[held]=float(np.mean(g));assert abs(gains[held]-stored['seed_gains'][str(held)])<1e-12
 mean=float(np.mean(list(gains.values())));sd=float(np.std(list(gains.values()),ddof=1));lcb=mean-float(t.ppf(.95,3))*sd/2
 assert abs(mean-stored['seed_summary']['mean'])<1e-12 and abs(lcb-stored['seed_summary']['lcb95'])<1e-12
 passed=all(g>0 for g in gains.values()) and lcb>0
 if label in summary['decisions']:assert summary['decisions'][label]['admitted']==passed
 records[label]={'seed_gains':gains,'alphas':alphas,'mean':mean,'sd':sd,'lcb95':lcb,'positive_seeds':sum(g>0 for g in gains.values()),'positive_folds':count,'four_seed_pass':bool(passed)}
dev_summary=json.loads((dev/'summary.json').read_text())
row=next(r for r in dev_summary['records'] if r['target']=='tap_time_len' and r['recipe']=='task_gated_experts')
score=float(np.mean(list(row['development_package_scores'].values())))
assert abs(score-summary['development_package_score'])<1e-12
assert summary['local_working_gate_met']==(score>=96.25)
result={'status':'passed','prediction_count':len(checked),'hashes':checked,'training_budget':budgets,'comparisons':records,'development_package_score':score,'local_working_gate_met':score>=96.25,'new_fits':0,'joint_target_scale_reconstructions':20,'audit_code_sha256':file_hash(__file__)}
write_new(out/'audit-r1.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['hashes','training_budget']},indent=2))
