"""Budgeted V46 paired offline runner and independent saved-artifact audit."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse,hashlib,importlib.metadata,json,os,time,traceback
from pathlib import Path
import numpy as np
import yaml
from .data import TARGETS
from .v46_beta_nll import ARMS,default_settings,fit_partition,GaussianRegressor,eligible,final_decision
from .v46_cache import load_reference_cache
from .v7_periodic import digest
from .v3_4_bags import group_safe_inner_folds
SPEC='configs/round2_v46_beta_nll/SPEC.yaml'
G0='local/v46-beta-nll-20260929/G0-recovery-r1'

def write_new(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)

def read(path):return json.loads(Path(path).read_text())

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze_hashes(root,paths):return {str(Path(p).relative_to(root)):sha(p) for p in paths}

def verify_hashes(root,hashes):
 for name,value in hashes.items():
  if sha(Path(root)/name)!=value:raise ValueError('Frozen identity changed: '+name)

def runtime(root):
 spec=yaml.safe_load((Path(root)/SPEC).read_text())
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
  if os.environ.get(key)!='1':raise ValueError('Numeric thread cap must be1: '+key)
 for package,version in spec['runtime_versions'].items():
  if importlib.metadata.version(package)!=version:raise ValueError('Frozen runtime changed: '+package)
 return spec

def phase_tasks(phase,development_eligible=False):
 if phase=='development':seeds=[42,3407]
 elif phase=='confirmation' and development_eligible:seeds=[7777,12011]
 else:raise ValueError('Complete audited eligible development required')
 return [dict(arm=a,seed=s,fold=f) for s in seeds for f in range(5) for a in ARMS]

class FitLedger:
 def __init__(self,path,arm,seed,fold):self.path,self.arm,self.seed,self.fold=Path(path),arm,seed,fold
 def __call__(self,event):
  if event.get('stage') not in ['inner','outer'] or event.get('arm')!=self.arm or event.get('event')!='optimizer_started':
   raise ValueError('Unfrozen optimizer stage or arm')
  record=dict(event,seed=self.seed,fold=self.fold,time=time.time())
  write_new(self.path/('optimizer-'+event['stage']+'.json'),record)
  with (self.path/'optimizer-ledger.jsonl').open('a') as stream:
   stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())

def wmape(y,p):return float(np.abs(y-p).sum()/np.abs(y).sum())

def score_records(frame,folds,current,historical,predictions):
 if set(predictions)!=set(ARMS):raise ValueError('Complete paired coverage required')
 yi=frame.tap_iron.to_numpy();yt=frame.tap_time_len.to_numpy();records={a:{} for a in ARMS}
 for arm in ARMS:
  if set(predictions[arm])!=set(folds):raise ValueError('Complete seed coverage required')
  for seed,fv in folds.items():
   p=np.asarray(predictions[arm][seed])
   if p.shape!=yt.shape or not np.isfinite(p).all() or set(fv)!=set(range(5)):
    raise ValueError('Complete finite prediction coverage required')
   released=.8*current[seed]['tap_time_len']+.2*p
   base=100-50*(wmape(yi,current[seed]['tap_iron'])+wmape(yt,current[seed]['tap_time_len']))
   candidate=100-50*(wmape(yi,current[seed]['tap_iron'])+wmape(yt,released))
   old=100-50*(wmape(yi,historical[seed]['tap_iron'])+wmape(yt,historical[seed]['tap_time_len']))
   records[arm][str(seed)]=dict(folds=5,gain=candidate-base,candidate_score=candidate,reference_score=base,
    B0_gain=candidate-old,B0_score=old,member_time_wmape=wmape(yt,p),blend_time_wmape=wmape(yt,released),
    fold_gains=[50*(wmape(yt[fv==f],current[seed]['tap_time_len'][fv==f])-wmape(yt[fv==f],released[fv==f])) for f in range(5)])
 return records

def prepare(root,output):
 root=Path(root).resolve();output=Path(output).resolve()
 if not output.is_relative_to(root/'local/runs/round2-v46-beta-nll'):raise ValueError('Private V46 output required')
 spec=runtime(root);frame,folds,current,historical,cache=load_reference_cache(root)
 if not cache['all_four_seeds_verified']:raise ValueError('Four-seed cache prerequisite')
 base=root/G0;decision=read(base/'resource-decision.json')
 if not decision['passed']:raise ValueError('G0 resource gate')
 expected={f'{stage}-{arm}' for stage in ['resource','learnability'] for arm in ARMS}
 starts=[read(p) for p in base.glob('*-started.json')]
 ledger=[json.loads(line) for line in (base/'optimizer-ledger.jsonl').read_text().splitlines()]
 if len(starts)!=4 or len(ledger)!=4 or {x['key'] for x in starts}!=expected:
  raise ValueError('Exactly four G0 optimizer starts required')
 if {x['key'] for x in ledger}!=expected:raise ValueError('G0 ledger coverage')
 probe=root/'local/v46-beta-nll-20260929/g0_recovery_probe.py'
 if any(x['source_sha256']!=sha(probe) for x in starts):raise ValueError('G0 source identity changed')
 for stage in ['resource','learnability']:
  for arm in ARMS:
   report=read(base/f'{stage}-{arm}.json')
   if report['status']!='passed':raise ValueError('G0 probe failed')
 for seed,fv in folds.items():
  for fold in range(5):
   train=frame.loc[fv!=fold].reset_index(drop=True)
   inner=group_safe_inner_folds(train,seed=42)['fold']
   if len(train)>2204 or int((inner!=0).sum())>2204 or int((inner==0).sum())>551:
    raise ValueError('Measured resource row bounds exceeded')
 files=list((root/'src/bf_tap_r2').glob('*.py'))+[root/SPEC,probe]
 files+=list(base.glob('*.json'))+[base/'optimizer-ledger.jsonl']
 hashes=dict(cache['hashes']);hashes.update(freeze_hashes(root,files))
 manifest=dict(identity='V46_TIME_BETA_NLL',status='passed',spec_sha256=sha(root/SPEC),hashes=hashes,
  runtime_versions=spec['runtime_versions'],fold_hashes=cache['fold_hashes'],data_digest=cache['data_digest'],
  G0_optimizer_runs=4,resource=decision,reference_seeds=[42,3407,7777,12011],workers=4)
 output.parent.mkdir(parents=True,exist_ok=True);write_new(output,manifest)
 return manifest

def key(task):return f"{task['arm']}-s{task['seed']}-f{task['fold']}"

def run_fit(root,output,task,manifest,frame,fv):
 root=Path(root);output=Path(output);started=time.time()
 runtime(root);verify_hashes(root,manifest['hashes'])
 directory=output/key(task);directory.mkdir(exist_ok=False)
 write_new(directory/'fit-started.json',dict(task,time=started))
 try:
  mask=fv==task['fold'];training=frame.loc[~mask].reset_index(drop=True)
  query=frame.loc[mask].drop(columns=list(TARGETS)).reset_index(drop=True)
  selector,model,prediction,metadata=fit_partition(training,query,task['arm'],default_settings(),
    FitLedger(directory,task['arm'],task['seed'],task['fold']))
  selector.save(directory/'inner.pt');model.save(directory/'outer.pt')
  inner=group_safe_inner_folds(training,seed=42)['fold'];held=training.loc[inner==0].drop(columns=list(TARGETS)).reset_index(drop=True)
  with (directory/'inner-calibration.npy').open('xb') as stream:np.save(stream,selector.predict(held),allow_pickle=False)
  with (directory/'prediction.npy').open('xb') as stream:np.save(stream,prediction,allow_pickle=False)
  metadata.update(task=task,query_ids=query.sample_id.astype(str).tolist(),
    training_ids=training.sample_id.astype(str).tolist(),elapsed_seconds=time.time()-started)
  write_new(directory/'metadata.json',metadata)
  files=[p for p in directory.iterdir() if p.is_file()]
  completion=dict(task,status='complete',hashes=freeze_hashes(root,files))
  write_new(directory/'complete.json',completion)
  return completion
 except Exception as exc:
  write_new(directory/'failure.json',dict(task,status='failed',error=repr(exc),traceback=traceback.format_exc()))
  raise

def reserve_phase(root,output,phase,preflight_sha256):
 if phase not in ['development','confirmation']:raise ValueError('Unfrozen experiment phase')
 base=Path(root)/'local/runs/round2-v46-beta-nll/phase-reservations'
 base.mkdir(parents=True,exist_ok=True)
 path=base/f'{phase}.json'
 write_new(path,dict(identity='V46_TIME_BETA_NLL',phase=phase,
  output=str(Path(output).relative_to(root)),outer_fit_allocation=20,optimizer_allocation=40,
  preflight_sha256=preflight_sha256,time=time.time(),failure_does_not_release_allocation=True))
 return path

def settle_completed(pending,done,refill):
 completed=list(done)
 try:
  for future in completed:future.result()
 except Exception:
  for future in pending:future.cancel()
  raise
 tasks=[pending.pop(future) for future in completed]
 for _ in tasks:refill()
 return tasks

def run_batch(root,output,preflight,phase,development=None):
 root=Path(root).resolve();output=Path(output).resolve()
 if not output.is_relative_to(root/'local/runs/round2-v46-beta-nll'):raise ValueError('Private V46 output required')
 manifest=read(preflight);runtime(root);verify_hashes(root,manifest['hashes'])
 selected=False
 if phase=='confirmation':
  if development is None:raise ValueError('Audited eligible development required')
  selected=audit(root,Path(development))['selected_for_confirmation']==['BETA05']
 tasks=phase_tasks(phase,selected)
 frame,folds,_,_,cache=load_reference_cache(root)
 if cache['fold_hashes']!=manifest['fold_hashes']:raise ValueError('Frozen folds changed')
 if output.exists():raise FileExistsError(output)
 reservation=reserve_phase(root,output,phase,sha(preflight))
 output.mkdir(parents=True,exist_ok=False)
 record=dict(manifest,phase=phase,tasks=tasks,preflight_sha256=sha(preflight),
   phase_reservation=str(reservation.relative_to(root)),phase_reservation_sha256=sha(reservation),
   development=str(Path(development).resolve().relative_to(root)) if development else None,
   preflight_path=str(Path(preflight).resolve().relative_to(root)))
 write_new(output/'manifest.json',record)
 remaining=iter(tasks);complete=0
 try:
  with ProcessPoolExecutor(max_workers=4) as pool:
   pending={}
   def submit_next():
    task=next(remaining,None)
    if task is not None:pending[pool.submit(run_fit,root,output,task,manifest,frame,folds[task['seed']])]=task
   for _ in range(4):submit_next()
   while pending:
    done,_=wait(pending,return_when=FIRST_COMPLETED)
    for task in settle_completed(pending,done,submit_next):
     complete+=1
     write_new(output/f'progress-{complete:02}.json',dict(completed_outer_fits=complete,total_outer_fits=20,time=time.time()))
     print(json.dumps(dict(completed=complete,total=20,task=task)),flush=True)
  write_new(output/'training-complete.json',dict(status='complete',outer_fits=20,optimizer_runs=40))
 except Exception as exc:
  write_new(output/'batch-failure.json',dict(status='failed',completed_outer_fits=complete,error=repr(exc),traceback=traceback.format_exc()))
  raise
 return output

def cold_difference(model_path,query,expected):
 model=GaussianRegressor.load(model_path)
 plain=model.predict(query);chunk=model.predict(query,chunk_rows=127)
 order=np.random.default_rng(42).permutation(len(query));shuffled=model.predict(query.iloc[order],chunk_rows=37)
 restored=np.empty_like(shuffled);restored[order]=shuffled
 difference=max(float(np.max(np.abs(expected-plain))),float(np.max(np.abs(expected-chunk))),float(np.max(np.abs(expected-restored))))
 if difference>1e-8:raise ValueError('Cold/order/chunk prediction failure')
 return difference

def audit(root,output):
 """Recompute from original caches and cold saved models; ignore summaries."""
 root=Path(root).resolve();output=Path(output).resolve()
 manifest=read(output/'manifest.json');runtime(root);verify_hashes(root,manifest['hashes'])
 if sha(root/manifest['preflight_path'])!=manifest['preflight_sha256']:raise ValueError('Preflight changed')
 reservation=root/manifest['phase_reservation']
 if sha(reservation)!=manifest['phase_reservation_sha256'] or read(reservation)['output']!=str(output.relative_to(root)):
  raise ValueError('Experiment phase reservation changed')
 completion=read(output/'training-complete.json')
 if completion!={'status':'complete','outer_fits':20,'optimizer_runs':40} or (output/'batch-failure.json').exists():
  raise ValueError('Complete successful formal coverage required')
 frame,folds,current,historical,cache=load_reference_cache(root)
 if cache['fold_hashes']!=manifest['fold_hashes'] or cache['data_digest']!=manifest['data_digest']:
  raise ValueError('Reference data/folds changed')
 tasks=phase_tasks(manifest['phase'],manifest['phase']=='confirmation')
 if manifest['tasks']!=tasks:raise ValueError('Frozen task pool changed')
 seeds=sorted({t['seed'] for t in tasks})
 predictions={a:{s:np.full(len(frame),np.nan) for s in seeds} for a in ARMS}
 audit_hashes={};cold_checks=[];optimizer_starts=0
 actual={p.name for p in output.iterdir() if p.is_dir()}
 if actual!={key(task) for task in tasks}:raise ValueError('Exact paired task coverage required')
 for task in tasks:
  directory=output/key(task)
  completed=read(directory/'complete.json')
  if completed['status']!='complete' or any(completed[k]!=v for k,v in task.items()) or (directory/'failure.json').exists():
   raise ValueError('Complete fit identity required')
  verify_hashes(root,completed['hashes']);audit_hashes.update(completed['hashes'])
  audit_hashes[str((directory/'complete.json').relative_to(root))]=sha(directory/'complete.json')
  meta=read(directory/'metadata.json');fv=folds[task['seed']];mask=fv==task['fold']
  training=frame.loc[~mask].reset_index(drop=True);query=frame.loc[mask].drop(columns=list(TARGETS)).reset_index(drop=True)
  inner=group_safe_inner_folds(training,seed=42)['fold'];fitting=training.loc[inner!=0].reset_index(drop=True)
  held=training.loc[inner==0].drop(columns=list(TARGETS)).reset_index(drop=True)
  if meta['task']!=task or meta['query_ids']!=query.sample_id.astype(str).tolist() or meta['training_ids']!=training.sample_id.astype(str).tolist():
   raise ValueError('Training/query identity mismatch')
  if meta['query_ids_digest']!=digest(query.sample_id.astype(str).tolist()) or meta['calibration_ids_digest']!=digest(held.sample_id.astype(str).tolist()):
   raise ValueError('Query/calibration digest mismatch')
  for stage,data in [('inner',fitting),('outer',training)]:
   info=meta[stage]
   if info['fit_rows']!=len(data) or info['fit_ids_digest']!=digest(data.sample_id.astype(str).tolist()) or info['settings']!=default_settings() or info['arm']!=task['arm']:
    raise ValueError('Native refit identity mismatch')
   if len(info['history'])!=info['stopped_epoch'] or not 1<=info['selected_epoch']<=info['stopped_epoch']<=240:
    raise ValueError('Frozen epoch/trace identity mismatch')
   event=read(directory/f'optimizer-{stage}.json')
   if event['event']!='optimizer_started' or event['stage']!=stage or event['arm']!=task['arm'] or event['seed']!=task['seed'] or event['fold']!=task['fold']:
    raise ValueError('Optimizer ledger identity mismatch')
  ledger=[json.loads(line) for line in (directory/'optimizer-ledger.jsonl').read_text().splitlines()]
  if [x['stage'] for x in ledger]!=['inner','outer'] or len(list(directory.glob('optimizer-*.json')))!=2:
   raise ValueError('Exactly two optimizer starts per formal fit required')
  optimizer_starts+=2
  if meta['outer']['selected_epoch']!=meta['inner']['selected_epoch'] or meta['outer']['stopped_epoch']!=meta['inner']['selected_epoch']:
   raise ValueError('Fresh-refit selected epoch mismatch')
  p=np.load(directory/'prediction.npy',allow_pickle=False)
  calibration=np.load(directory/'inner-calibration.npy',allow_pickle=False)
  if p.shape!=(len(query),) or not np.isfinite(p).all() or calibration.shape!=(len(held),) or not np.isfinite(calibration).all():
   raise ValueError('Prediction coverage mismatch')
  diff=cold_difference(directory/'outer.pt',query,p)
  diff=max(diff,cold_difference(directory/'inner.pt',held,calibration))
  cold_checks.append(dict(task,max_difference=diff))
  predictions[task['arm']][task['seed']][mask]=p
 if optimizer_starts!=40:raise ValueError('Formal optimizer budget mismatch')
 records=score_records(frame,{s:folds[s] for s in seeds},current,historical,predictions)
 selected=[];decision=None
 if manifest['phase']=='development':
  selected=['BETA05'] if eligible(records) else []
 else:
  dev=audit(root,root/manifest['development'])
  if dev['selected_for_confirmation']!=['BETA05']:raise ValueError('Confirmation has no eligible development')
  merged={a:{**dev['records'][a],**records[a]} for a in ARMS}
  decision=final_decision(merged)
  audit_hashes.update(dev['hashes'])
  records=merged
 result=dict(status='passed',identity='V46_TIME_BETA_NLL',phase=manifest['phase'],records=records,
   selected_for_confirmation=selected,decision=decision,outer_fits=20,optimizer_starts=optimizer_starts,
   cold_checks=cold_checks,hashes=audit_hashes,G0='passed',G1='measured_no_forecast',packages=0,uploads=0)
 for name in ['manifest.json','training-complete.json']:result['hashes'][str((output/name).relative_to(root))]=sha(output/name)
 index=1
 while (output/f'audit-r{index}.json').exists():index+=1
 write_new(output/f'audit-r{index}.json',result)
 print(json.dumps({k:result[k] for k in ['status','phase','records','selected_for_confirmation','decision']}),flush=True)
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--preflight',action='store_true')
 parser.add_argument('--audit',action='store_true')
 parser.add_argument('--phase',choices=['development','confirmation'])
 parser.add_argument('--preflight-file',type=Path)
 parser.add_argument('--development',type=Path)
 args=parser.parse_args();root=Path.cwd()
 if args.preflight:prepare(root,args.output)
 elif args.audit:audit(root,args.output)
 elif args.phase and args.preflight_file:run_batch(root,args.output,args.preflight_file,args.phase,args.development)
 else:parser.error('Choose preflight, audit or phase with preflight-file')
 return 0

if __name__=='__main__':raise SystemExit(main())
