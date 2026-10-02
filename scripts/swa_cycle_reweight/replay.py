"""Fixed zero-fit replay of saved cycle-tail SWA recency hypotheses."""
from pathlib import Path
import argparse,json,math,os,resource,subprocess,sys,time,traceback
import numpy as np
from bf_tap_r2.tabm_swa_protocol import sha,verify_tree,write_new,digest
from bf_tap_r2.swa_cycle_reweight import POOL,reweighted_state,replay_candidates,exploration_decision
RUN='local/runs/swa-cycle-reweight-v1';AUX='local/swa-cycle-reweight-20261002';BRANCH='codex/swa-cycle-reweight-v1'
SPEC='configs/swa_cycle_reweight_v1/SPEC.json';SCRIPT='scripts/swa_cycle_reweight/replay.py';SOURCE='local/runs/swa-cycle-tail-v1';SEEDS=(42,3407,271828,314159)
def source_dir(root,s):return root/SOURCE/('development-r1' if s in (42,3407) else 'confirmation-r1')
def sources(root):
    p=list((root/'src/bf_tap_r2').glob('*.py'))+list((root/'tests').glob('*.py'))
    p += [root/SPEC,root/SCRIPT,root/'docs/swa_cycle_reweight_v1/STRATEGY.md',root/'configs/protection.yaml',root/'uv.lock',root/'pyproject.toml']
    return {str(v.relative_to(root)):sha(v) for v in p}
def check(root,frozen,full=False):
    if subprocess.check_output(['git','branch','--show-current'],text=True,cwd=root).strip()!=BRANCH:raise ValueError('Different branch; no switching')
    verify_tree(root,frozen['source_hashes'])
    if full:verify_tree(root,frozen['reference_files'])
    from bf_tap_r2.tabm_swa_run import runtime
    if runtime()!=frozen['runtime']:raise ValueError('Frozen runtime differs')
def threads():
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        if os.environ.get(key)!='1':raise ValueError('Numeric thread must be one')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
def frame_parts(root,s):
    import pandas as pd
    from bf_tap_r2.data import FEATURES
    with np.load(source_dir(root,s)/'inputs.npz',allow_pickle=False) as a:
        f=pd.DataFrame(a['x'],columns=FEATURES);f['spout_no']=a['spouts'];f['sample_id']=a['ids']
        parts={i:{key:a[key+f'__s{s}-f{i}'].copy() for key in ['query','train','inner','cal']} for i in range(5)}
    return f,parts

def cold_states(witness):
    """Separate NumPy float64 weighting, without using production averaging."""
    import torch
    out={}
    for mode in POOL:
        raw=witness['states'][-3:] if mode=='CYCLE_LAST3' else witness['states']
        weights=np.ones(len(raw),float) if mode=='CYCLE_LAST3' else np.arange(1,len(raw)+1,dtype=float)
        state={}
        for key,first in raw[0].items():
            if first.is_floating_point():
                matrix=np.stack([r[key].detach().cpu().numpy() for r in raw]).astype(np.float64)
                mean=np.average(matrix,axis=0,weights=weights).astype(first.detach().cpu().numpy().dtype)
                state[key]=torch.from_numpy(np.asarray(mean).copy())
            else:state[key]=first.detach().cpu().clone()
        out[mode]=state
    return out

def worker(root,s,fold,cold=False):
    import torch
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.cycle_swa_protocol import verify_cycle_window
    from bf_tap_r2.data import FEATURES
    def forbidden(*args,**kw):raise RuntimeError('Fits and optimizers prohibited in replay')
    ComponentRegressor.fit=forbidden;torch.optim.AdamW=forbidden;torch.optim.Adam=forbidden
    frozen=json.loads((root/RUN/'freeze.json').read_text());check(root,frozen)
    source=source_dir(root,s)/f's{s}-f{fold}';old=json.loads((source/'complete.json').read_text());verify_tree(source,old['hashes'])
    frame,parts=frame_parts(root,s);q=parts[fold]['query'];train=parts[fold]['train']
    with np.load(source/'partitions.npz',allow_pickle=False) as a:
        for k,v in parts[fold].items():np.testing.assert_array_equal(a[k],v)
    groups=__import__('pandas').util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy()
    if set(groups[q])&set(groups[train]):raise ValueError('Duplicate leakage')
    model=ComponentRegressor.load(source/'SWA_CYCLE_TAIL/refit.pt');witness=torch.load(source/'SWA_CYCLE_TAIL/refit-window.pt',map_location='cpu',weights_only=True)
    verify_cycle_window(model.saved,witness)
    if model.saved['trace']['fit_ids_digest']!=digest(frame.iloc[train].sample_id.astype(str).tolist()):raise ValueError('Model fit identity differs')
    held=frame.iloc[q].reset_index(drop=True);ids=np.asarray(held.sample_id,dtype=str);unit=root/RUN/f's{s}-f{fold}'
    before={k:v.detach().clone() for k,v in model.model_.state_dict().items()};maximum=0.;difference=0.
    if cold:
        states=cold_states(witness);saved=torch.load(unit/'reweighted-states.pt',map_location='cpu',weights_only=True);columns={}
        try:
            for mode in POOL:
                for key,v in states[mode].items():
                    if v.is_floating_point():
                        np.testing.assert_allclose(v.numpy(),saved[mode][key].numpy(),rtol=1e-6,atol=5e-7)
                        maximum=max(maximum,float(np.max(np.abs(v.numpy()-saved[mode][key].numpy()))))
                    else:np.testing.assert_array_equal(v.numpy(),saved[mode][key].numpy())
                model.model_.load_state_dict(states[mode]);model.model_.eval();reverse=held.iloc[::-1].reset_index(drop=True)
                columns[mode]=np.concatenate([model.predict(reverse.iloc[i:i+37])[:,1] for i in range(0,len(reverse),37)])[::-1]
        finally:model.model_.load_state_dict(before);model.model_.eval()
        with np.load(unit/'predictions.npz',allow_pickle=False) as a:
            np.testing.assert_array_equal(ids,a['ids']);np.testing.assert_array_equal(witness['epochs'],a['epochs'])
            for mode in POOL:
                np.testing.assert_allclose(columns[mode],a[mode],rtol=0,atol=5e-4)
                difference=max(difference,float(np.max(np.abs(columns[mode]-a[mode]))))
        status={'status':'passed_independent_numpy_window_reverse_chunk_cold','max_prediction_difference':difference,'max_window_parameter_difference':maximum}
    else:
        columns=replay_candidates(model,witness,held);states={mode:reweighted_state(witness,mode) for mode in POOL}
        unit.mkdir(exist_ok=False);torch.save(states,unit/'reweighted-states.pt')
        with (unit/'predictions.npz').open('xb') as f:np.savez(f,ids=ids,epochs=witness['epochs'],**columns)
        status={'status':'completed_zero_fit_reweight_replay','source_directory':str(source.relative_to(root)),'source_model_sha256':sha(source/'SWA_CYCLE_TAIL/refit.pt'),
                'source_window_sha256':sha(source/'SWA_CYCLE_TAIL/refit-window.pt'),'source_complete_sha256':sha(source/'complete.json'),
                'predictions_sha256':sha(unit/'predictions.npz'),'reweighted_states_sha256':sha(unit/'reweighted-states.pt'),'raw_states':len(witness['states'])}
    for k,v in model.model_.state_dict().items():
        if not torch.equal(v,before[k]):raise ValueError('Original model state not restored')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if rss>1024:raise ValueError('RSS limit exceeded')
    check(root,frozen);write_new(unit/('cold-audit.json' if cold else 'complete.json'),dict(**status,peak_rss_mib=rss,new_fits=0,new_optimizers=0,new_control_inferences=0))

def access(root):
    with (root/RUN/'label-access.jsonl').open('a') as f:
        f.write(json.dumps({'event':'before_official_round2_cached_labels','freeze_sha256':sha(root/RUN/'freeze.json'),'protection_sha256':sha(root/'configs/protection.yaml'),'protected_labels_requested':False,'new_fits':0,'time':time.time()})+'\n');f.flush();os.fsync(f.fileno())
def evaluate(root,frozen):
    from bf_tap_r2.cycle_swa_run import parents
    access(root);parent=parents(root);rows={};gains={mode:[] for mode in POOL};controls=[];allcold=[]
    for s in SEEDS:
        frame,parts=frame_parts(root,s);n=len(frame);columns={mode:np.full(n,np.nan) for mode in POOL};control=np.full(n,np.nan);seen=np.zeros(n,int)
        with np.load(source_dir(root,s)/'inputs.npz',allow_pickle=False) as a:y=a['y'][:,1].copy()
        for fold in range(5):
            unit=root/RUN/f's{s}-f{fold}';q=parts[fold]['query'];c=json.loads((unit/'complete.json').read_text());audit=json.loads((unit/'cold-audit.json').read_text())
            if c['predictions_sha256']!=sha(unit/'predictions.npz') or c['reweighted_states_sha256']!=sha(unit/'reweighted-states.pt') or audit['status']!='passed_independent_numpy_window_reverse_chunk_cold':raise ValueError('Complete cold evidence missing')
            with np.load(unit/'predictions.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'],np.asarray(frame.iloc[q].sample_id,dtype=str))
                for mode in POOL:columns[mode][q]=a[mode]
            with np.load(source_dir(root,s)/f's{s}-f{fold}/predictions.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'],np.asarray(frame.iloc[q].sample_id,dtype=str));control[q]=a['SWA_CYCLE_TAIL']
            seen[q]+=1;allcold.append(audit)
        if n!=2754 or len(set(frame.sample_id))!=2754 or not np.all(seen==1):raise ValueError('Complete same-seed coverage required')
        mass=math.fsum(float(v) for v in y);base=math.fsum(abs(float(v-p)) for v,p in zip(y,parent[s]));rows[str(s)]={}
        for mode,column in dict(columns,CYCLE_UNIFORM5_CONTROL=control).items():
            endpoint=.8*parent[s]+.2*column
            if not np.isfinite(endpoint).all() or (endpoint<0).any():raise ValueError('Invalid endpoint; do not clip')
            error=math.fsum(abs(float(v-p)) for v,p in zip(y,endpoint));gain=50*(base-error)/mass
            rows[str(s)][mode]={'gain':gain,'endpoint_wmape':error/mass,'member_wmape':math.fsum(abs(float(v-p)) for v,p in zip(y,column))/mass}
            (controls if mode=='CYCLE_UNIFORM5_CONTROL' else gains[mode]).append(gain)
            with (root/RUN/f'endpoint-{mode}-s{s}.npy').open('xb') as f:np.save(f,endpoint,allow_pickle=False)
        with (root/RUN/f'oof-s{s}.npz').open('xb') as f:np.savez(f,ids=np.asarray(frame.sample_id,dtype=str),y=y,parent=parent[s],control=control,**columns)
    decision=exploration_decision(gains,controls)
    write_new(root/RUN/'evaluation.json',{'rows':rows,'decision':decision,'complete_package_local_score_available':False,'platform_reference_user_reported':96.392})
    write_new(root/RUN/'audit.json',{'status':'passed_complete_20_windows_40_parameter_transforms_independent_cold','max_prediction_difference':max(a['max_prediction_difference'] for a in allcold),'max_window_parameter_difference':max(a['max_window_parameter_difference'] for a in allcold),'peak_rss_mib':max(a['peak_rss_mib'] for a in allcold),'production_units':20,'independent_cold_units':20,'new_fits':0,'new_optimizers':0,'new_control_inferences':0})
    return decision

def prepare(root):
    from bf_tap_r2.tabm_swa_run import runtime
    root=Path(root).resolve();run=root/RUN
    if run.exists():raise ValueError('Existing run must not be restarted')
    spec=json.loads((root/SPEC).read_text())
    if spec['candidate_pool']!=list(POOL) or spec['split_seeds']!=list(SEEDS) or spec['formal_promotion_allowed'] or spec['alpha']!=.2:raise ValueError('SPEC differs')
    previous=json.loads((root/SOURCE/'preflight.json').read_text());verify_tree(root,previous['source_hashes']);verify_tree(root,previous['reference_files'])
    receipt=json.loads((root/AUX/'tests-complete-r1.json').read_text())
    if receipt['status']!='passed' or receipt['source_hashes']!=sources(root):raise ValueError('Missing or stale necessary tests')
    verify_tree(root,receipt['logs']);deps=dict(previous['reference_files'])
    for p in (root/SOURCE).rglob('*'):
        if p.is_file():deps[str(p.relative_to(root))]=sha(p)
    for phase in ['development','confirmation']:
        audit=json.loads((root/SOURCE/(phase+'-r1')/'audit.json').read_text())
        if audit['status']!='passed' or audit['new_fits']!=0:raise ValueError('Source cold audit not passed')
    frozen={'source_hashes':sources(root),'reference_files':deps,'runtime':runtime(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'new_fits':0,'new_optimizers':0,'tests_receipt_sha256':sha(root/AUX/'tests-complete-r1.json')}
    check(root,frozen,full=True);run.mkdir(parents=True,exist_ok=False);write_new(run/'freeze.json',frozen);write_new(run/'preflight.json',dict(frozen,status='necessary_engineering_tests_and_source_cold_evidence_passed'))
def execute(root):
    frozen=json.loads((root/RUN/'freeze.json').read_text());check(root,frozen,full=True)
    write_new(root/RUN/'controller-started.json',{'pid':os.getpid(),'time':time.time()})
    try:
        for s in SEEDS:
            for fold in range(5):
                for action in ['worker','cold']:
                    with (root/RUN/f's{s}-f{fold}-{action}.log').open('x') as log:subprocess.run([sys.executable,'-u',SCRIPT,action,'--seed',str(s),'--fold',str(fold)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=root)
        decision=evaluate(root,frozen);check(root,frozen,full=True)
        write_new(root/RUN/'controller-finished.json',{'status':'completed_zero_fit_post_selection_exploration','decision':decision,'new_fits':0,'new_optimizers':0,'reference_fits':0,'fullfit':0,'packages':0,'uploads':0,'time':time.time()})
    except BaseException as e:
        write_new(root/RUN/'controller-failed.json',{'status':'failed_evidence_preserved','error':repr(e),'traceback':traceback.format_exc(),'new_fits':0,'new_optimizers':0,'time':time.time()});raise
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','execute','worker','cold']);p.add_argument('--seed',type=int);p.add_argument('--fold',type=int);args=p.parse_args();threads();root=Path.cwd().resolve()
    if args.action=='prepare':prepare(root)
    elif args.action=='execute':execute(root)
    else:worker(root,args.seed,args.fold,args.action=='cold')