"""Exclusive zero-fit same-window replay and independent new-process cold audit."""
from pathlib import Path
import argparse,json,math,os,resource,subprocess,sys,time,traceback
import numpy as np
from scipy.stats import t
from bf_tap_r2.tabm_swa_protocol import sha,verify_tree,write_new,digest
from bf_tap_r2.swa_predict_avg import replay_window

SEEDS=(42,3407,271828,314159)
BRANCH='codex/swa-predict-avg-v1'
RUN='local/runs/swa-predict-avg-v1'
SPEC='configs/swa_predict_avg_v1/SPEC.json'
AUX='local/swa-predict-avg-20261002'
DEV='local/runs/tabm-time-swa-v1/development-r1'
CONF='local/runs/tabm-time-swa-confirm-271828-314159-20261001/confirmation-r1'
SCRIPT='scripts/swa_predict_avg/replay.py'

def source_dir(root,seed):return root/(DEV if seed in (42,3407) else CONF)
def assert_branch(root):
    if subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()!=BRANCH:raise ValueError('Different branch; no switch')
def threads():
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        if os.environ.get(name)!='1':raise ValueError('Single numeric threads required')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)

def check(root,frozen):
    assert_branch(root);verify_tree(root,frozen['source_hashes']);verify_tree(root,frozen['reference_files'])
    from bf_tap_r2.tabm_swa_run import runtime
    if runtime()!=frozen['runtime']:raise ValueError('Runtime differs')
def source_hashes(root):
    paths=list((root/'src/bf_tap_r2').glob('*.py'))+list((root/'tests').glob('*.py'))
    paths += [root/SCRIPT,root/SPEC,root/'docs/swa_predict_avg_v1/STRATEGY.md',root/'configs/protection.yaml',root/'uv.lock',root/'pyproject.toml']
    paths += [root/AUX/n for n in ['start_once.py','monitor_once.py']]
    return {str(p.relative_to(root)):sha(p) for p in paths}

def inputs(root,seed):
    import pandas as pd
    from bf_tap_r2.data import FEATURES
    d=source_dir(root,seed)
    with np.load(d/'inputs.npz',allow_pickle=False) as a:
        frame=pd.DataFrame(a['x'],columns=FEATURES);frame['spout_no']=a['spouts'];frame['sample_id']=a['ids']
        parts={f:{n:a[n+f'__s{seed}-f{f}'].copy() for n in ['query','train','inner','cal']} for f in range(5)}
    return frame,parts

def access(root,frozen,reason):
    p=root/RUN/'label-access.jsonl'
    with p.open('a') as stream:
        stream.write(json.dumps(dict(time=time.time(),event='before_official_round2_cached_label_read',reason=reason,
                                    freeze_sha256=sha(root/RUN/'freeze.json'),protection_sha256=sha(root/'configs/protection.yaml'),protected_labels_requested=False))+'\n')
        stream.flush();os.fsync(stream.fileno())

def worker(root,seed,fold,cold=False):
    import torch
    from bf_tap_r2.component_regularization import ComponentRegressor
    from bf_tap_r2.data import FEATURES
    def forbidden(*args,**kw):raise RuntimeError('No fit or optimizer permitted')
    ComponentRegressor.fit=forbidden;torch.optim.AdamW=forbidden;torch.optim.Adam=forbidden
    run=root/RUN;frozen=json.loads((run/'freeze.json').read_text());check(root,frozen)
    source=source_dir(root,seed)/f's{seed}-f{fold}';c=json.loads((source/'complete.json').read_text());verify_tree(source,c['hashes'])
    frame,parts=inputs(root,seed);q=parts[fold]['query'];train=parts[fold]['train'];held=frame.iloc[q].reset_index(drop=True)
    groups=__import__('pandas').util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy()
    if set(groups[q])&set(groups[train]):raise ValueError('Duplicate leakage')
    with np.load(source/'partitions.npz',allow_pickle=False) as a:
        for k,indices in parts[fold].items():np.testing.assert_array_equal(a[k],indices)
    model=ComponentRegressor.load(source/'SWA_WINDOW10/refit.pt')
    witness=torch.load(source/'SWA_WINDOW10/refit-window.pt',map_location='cpu',weights_only=True)
    if model.saved['mechanisms']!={'uniform_epoch_window':10} or model.saved['trace']['fit_ids_digest']!=digest(frame.iloc[train].sample_id.astype(str).tolist()):raise ValueError('Model source identity differs')
    # Production replays raw states on full query. Independent cold replays
    # reversed query in chunks to check inference order/batch invariance.
    if cold:
        reverse=held.iloc[::-1].reset_index(drop=True);matrices=[]
        for start in range(0,len(reverse),37):
            matrix,_=replay_window(model,witness,reverse.iloc[start:start+37]);matrices.append(matrix)
        matrix=np.concatenate(matrices,axis=1)[:,::-1]
        average=np.asarray([math.fsum(float(v) for v in column)/len(column) for column in matrix.T])
    else:matrix,average=replay_window(model,witness,held)
    unit=run/f's{seed}-f{fold}';ids=np.asarray(held.sample_id,dtype=str)
    if cold:
        with np.load(unit/'predictions.npz',allow_pickle=False) as a:
            np.testing.assert_array_equal(ids,a['ids']);np.testing.assert_array_equal(witness['epochs'],a['epochs'])
            difference=float(np.max(np.abs(matrix-a['matrix'])))
            np.testing.assert_allclose(matrix,a['matrix'],rtol=0,atol=5e-4);np.testing.assert_allclose(average,a['average'],rtol=0,atol=5e-4)
            scalar=np.asarray([math.fsum(float(v) for v in col)/matrix.shape[0] for col in a['matrix'].T])
            np.testing.assert_allclose(scalar,a['average'],rtol=0,atol=1e-12)
        status={'status':'passed_independent_reversed_chunked_raw_state_cold','max_difference':difference,'new_fits':0,'new_optimizers':0,'new_control_inferences':0}
    else:
        unit.mkdir(exist_ok=False)
        with (unit/'predictions.npz').open('xb') as stream:np.savez(stream,ids=ids,epochs=witness['epochs'],matrix=matrix,average=average)
        status={'status':'complete_zero_fit_raw_epoch_replay','source_directory':str(source.relative_to(root)),'seed':seed,'fold':fold,
                'source_complete_sha256':sha(source/'complete.json'),'query_ids_digest':digest(ids.tolist()),'model_sha256':sha(source/'SWA_WINDOW10/refit.pt'),
                'window_sha256':sha(source/'SWA_WINDOW10/refit-window.pt'),'predictions_sha256':sha(unit/'predictions.npz'),'raw_states':len(witness['states']),
                'new_fits':0,'new_optimizers':0,'new_control_inferences':0}
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if rss>1024:raise ValueError('Worker RSS exceeded')
    check(root,frozen);write_new(unit/('cold-audit.json' if cold else 'complete.json'),dict(**status,peak_rss_mib=rss))

def evaluate(root,frozen):
    from bf_tap_r2.time_swa_run import parents
    from bf_tap_r2.time_swa_run import readonly_verifier,BUNDLE
    # Prior bundle closure is byte frozen; parent arithmetic/source audit is verified.
    access(root,frozen,'complete OOF scalar comparison and Q75 source reindex');parent=parents(root)
    rows={};gains=[];controls=[];cold=[]
    for seed in SEEDS:
        frame,parts=inputs(root,seed);n=len(frame);seen=np.zeros(n,int);member=np.full(n,np.nan);control=np.full(n,np.nan)
        with np.load(source_dir(root,seed)/'inputs.npz',allow_pickle=False) as a:y=a['y'][:,1].copy()
        for f in range(5):
            q=parts[f]['query'];unit=root/RUN/f's{seed}-f{f}';c=json.loads((unit/'complete.json').read_text());audit=json.loads((unit/'cold-audit.json').read_text())
            if c['predictions_sha256']!=sha(unit/'predictions.npz') or audit['status']!='passed_independent_reversed_chunked_raw_state_cold':raise ValueError('Missing complete cold audit')
            with np.load(unit/'predictions.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'],np.asarray(frame.iloc[q].sample_id,dtype=str));member[q]=a['average']
            with np.load(source_dir(root,seed)/f's{seed}-f{f}/predictions.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['ids'],np.asarray(frame.iloc[q].sample_id,dtype=str));control[q]=a['SWA_WINDOW10']
            seen[q]+=1;cold.append(audit)
        if len(frame)!=2754 or len(set(frame.sample_id))!=2754 or not np.all(seen==1):raise ValueError('Complete same-seed OOF required')
        baseline=math.fsum(abs(float(v-p)) for v,p in zip(y,parent[seed]));mass=math.fsum(float(v) for v in y)
        rows[str(seed)]={}
        for name,column in [('PREDICT_MEAN10',member),('PARAMETER_MEAN10_CONTROL',control)]:
            p=.8*parent[seed]+.2*column
            if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid endpoint; no clipping')
            benefit=50*(baseline-math.fsum(abs(float(v-b)) for v,b in zip(y,p)))/mass
            rows[str(seed)][name]={'gain':benefit,'member_wmape':math.fsum(abs(float(v-b)) for v,b in zip(y,column))/mass,
                                     'endpoint_wmape':math.fsum(abs(float(v-b)) for v,b in zip(y,p))/mass}
            with (root/RUN/f'endpoint-{name}-s{seed}.npy').open('xb') as stream:np.save(stream,p,allow_pickle=False)
        gains.append(rows[str(seed)]['PREDICT_MEAN10']['gain']);controls.append(rows[str(seed)]['PARAMETER_MEAN10_CONTROL']['gain'])
        with (root/RUN/f'oof-s{seed}.npz').open('xb') as stream:np.savez(stream,ids=np.asarray(frame.sample_id,dtype=str),y=y,parent=parent[seed],candidate=member,control=control)
    mean=math.fsum(gains)/4;mechanism=math.fsum(a-b for a,b in zip(gains,controls))/4
    lcb=mean-float(t.ppf(.95,3))*math.sqrt(math.fsum((a-mean)**2 for a in gains)/3)/2
    keep=mean>0 and mechanism>0
    decision={'classification':'post_selection_exploration_only_not_release_authorized','formal_promoted':False,'seed_gains':gains,'control_gains':controls,
              'mean_gain':mean,'mechanism_mean':mechanism,'seed_lcb95_descriptive':lcb,'exploration_selected':['PREDICT_MEAN10'] if keep else [],'release_authorized':False}
    write_new(root/RUN/'evaluation.json',dict(rows=rows,decision=decision,complete_package_local_score_available=False,current_platform_score_user_reported=96.392))
    write_new(root/RUN/'audit.json',dict(status='passed_complete_20_windows_independent_cold_and_scalar',raw_states_replayed=sum(json.loads((root/RUN/f's{s}-f{f}/complete.json').read_text())['raw_states'] for s in SEEDS for f in range(5)),
                                      max_difference=max(v['max_difference'] for v in cold),peak_rss_mib=max(v['peak_rss_mib'] for v in cold),new_fits=0,new_optimizers=0,new_control_inferences=0))
    return decision

def execute(root):
    root=Path(root).resolve();run=root/RUN;check(root,json.loads((run/'freeze.json').read_text()));frozen=json.loads((run/'freeze.json').read_text())
    write_new(run/'controller-started.json',dict(pid=os.getpid(),time=time.time()))
    try:
        for s in SEEDS:
            for f in range(5):
                for cold in [False,True]:
                    check(root,frozen)
                    action='cold' if cold else 'worker'
                    with (run/f's{s}-f{f}-{action}.log').open('x') as log:
                        subprocess.run([sys.executable,'-u',SCRIPT,action,'--seed',str(s),'--fold',str(f)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        decision=evaluate(root,frozen);check(root,frozen)
        write_new(run/'controller-finished.json',dict(status='completed_zero_fit_post_selection_exploration',time=time.time(),decision=decision,new_fits=0,new_optimizers=0,fullfit=0,packages=0,uploads=0))
    except BaseException as e:
        write_new(run/'controller-failed.json',dict(status='failed_evidence_preserved',time=time.time(),error=repr(e),traceback=traceback.format_exc(),new_fits=0,new_optimizers=0));raise

def prepare(root):
    from bf_tap_r2.tabm_swa_run import runtime
    from bf_tap_r2.time_swa_run import BUNDLE
    root=Path(root).resolve();assert_branch(root);run=root/RUN
    if run.exists():raise ValueError('Run directory exists; no overwrite or automatic retry')
    spec=json.loads((root/SPEC).read_text())
    if spec['candidate_pool']!=['PREDICT_MEAN10'] or spec['split_seeds']!=list(SEEDS) or spec['formal_promotion_allowed'] or spec['alpha']!=.2:raise ValueError('Specification changed')
    receipt=json.loads((root/AUX/'tests-complete-r1.json').read_text());sources=source_hashes(root)
    if receipt['source_hashes']!=sources or receipt['status']!='passed':raise ValueError('Tests/source differ')
    verify_tree(root,receipt['logs']);deps={}
    for directory in [root/DEV,root/CONF,root/BUNDLE]:
        for p in directory.rglob('*'):
            if p.is_file():deps[str(p.relative_to(root))]=sha(p)
    for directory in [root/DEV,root/CONF]:
        audit=json.loads((directory/'audit.json').read_text())
        if audit['status']!='passed' or audit['new_fits']!=0:raise ValueError('Original model cold audit missing')
        old=json.loads((directory.parent/'preflight.json').read_text())
        # Freeze all declared ancestor dependencies without rewriting originals.
        prior=old.get('reference',{}).get('frozen_files',{})
        verify_tree(root,prior);deps.update(prior)
        for name in ['preflight.json','freeze.json','ledger/fits.jsonl']:
            p=directory.parent/name
            if p.exists():deps[str(p.relative_to(root))]=sha(p)
    frozen={'source_hashes':sources,'reference_files':deps,'runtime':runtime(),'spec_sha256':sha(root/SPEC),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'tests_receipt_sha256':sha(root/AUX/'tests-complete-r1.json'),'new_fits':0,'new_optimizers':0}
    check(root,frozen);run.mkdir(parents=True,exist_ok=False);write_new(run/'freeze.json',frozen)
    write_new(run/'preflight.json',dict(**frozen,status='engineering_tests_and_original_window_cold_evidence_passed_zero_fit'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','execute','worker','cold']);parser.add_argument('--seed',type=int);parser.add_argument('--fold',type=int);args=parser.parse_args()
    threads();root=Path.cwd()
    if args.action=='prepare':prepare(root)
    elif args.action=='execute':execute(root)
    else:worker(root,args.seed,args.fold,cold=args.action=='cold')
