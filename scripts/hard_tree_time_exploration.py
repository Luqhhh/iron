"""Single manual hard-tree time release; original V39 science stays read-only."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/lux1/iron-v39-hard-tree-quality')
RUN=ROOT/'local/runs/hard-tree-time-exploration-20261003/release-r1'
OLD=SOURCE/'local/runs/round2-v39/development-r1'
REVIEW=ROOT/'local/runs/q75-hard-tree-reuse-20261003/review-r1'
REF=ROOT/'local/runs/modernnca-q75-readiness-20261001/preparation-r1/reference-r2'
NAME='HARDTREE_GLOBAL_TIME_A20'
PROTOCOL=ROOT/'docs/hard_tree_time_exploration/PREREGISTRATION.md'
helper_spec=importlib.util.spec_from_file_location('release_io',ROOT/'scripts/modernnca_time_exploration.py')
helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
sha,read,write,payload,zip_bytes=helper.sha,helper.read,helper.write,helper.payload,helper.zip_bytes


def setup():
    if sys.version_info[:2]!=(3,12) or any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise ValueError('Locked Python3.12 and pre-import thread pinning required')
    sys.path.insert(0,str(SOURCE/'src'))
    import torch,numpy,sklearn
    if (torch.__version__,numpy.__version__,sklearn.__version__)!=('2.14.0+cpu','2.2.6','1.8.0'):
        raise ValueError('Frozen environment differs')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def verify(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('Frozen dependency differs: '+p)


def memory():
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>1536:raise ValueError('RSS gate exceeded')
    return peak


def frames():
    import numpy as np,pandas as pd
    from bf_tap_r2.data import FEATURES,TARGETS
    from bf_tap_r2.v2_release import load_v2
    ref=read(REF/'q75-reference-receipt.json')
    with np.load(REF/ref['reference_file'],allow_pickle=False) as a:
        frame=pd.DataFrame(a['numeric'],columns=FEATURES)
        frame['sample_id'],frame['spout_no']=a['ids'],a['spout']
        for j,t in enumerate(TARGETS):frame[t]=a['targets'][:,j]
        frame=frame.loc[:,ref['frame_columns']].astype(ref['frame_dtypes'])
    return frame,load_v2(ROOT/'复赛_test','test',322)


def parts(frame):
    from bf_tap_r2.v3_4_bags import group_safe_inner_folds
    info=group_safe_inner_folds(frame,seed=27001,n_splits=5)
    fitting=frame.loc[info['fold']!=0].reset_index(drop=True)
    calibration=frame.loc[info['fold']==0].reset_index(drop=True)
    if set(fitting.sample_id)&set(calibration.sample_id):raise ValueError('Inner IDs overlap')
    return fitting,calibration,info


def freeze(checks):
    import numpy as np,yaml,csv
    from bf_tap_r2.data import FEATURES,TARGETS
    if RUN.exists():raise FileExistsError('Release output already exists')
    best=read(ROOT/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if (best['candidate'],best['score'],best['zip_sha256'])!=('EMA_TIME_Q75',96.392,helper.PARENT_SHA):
        raise ValueError('Current reference changed')
    report=read(REVIEW/'report.json');audit=read(REVIEW/'independent-audit.json');review=read(REVIEW/'manifest.json')
    if audit['status']!='passed' or audit['report_sha256']!=sha(REVIEW/'report.json'):
        raise ValueError('Complete Q75 review required')
    verify(review['files'])
    tests=read(checks)
    if tests['exit_code']!=0 or tests['python']!='3.12' or tests['runner_sha256']!=sha(__file__) or sha(tests['junit'])!=tests['junit_sha256']:
        raise ValueError('Passing current locked checks required')
    files=dict(review['files'])
    for p in list((SOURCE/'src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/modernnca_time_exploration.py',
            PROTOCOL,ROOT/'uv.lock',ROOT/'pyproject.toml',Path(checks),Path(tests['junit']),
            REVIEW/'report.json',REVIEW/'independent-audit.json',ROOT/'configs/protection.yaml',
            ROOT/'local/authorizations/optimization-standing-20261002-r1.json']:
        files[str(p)]=sha(p)
    old=read(OLD/'manifest.json')
    for name in ['v39_regressor.py','v37_hard_tree.py','v36_hard_tree.py']:
        rel='src/bf_tap_r2/'+name
        if files[str(SOURCE/rel)]!=old['source_hashes'][rel]:raise ValueError('Native scientific source changed')
    parent=ROOT/'local/runs/ema-time-followup-20261001/probes-r2/EMA_TIME_Q75/Luqhhh_bf_tap_predict_round2.zip'
    if sha(parent)!=helper.PARENT_SHA:raise ValueError('Parent differs')
    files[str(parent)]=helper.PARENT_SHA
    for p in [ROOT/'复赛_test/result_template.csv',ROOT/'复赛_test/test_samples.csv',ROOT/'复赛_test/test_features.csv',SOURCE/'configs/round2_v39/SPEC.yaml']:
        files[str(p)]=sha(p)
    verify(files);RUN.mkdir(parents=True,exist_ok=False);write(RUN/'frozen-files.json',files)
    write(RUN/'access-before-round2-read.json',dict(scope='authorized_round2_only',protected_preliminary_targets=False,frozen_files_sha256=sha(RUN/'frozen-files.json')))
    frame,query=frames();fitting,calibration,info=parts(frame)
    if len(frame)!=2754 or len(query)!=322 or set(frame.sample_id)&set(query.sample_id) or any(t in query for t in TARGETS):raise ValueError('Bad train/query schema')
    template=list(csv.DictReader((ROOT/'复赛_test/result_template.csv').open()))
    if query.sample_id.tolist()!=[r['sample_id'] for r in template]:raise ValueError('Template mismatch')
    with (RUN/'query.npz').open('xb') as f:np.savez(f,numeric=query[list(FEATURES)].to_numpy(float),ids=query.sample_id.to_numpy(str),spout=query.spout_no.to_numpy())
    with (RUN/'parent.csv').open('xb') as f:f.write(zip_bytes(parent))
    for n in ['query.npz','parent.csv']:files[str(RUN/n)]=sha(RUN/n)
    settings=yaml.safe_load((SOURCE/'configs/round2_v39/SPEC.yaml').read_text())['training']
    write(RUN/'manifest.json',dict(candidate=NAME,files=files,settings=settings,recipe='GLOBAL',weight=.2,
          identity=dict(source_directory=str(RUN),split_seed=-1,trial_id=NAME),reference=best['candidate'],
          reference_score_user_reported=best['score'],formal_promoted=False,inner_fold_hash=info['inner_fold_hash'],group_hash=info['group_hash'],
          rows=dict(training=len(frame),fitting=len(fitting),calibration=len(calibration),query=len(query)),
          runner_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          budget=dict(estimator=2,optimizer=2,packages=1,cv=0,confirmation_seeds=0,desktop_writes=0,agent_uploads=0)))
    print(json.dumps(dict(status='frozen',files=len(files),rows=read(RUN/'manifest.json')['rows'])))


def fit():
    import numpy as np,torch
    from bf_tap_r2.v39_regressor import TreeRegressor
    from bf_tap_r2.data import FEATURES
    m=read(RUN/'manifest.json');verify(m['files']);frame,query=frames();fitting,calibration,info=parts(frame)
    if info['inner_fold_hash']!=m['inner_fold_hash']:raise ValueError('Inner partition changed')
    columns=['sample_id','spout_no',*FEATURES];real_adam=torch.optim.Adam;constructors=[];step_counts={};current_role=None
    def counted(*a,**kw):
        if len(constructors)>=2:raise ValueError('Optimizer budget exceeded')
        role=current_role;constructors.append(role);step_counts[role]=0
        write(RUN/(role+'-optimizer-start.json'),dict(identity=m['identity'],role=role,time_ns=time.time_ns()))
        opt=real_adam(*a,**kw);real_step=opt.step
        def step(*a,**kw):
            result=real_step(*a,**kw);step_counts[role]+=1;return result
        opt.step=step
        return opt
    torch.optim.Adam=counted
    artifacts={};metadata={};predictions={};selected=None
    for role,part in [('selector',fitting),('refit',frame)]:
        current_role=role
        write(RUN/(role+'-estimator-start.json'),dict(identity=m['identity'],role=role,fit_ids=part.sample_id.tolist(),time_ns=time.time_ns()))
        model=TreeRegressor('GLOBAL',m['settings']).initialize(part[columns],part.tap_time_len.to_numpy())
        epoch=model.train(m['settings']['max_epochs'] if role=='selector' else selected,
              (calibration[columns],calibration.tap_time_len.to_numpy()) if role=='selector' else None)
        model.save(RUN/(role+'.pt'));artifacts[role]=sha(RUN/(role+'.pt'));metadata[role]=model.metadata()
        predictions[role]=model.predict(query[columns])
        if role=='selector':selected=epoch;predictions['calibration']=model.predict(calibration[columns])
        expected_steps=model.stopped_epoch_*math.ceil(len(part)/m['settings']['batch_size'])
        if step_counts[role]!=expected_steps:raise ValueError('Actual optimizer steps differ')
        memory()
        write(RUN/(role+'-complete.json'),dict(model_sha256=artifacts[role],metadata=metadata[role],optimizer_steps=step_counts[role],time_ns=time.time_ns()))
    if constructors!=['selector','refit']:raise ValueError('Constructor inventory differs')
    with (RUN/'warm-predictions.npz').open('xb') as f:np.savez(f,query_ids=query.sample_id.to_numpy(str),**predictions)
    verify(m['files']);write(RUN/'warm.json',dict(artifacts=artifacts,metadata=metadata,constructors=constructors,steps=step_counts,
          prediction_sha256=sha(RUN/'warm-predictions.npz'),manifest_sha256=sha(RUN/'manifest.json'),peak_rss_mib=memory()))
    print(json.dumps(dict(status='warm_complete',selected_epoch=selected,peak_rss_mib=memory())))


def forbidden(*a,**kw):raise RuntimeError('Cold audit attempted scientific fitting')


def cold():
    import numpy as np,torch
    from bf_tap_r2.data import FEATURES
    from bf_tap_r2.v7_periodic import digest
    from bf_tap_r2.v39_regressor import TreeRegressor,TreePreprocessor
    m=read(RUN/'manifest.json');verify(m['files']);w=read(RUN/'warm.json')
    if w['manifest_sha256']!=sha(RUN/'manifest.json') or w['prediction_sha256']!=sha(RUN/'warm-predictions.npz'):raise ValueError('Warm identity differs')
    TreeRegressor.initialize=TreeRegressor.train=TreePreprocessor.fit=forbidden;torch.optim.Adam=forbidden
    frame,query=frames();fitting,calibration,_=parts(frame);maximum=0.;rows={}
    with np.load(RUN/'warm-predictions.npz',allow_pickle=False) as p:
        if p['query_ids'].tolist()!=query.sample_id.tolist():raise ValueError('Warm query IDs changed')
        for role,part in [('selector',fitting),('refit',frame)]:
            path=RUN/(role+'.pt')
            if sha(path)!=w['artifacts'][role]:raise ValueError('Saved state differs')
            model=TreeRegressor.load(path);meta=w['metadata'][role];y=part.tap_time_len.to_numpy(float)
            if model.recipe!='GLOBAL' or model.settings!=m['settings']:raise ValueError('Recipe changed')
            pp=model.preprocessor_;raw=part[list(FEATURES)].to_numpy(float);references=np.linspace(0,1,min(1000,len(part)))
            quantiles=np.maximum.accumulate(np.nanpercentile(raw,references*100,axis=0),axis=0)
            if (not np.array_equal(quantiles,pp.transformer_.quantiles_) or not np.array_equal(references,pp.transformer_.references_)
                    or pp.categories_!=sorted(int(x) for x in part.spout_no.unique()) or pp.fit_ids_digest_!=digest(part.sample_id.tolist())
                    or model.mean_!=float(y.mean()) or model.std_!=float(y.std())):raise ValueError('Partition preprocessing/target scale differs')
            if not all(torch.isfinite(v).all() for v in model.model_.state_dict().values()):raise ValueError('Nonfinite state')
            history=meta['history'];selected=meta['selected_epoch'];stopped=meta['stopped_epoch']
            if len(history)!=stopped or [r['epoch'] for r in history]!=list(range(1,stopped+1)):raise ValueError('Epoch history not contiguous')
            if role=='selector':
                best=float('inf');best_epoch=0;stale=0
                for r in history:
                    loss=r['calibration_standardized_mae']
                    if not math.isfinite(loss):raise ValueError('Nonfinite calibration')
                    if loss<best-m['settings']['min_delta']:best=loss;best_epoch=r['epoch'];stale=0
                    else:stale+=1
                    if r['epoch']<stopped and stale>=m['settings']['patience']:raise ValueError('Training passed stopping epoch')
                if selected!=best_epoch or (stopped<m['settings']['max_epochs'] and stale<m['settings']['patience']):raise ValueError('Wrong selected/stopped epoch')
                cp=model.predict(calibration);maximum=max(maximum,float(np.max(np.abs(cp-p['calibration']))))
                if abs(float(np.abs(cp-calibration.tap_time_len.to_numpy()).mean()/model.std_)-best)>1e-6:raise ValueError('Checkpoint does not match selected calibration epoch')
            elif selected!=stopped or selected!=w['metadata']['selector']['selected_epoch'] or any('calibration_standardized_mae' in r for r in history):raise ValueError('Refit history differs')
            for prediction in [model.predict(query),model.predict(query.iloc[::-1])[::-1],np.concatenate([model.predict(query.iloc[i:i+37]) for i in range(0,len(query),37)])]:maximum=max(maximum,float(np.max(np.abs(prediction-p[role]))))
            if w['steps'][role]!=stopped*math.ceil(len(part)/m['settings']['batch_size']):raise ValueError('Optimizer count differs')
            rows[role]=dict(selected_epoch=selected,stopped_epoch=stopped,fit_rows=len(part),optimizer_steps=w['steps'][role])
    if maximum>1e-6:raise ValueError('Cold prediction differs')
    write(RUN/'cold.json',dict(status='passed',maximum_difference=maximum,models=rows,warm_sha256=sha(RUN/'warm.json'),peak_rss_mib=memory()))
    print(json.dumps(dict(status='cold_passed',maximum_difference=maximum)))


def package_cold():
    import numpy as np,pandas as pd,torch
    from bf_tap_r2.data import FEATURES
    from bf_tap_r2.v39_regressor import TreeRegressor,TreePreprocessor
    from bf_tap_r2.submission import package,validate_result,deny_training_reads,ZIP_NAME
    TreeRegressor.initialize=TreeRegressor.train=TreePreprocessor.fit=forbidden;torch.optim.Adam=forbidden
    sys.addaudithook(deny_training_reads)
    m=read(RUN/'manifest.json');w=read(RUN/'warm.json');c=read(RUN/'cold.json')
    if c['status']!='passed' or c['warm_sha256']!=sha(RUN/'warm.json') or w['manifest_sha256']!=sha(RUN/'manifest.json'):raise ValueError('Closed cold audit required')
    verify({p:h for p,h in m['files'].items() if Path(p).is_relative_to(SOURCE/'src') or Path(p) in [Path(__file__),PROTOCOL,ROOT/'scripts/modernnca_time_exploration.py',RUN/'query.npz',RUN/'parent.csv']})
    if sha(RUN/'refit.pt')!=w['artifacts']['refit'] or sha(RUN/'warm-predictions.npz')!=w['prediction_sha256']:raise ValueError('Warm state identity changed')
    with np.load(RUN/'query.npz',allow_pickle=False) as a:
        query=pd.DataFrame(a['numeric'],columns=FEATURES);query['sample_id']=a['ids'];query['spout_no']=a['spout']
    member=TreeRegressor.load(RUN/'refit.pt').predict(query)
    with np.load(RUN/'warm-predictions.npz',allow_pickle=False) as p:
        maximum=float(np.max(np.abs(member-p['refit'])))
        if maximum>1e-6 or p['query_ids'].tolist()!=query.sample_id.tolist():raise ValueError('Label-free inference differs')
        content=payload((RUN/'parent.csv').read_bytes(),query.sample_id,p['refit'])
    rows=validate_result(content,query.sample_id);parent=validate_result((RUN/'parent.csv').read_bytes(),query.sample_id)
    arithmetic=max(abs(float(r['pred_tap_time_len'])-(.8*float(o['pred_tap_time_len'])+.2*float(v))) for r,o,v in zip(rows,parent,member))
    if arithmetic>1e-6 or any(r['pred_tap_iron']!=o['pred_tap_iron'] for r,o in zip(rows,parent)):raise ValueError('Package arithmetic/iron fields differ')
    out=RUN/NAME;out.mkdir(exist_ok=False);package(out,content,query.sample_id)
    if zip_bytes(out/ZIP_NAME)!=content:raise ValueError('Readback differs')
    write(RUN/'package-audit.json',dict(status='passed',candidate=NAME,package=str(out/ZIP_NAME),zip_sha256=sha(out/ZIP_NAME),csv_sha256=sha(out/'result.csv'),
          rows=322,unique_ids=322,template_order=True,members=['result.csv'],CRC=True,finite_nonnegative=True,iron_string_mismatches=0,
          maximum_cold_difference=maximum,maximum_arithmetic_difference=arithmetic,cold_sha256=sha(RUN/'cold.json'),peak_rss_mib=memory(),
          G1='manual_exploration_two_development_gains_negative_not_formally_promoted_platform_unmeasured'))
    print(json.dumps(dict(status='package_passed',zip_sha256=sha(out/ZIP_NAME))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','fit','cold','package']);p.add_argument('--checks');a=p.parse_args();setup()
    if a.stage=='freeze':freeze(a.checks)
    else:{'fit':fit,'cold':cold,'package':package_cold}[a.stage]()
