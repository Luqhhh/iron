"""Frozen nonlinear-family OOF comparison with independent replay."""
from __future__ import annotations

import argparse
import json
import hashlib
import shutil
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .audit import digest,write_json
from .candidate_tiers import classify_candidates
from .cv import target_metrics
from .data import TARGETS,FEATURES
from .low_signal_cv import sensitivity
from .nonlinear_models import NonlinearRegressor
from .v2_refinement import fit_once,fold_vector
from .v2_release import load_v2
from .v2_robust_joint import id_digest
from .v2_smooth_search import aligned,check


def initialize(root,output):
    if not output.is_relative_to(root/'local/runs/round2-v2.7'):
        raise ValueError('Fresh private V2.7 directory required')
    path=root/'configs/round2_v2_7/experiment.yaml'
    spec=yaml.safe_load(path.read_text())
    frozen=check(root,root/spec['source_run'])
    files=dict(frozen['files'])
    extras=[path,Path(__file__),Path(__file__).with_name('nonlinear_models.py'),root/spec['policy'],root/'uv.lock']
    for key in ('reference_run','time_best_run'):
        extras.extend((root/spec[key]/'oof').glob('*.csv'))
        extras.extend([root/spec[key]/'selection/summary.json',root/spec[key]/'selection/independent_verification.json'])
    # Bind all newly queued packages as well as the previously frozen originals.
    for directory in ('round2-v2.4/release-r1','round2-v2.5/release-r1'):
        r=root/'local/runs'/directory
        extras.append(r/'release_manifest.json')
        for name,record in json.loads((r/'release_manifest.json').read_text())['packages'].items():
            extras.extend([r/'release'/name/'result.csv',r/'release'/name/'Luqhhh_bf_tap_predict_round2.zip',Path(record['desktop'])/'Luqhhh_bf_tap_predict_round2.zip'])
    for p in extras:
        files[str(p.relative_to(root)) if p.is_relative_to(root) else str(p)]=digest(p)
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('models','oof','selection'):
        (output/folder).mkdir()
    write_json(output/'manifest.json',{'files':files,'spec':spec,'policy':yaml.safe_load((root/spec['policy']).read_text()),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()})


def compatible_parameters(stored, desired, route, report, allow_old_cap=False):
    if stored == desired:
        return True
    if not allow_old_cap or route != 'KS1':
        return False
    return ({k:v for k,v in stored.items() if k!='max_iter'} == {k:v for k,v in desired.items() if k!='max_iter'}
            and 0 < stored['max_iter'] <= desired['max_iter']
            and report['fit_status']==0 and not report['convergence_warning']
            and report['iterations'] < stored['max_iter'])


def compute(root,output,replay=False):
    frozen=check(root,output)
    spec=frozen['spec']
    train=load_v2(root/'复赛_train','train',2754)
    metrics={t:{r:{} for r in [spec['reference_by_target'][t],spec['best_by_target'][t],*spec['candidates'][t]]} for t in TARGETS}
    predictions={t:{r:{} for r in metrics[t]} for t in TARGETS}
    for split in spec['split_seeds']:
        folds=fold_vector(root/spec['fold_run'],train,split)
        values={t:{r:np.full(len(train),np.nan) for r in spec['models']} for t in TARGETS}
        for fold in range(5):
            training,valid=train.loc[folds!=fold],train.loc[folds==fold]
            for route,target in [('JM1',None),('KR1',None),('KS1',TARGETS[0]),('KS1',TARGETS[1])]:
                path=output/'models'/f'{split}-{fold}-{route}-{target or "joint"}.joblib'
                identity={'route':route,'target':target,'split':split,'heldout_fold':fold,'train_ids':id_digest(training),'valid_ids':id_digest(valid)}
                fields=TARGETS if target is None else (target,)
                desired={**spec['models'][route],**frozen.get('execution_overrides',{}).get(route,{})}
                if replay or (frozen.get('allow_cached_models',False) and path.exists()):
                    model=joblib.load(path)
                    if model.model_identity_!=identity or not compatible_parameters(model.spec,desired,route,model.fit_report_,frozen.get('allow_cached_models',False)) or tuple(model.target_fields_)!=fields:
                        raise ValueError('Model identity/spec mismatch')
                else:
                    model=NonlinearRegressor(route,desired,target)
                    model.model_identity_=identity
                    fit_once(model,training,training[list(TARGETS)] if target is None else training[target],path,output,'cv',identity,frozen.get('fit_attempt_limit',40))
                    write_json(path.with_suffix('.json'),{'model_sha256':digest(path),'parameters':model.actual_parameters_,
                        'fit_report':model.fit_report_,'target_scales':model.scales_.tolist()})
                record=json.loads(path.with_suffix('.json').read_text())
                if digest(path)!=record['model_sha256'] or model.actual_parameters_!=record['parameters'] or model.fit_report_!=record['fit_report']:
                    raise ValueError('Saved fit metadata mismatch')
                np.testing.assert_allclose(model.scales_,training[list(fields)].mean(),rtol=1e-14)
                np.testing.assert_allclose(model.preprocessor_.named_transformers_['numeric'].mean_,training[list(FEATURES)].mean(),rtol=1e-12,atol=1e-10)
                with threadpool_limits(limits=1):
                    pred=model.predict(valid)
                    if replay:
                        np.testing.assert_allclose(pred,model.predict(valid.iloc[::-1])[::-1],rtol=1e-11,atol=1e-9)
                        batches=np.concatenate([model.predict(valid.iloc[i:i+127]) for i in range(0,len(valid),127)])
                        np.testing.assert_allclose(pred,batches,rtol=1e-11,atol=1e-9)
                        np.testing.assert_allclose(pred[:1],model.predict(valid.iloc[:1]),rtol=1e-11,atol=1e-9)
                for i,t in enumerate(fields):
                    values[t][route][folds==fold]=pred[:,i] if target is None else pred
        for target in TARGETS:
            reference,best=spec['reference_by_target'][target],spec['best_by_target'][target]
            previous=aligned(root/spec['reference_run']/'oof'/f'{split}-{target}.csv',train,folds)
            np.testing.assert_allclose(previous[target],train[target],rtol=1e-14)
            values[target][reference]=previous[reference].to_numpy()
            source=previous if target=='tap_iron' else aligned(root/spec['time_best_run']/'oof'/f'{split}-{target}.csv',train,folds)
            values[target][best]=source[best].to_numpy()
            for route in spec['models']:
                values[target]['A'+route]=.5*values[target][reference]+.5*values[target][route]
            saved=train[['sample_id','spout_no',target]].copy()
            saved['fold']=folds
            for route,pred in values[target].items():
                saved[route]=pred
                metrics[target][route][str(split)]=target_metrics(train,target,pred,folds)
                predictions[target][route][split]=pred
            path=output/'oof'/f'{split}-{target}.csv'
            if replay:
                old=aligned(path,train,folds)
                np.testing.assert_allclose(old[saved.columns.drop('sample_id')],saved.drop(columns='sample_id'),rtol=1e-11,atol=1e-9)
            else:
                saved.to_csv(path,index=False,mode='x',float_format='%.17g')
    tiers=classify_candidates(metrics,spec,frozen['policy'])
    rows=[]
    for target in TARGETS:
        best=spec['best_by_target'][target]
        for route,d in tiers['decisions'][target].items():
            rows.append({'target':target,'candidate':route,'mean_wmape':d['mean_wmape'],'delta_current':-d['mean_gain'],
                'delta_best':d['mean_wmape']-np.mean([metrics[target][best][str(s)]['wmape'] for s in spec['split_seeds']]),
                'improved_folds':d['improved_folds'],'worst_spout_delta':d['worst_spout_delta'],'tier':d['tier'],
                'failed_conditions':','.join(d['failed_conditions'])})
    bootstrap={}
    for d in tiers['submission_priority']:
        target,route=d['target'],d['candidate']
        distribution,intervals=sensitivity(train,target,predictions[target][route],predictions[target][spec['reference_by_target'][target]],spec['bootstrap'])
        key=f'{target}-{route}'
        bootstrap[key]={'intervals':intervals,'fraction_negative':float((distribution['mean']<0).mean())}
        path=output/'selection'/f'bootstrap-{key}.csv'
        if replay:
            np.testing.assert_allclose(distribution,pd.read_csv(path),rtol=1e-10,atol=1e-14)
        else:
            distribution.to_csv(path,index=False,mode='x')
    result={'metrics':metrics,'tiers':tiers,'comparisons':rows,'bootstrap':bootstrap,'new_cv_fits':40,
            'new_full_fits':0,'new_packages':0,'reused_development_splits':True,
            'total_fit_attempts':frozen.get('fit_attempt_limit',40),'reused_converged_models':frozen.get('reused_converged_models',0)}
    if replay:
        if result!=json.loads((output/'selection/summary.json').read_text()):
            raise ValueError('Independent summary mismatch')
    else:
        write_json(output/'selection/summary.json',result)
        pd.DataFrame(rows).to_csv(output/'selection/comparison.csv',index=False,mode='x')


def verify(root,output):
    es=[json.loads(s) for s in (output/'fit_ledger.jsonl').read_text().splitlines()]
    starts=[e for e in es if e['event']=='start'];done=[e for e in es if e['event']=='complete']
    if len(starts)!=check(root,output).get('fit_attempt_limit',40) or len(done)!=40 or len({e['model'] for e in done})!=40:
        raise ValueError('Nonlinear fit budget mismatch')
    for e in done:
        if digest(Path(e['model']))!=e['model_sha256']:
            raise ValueError('Fit ledger SHA mismatch')
    compute(root,output,True)
    write_json(output/'selection/independent_verification.json',{'status':'PASS','replayed_models':40,'new_fits':0,
        'training_only_transforms_and_fold_ids_checked':True,'metrics_tiers_bootstrap_reproduced':True})



def initialize_completion(root,source,output):
    private=root/'local/runs/round2-v2.7'
    if not source.is_relative_to(private) or not output.is_relative_to(private) or output.exists():
        raise ValueError('Completion needs a fresh private output')
    original=json.loads((source/'manifest.json').read_text())
    failure=json.loads((source/'FAILED.json').read_text())
    if 'max_iter=100000' not in failure['error'] or 'convergence' not in failure['error']:
        raise ValueError('This completion only handles the recorded SVR iteration cap')
    name=str(Path(__file__).relative_to(root))
    for path,sha in original['files'].items():
        if path!=name and digest(root/path)!=sha:
            raise ValueError(f'Frozen completion input changed: {path}')
    code=subprocess.check_output(['git','show',f"{original['commit']}:{name}"],cwd=root)
    if hashlib.sha256(code).hexdigest()!=original['files'][name]:
        raise ValueError('Original orchestration source not recoverable')
    events=[json.loads(line) for line in (source/'fit_ledger.jsonl').read_text().splitlines()]
    done=[e for e in events if e['event']=='complete'];starts=[e for e in events if e['event']=='start']
    if len(done)!=15 or len(starts)!=16:
        raise ValueError('Unexpected predecessor fit accounting')
    for event in done:
        if digest(Path(event['model']))!=event['model_sha256']:
            raise ValueError('Completed predecessor model changed')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'models',output/'models')
    for folder in ('oof','selection'):
        (output/folder).mkdir()
    shutil.copy2(source/'fit_ledger.jsonl',output/'fit_ledger.jsonl')
    (output/'source_before_completion.py').write_bytes(code)
    original['files'][name]=digest(Path(__file__))
    for path in source.rglob('*'):
        if path.is_file():
            original['files'][str(path.relative_to(root))]=digest(path)
    original.update(execution_overrides={'KS1':{'max_iter':1000000}},allow_cached_models=True,
                    fit_attempt_limit=41,reused_converged_models=15,original_failed_attempts=1,
                    completion_source=str(source.relative_to(root)),
                    completion_reason='SVR solver resource cap only: preserve C/epsilon/gamma/tol and reuse already converged models',
                    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    write_json(output/'manifest.json',original)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['evaluate','verify','complete-fits'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path)
    a=p.parse_args();root=Path.cwd().resolve();output=a.output.resolve()
    if a.action=='verify':
        verify(root,output)
    else:
        if a.action=='complete-fits':
            if a.source is None:
                p.error('complete-fits requires --source')
            initialize_completion(root,a.source.resolve(),output)
        else:
            initialize(root,output)
        try:
            compute(root,output)
            subprocess.run([sys.executable,'-m','bf_tap_r2.v2_nonlinear','verify','--output',str(output)],cwd=root,check=True)
            write_json(output/'COMPLETE.json',{'status':'PASS','completed_cv_models':40,'total_fit_attempts':check(root,output).get('fit_attempt_limit',40),
                'reused_converged_models':check(root,output).get('reused_converged_models',0),'new_full_fits':0,'new_packages':0})
        except Exception as e:
            write_json(output/'FAILED.json',{'error':str(e)})
            raise


if __name__=='__main__':
    main()
