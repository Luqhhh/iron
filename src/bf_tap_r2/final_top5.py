"""User-requested ranked isolated Top 5, preserving historical promotion decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import yaml
from .audit import digest,write_json
from .data import TARGETS
from .normalized_models import JointSnapshotRegressor
from .submission import ZIP_NAME,package,validate_result,deny_training_reads
from .v2_refinement import fit_once,fold_vector,isolated_payload
from .v2_release import load_v2
from .v2_robust_joint import id_digest
from .v2_smooth_search import aligned,check
from .v2_joint_seeds import cv as replay_joint_cv,old_full
from .v2_selected_release import BASE,file_node,checked_prediction,infer

EXISTING={
 'DJ':'local/runs/round2-v2.4/release-r1/release/V24_FORMAL_IRON_DJ',
 'AJ':'local/runs/round2-v2.3/candidate-release-r1/release/V23_AJ_I_ONLY'}
I2='local/runs/round2-v2.2/checkpoint-r1/release/V22_I_ONLY/result.csv'


def summary_path(root,run):
    p=root/run/'selection/summary.json'
    return p if p.exists() else root/run/'summary.json'


def canonical_metrics(metrics):
    if 'tap_iron' in metrics or 'tap_time_len' in metrics:
        return metrics
    return {t:{route:{s:by_split[s][t] for s in ('42','3407')} for route,by_split in metrics.items()} for t in TARGETS}


def ranking(root,spec):
    anchor=canonical_metrics(json.loads(summary_path(root,spec['anchor_oof_run']).read_text())['metrics'])
    anchor_values={t:np.array([anchor[t][spec['anchor_reference'][t]][str(s)]['wmape'] for s in spec['split_seeds']]) for t in TARGETS}
    seen={};rows=[]
    for run in spec['ranked_runs']:
        source=summary_path(root,run)
        m=canonical_metrics(json.loads(source.read_text())['metrics'])
        for target,routes in m.items():
            for route,by_split in routes.items():
                if route in spec['excluded']:
                    continue
                values=np.array([by_split[str(s)]['wmape'] for s in spec['split_seeds']])
                if not np.isfinite(values).all() or (values<0).any():
                    raise ValueError('Invalid catalog metrics')
                key=(target,route)
                if key in seen:
                    np.testing.assert_allclose(seen[key],values,rtol=1e-12,atol=1e-14)
                    continue
                seen[key]=values
                other=next(t for t in TARGETS if t!=target)
                scores=100*(1-(values+anchor_values[other])/2)
                rows.append({'target':target,'candidate':route,'source_run':run,'wmape_by_split':values.tolist(),
                    'local_score_by_split':scores.tolist(),'local_score':float(scores.mean())})
    return sorted(rows,key=lambda r:(-r['local_score'],r['target'],r['candidate']))


def mean_node(members):
    return {'kind':'mean','weights':[1/len(members)]*len(members),'members':members,
            'aggregation':'arithmetic_mean' if len(members)==3 else 'weighted_sum'}


def recipes(anchor,i2,joint_members,old_dj,old_aj):
    j3=mean_node(joint_members);aj3=mean_node([anchor,j3])
    return {'DJ':old_dj,'J3':j3,'DJ3':mean_node([i2,aj3]),'AJ3':aj3,'AJ':old_aj}


def initialize(root,out):
    cfg=root/'configs/round2_final_top5/release.yaml';spec=yaml.safe_load(cfg.read_text())
    if not out.is_relative_to(root/'local/runs/round2-final-top5') or out.exists() or Path(spec['desktop']).exists():
        raise ValueError('Fresh local and desktop final Top 5 destinations required')
    frozen=check(root,root/spec['source_run']);files=dict(frozen['files'])
    rows=ranking(root,spec);top=rows[:5]
    if [r['candidate'] for r in top]!=spec['top5_expected'] or any(r['target']!='tap_iron' for r in top):
        raise ValueError('Top 5 changed: inspect ranking before release')
    extras=[cfg,Path(__file__),root/'src/bf_tap_r2/v2_selected_release.py',root/spec['joint_config']]
    for run in spec['ranked_runs']:
        summary=summary_path(root,run);verification=summary.parent/'independent_verification.json'
        if json.loads(verification.read_text())['status']!='PASS':
            raise ValueError('Unverified ranking source')
        extras.extend([summary,verification])
    for r in top:
        extras.extend(root/r['source_run']/'oof'/f'{s}-tap_iron.csv' for s in spec['split_seeds'])
    for folder in EXISTING.values():
        extras.extend([root/folder/'result.csv',root/folder/ZIP_NAME])
    joint=root/spec['joint_run']
    for p in (joint/'models').glob('*'):
        if p.is_file():extras.append(p)
    extras.extend([joint/'manifest.json',joint/'fit_ledger.jsonl'])
    for p in extras:
        files[str(p.relative_to(root))]=digest(p)
    out.mkdir(parents=True,exist_ok=False)
    for name in ('models','release'):(out/name).mkdir()
    write_json(out/'manifest.json',{'files':files,'spec':spec,'ranking':rows,'top5':top,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    return spec


def preflight(root,out):
    frozen=check(root,out);spec=frozen['spec']
    if ranking(root,spec)!=frozen['ranking']:
        raise ValueError('Ranking replay mismatch')
    train=load_v2(root/'复赛_train','train',2754)
    for row in frozen['top5']:
        for i,split in enumerate(spec['split_seeds']):
            folds=fold_vector(root/spec['fold_run'],train,split)
            old=aligned(root/row['source_run']/'oof'/f'{split}-tap_iron.csv',train,folds)
            np.testing.assert_allclose(old.tap_iron,train.tap_iron,rtol=1e-14)
            metric=float(np.abs(old[row['candidate']].to_numpy()-train.tap_iron.to_numpy()).sum()/train.tap_iron.sum())
            np.testing.assert_allclose(metric,row['wmape_by_split'][i],rtol=1e-12)
    # Read-only replay: do not call old verify(), which writes into the old run.
    replay_joint_cv(root,root/spec['joint_run'],True)
    base=root/spec['base_csv']
    if digest(base.parent/ZIP_NAME)!=spec['base_zip_sha256']:
        raise ValueError('B anchor ZIP changed')
    return frozen,train


def generate(root,out):
    frozen,train=preflight(root,out);spec=frozen['spec']
    joint_spec=yaml.safe_load((root/spec['joint_config']).read_text())
    old_path=old_full(root,joint_spec)
    old_model=joblib.load(old_path)
    np.testing.assert_allclose(old_model.target_scales_,train[list(TARGETS)].mean(),rtol=1e-14)
    if old_model.parameters!={**joint_spec['common'],'random_seed':42}:
        raise ValueError('Existing full joint configuration mismatch')
    members=[file_node(root,old_path,'tap_iron','joint_iron')]
    for seed in spec['model_seeds'][1:]:
        model=JointSnapshotRegressor({**joint_spec['common'],'random_seed':seed})
        model.model_identity_={'stage':'full','route':'J1','model_seed':seed,'train_ids':id_digest(train),'target_fields':list(TARGETS)}
        path=out/'models'/f'full-J1-{seed}.joblib'
        fit_once(model,train,train[list(TARGETS)],path,out,'full',model.model_identity_,2)
        write_json(path.with_suffix('.json'),{'model_sha256':digest(path),'identity':model.model_identity_,
            'actual_parameters':model.actual_parameters_,'scales':model.target_scales_.tolist()})
        members.append(file_node(root,path,'tap_iron','joint_iron'))
    base=root/BASE;anchor=file_node(root,base,'tap_iron');i2=file_node(root,root/I2,'tap_iron')
    dj_manifest=json.loads((root/'local/runs/round2-v2.4/release-r1/release_manifest.json').read_text())
    dj=dj_manifest['packages']['V24_FORMAL_IRON_DJ']['node']
    aj=mean_node([anchor,members[0]])
    nodes=recipes(anchor,i2,members,dj,aj)
    test=load_v2(root/'复赛_test','test',322);base_bytes=base.read_bytes();base_rows=validate_result(base_bytes,test.sample_id)
    records={};seen=set()
    for rank,row in enumerate(frozen['top5'],1):
        route=row['candidate'];name=f'{rank:02d}_{route}_IRON';folder=out/'release'/name;folder.mkdir()
        pred=checked_prediction(root,nodes[route],test)
        payload=isolated_payload(base_bytes,test.sample_id,'tap_iron',pred)
        if route in EXISTING:
            original=root/EXISTING[route]
            if payload!=(original/'result.csv').read_bytes():
                raise ValueError('Existing package cold reconstruction changed')
            for filename in ('result.csv',ZIP_NAME):shutil.copy2(original/filename,folder/filename)
        else:
            package(folder,payload,test.sample_id)
        sha=hashlib.sha256(payload).hexdigest()
        if sha in seen:raise ValueError('Duplicate Top 5 payloads')
        seen.add(sha)
        current=validate_result(payload,test.sample_id)
        changed={'pred_'+t:sum(a['pred_'+t]!=b['pred_'+t] for a,b in zip(base_rows,current)) for t in TARGETS}
        if changed['pred_tap_time_len']!=0:raise ValueError('Anchor time strings changed')
        records[name]={**row,'target':'tap_iron','rank':rank,'node':nodes[route],'changed_rows':changed,
            'zip_sha256':digest(folder/ZIP_NAME),'csv_sha256':sha,'copied_original':route in EXISTING,
            'desktop':str(Path(spec['desktop'])/name)}
    write_json(out/'release_manifest.json',{'base_sha256':digest(base),'base_zip_sha256':spec['base_zip_sha256'],
        'packages':records,'new_cv_fits':0,'new_full_fits':2,'member_seeds':spec['model_seeds'],'historical_v26_decision_preserved':True})
    subprocess.run([sys.executable,'-m','bf_tap_r2.final_top5','cold','--output',str(out)],check=True)
    verify_release(root,out)
    check(root,out)
    desktop=Path(spec['desktop']);desktop.mkdir(parents=True,exist_ok=False)
    for name,record in records.items():
        dest=desktop/name;dest.mkdir()
        for filename in ('result.csv',ZIP_NAME):shutil.copy2(out/'release'/name/filename,dest/filename)
        if digest(dest/ZIP_NAME)!=record['zip_sha256']:raise ValueError('Desktop SHA mismatch')
    note=['# Final Top 5','', '按已完成的单目标替换方案本地OOF排序；不是平台预测。全部保持B包的完整B3时长字符串。','', '|排名|方案|本地分数|ZIP SHA-256|','|---|---|---:|---|']
    for rec in records.values():note.append(f"|{rec['rank']}|{rec['candidate']}|{rec['local_score']:.6f}|{rec['zip_sha256']}|")
    note+=['','每个子目录的Luqhhh_bf_tap_predict_round2.zip单独上传，不要将整个目录打包上传。旧包和旧排程作为历史保留，本批Top5是用户要求的独立交付。DJ和AJ为原ZIP逐字节复制；三个新包仅共用新增的两个联合全量模型。','这些方案高度相关，名次间差值极小，不代表平台次序或保证提分。用户自行上传并回传分数。']
    (desktop/'README.md').write_text('\n'.join(note)+'\n')
    shutil.copy2(out/'release_manifest.json',desktop/'manifest.json')
    write_json(out/'COMPLETE.json',{'status':'PASS','packages':list(records),'new_cv_fits':0,'new_full_fits':2,
        'new_unique_packages':3,'reused_original_packages':2,'cold_byte_identical':True,'desktop':str(desktop),'agent_uploads':0})


def verify_release(root,out):
    m=json.loads((out/'release_manifest.json').read_text());train=load_v2(root/'复赛_train','train',2754)
    es=[json.loads(s) for s in (out/'fit_ledger.jsonl').read_text().splitlines()]
    starts=[e for e in es if e['event']=='start'];done=[e for e in es if e['event']=='complete']
    if len(starts)!=2 or len(done)!=2 or any(e['stage']!='full' for e in es):raise ValueError('Full-fit ledger mismatch')
    for e in done:
        path=Path(e['model']);a=joblib.load(path);record=json.loads(path.with_suffix('.json').read_text())
        spec=check(root,out)['spec']
        expected=yaml.safe_load((root/spec['joint_config']).read_text())['common']
        seed=record['identity']['model_seed']
        if seed not in (2026,2027) or a.parameters!={**expected,'random_seed':seed} or a.model_identity_['train_ids']!=id_digest(train) or tuple(a.target_fields_)!=TARGETS or a.estimator_.tree_count_!=1500:
            raise ValueError('Unexpected full model training contract')
        if digest(path)!=e['model_sha256'] or digest(path)!=record['model_sha256'] or a.model_identity_!=record['identity'] or a.estimator_.get_all_params()!=record['actual_parameters']:
            raise ValueError('Full joint model identity mismatch')
        np.testing.assert_allclose(a.target_scales_,train[list(TARGETS)].mean(),rtol=1e-14)
    test=load_v2(root/'复赛_test','test',322)
    for name,rec in m['packages'].items():
        folder=out/'release'/name
        if (folder/'cold.csv').read_bytes()!=(folder/'result.csv').read_bytes() or digest(folder/ZIP_NAME)!=rec['zip_sha256']:
            raise ValueError('Cold release identity mismatch')
        payload=(folder/'result.csv').read_bytes()
        validate_result(payload,test.sample_id)
        with zipfile.ZipFile(folder/ZIP_NAME) as archive:
            if archive.namelist()!=['result.csv'] or archive.read('result.csv')!=payload:
                raise ValueError('ZIP root/content mismatch')
    write_json(out/'verification.json',{'status':'PASS','cold_process_training_reads_forbidden':True,
        'warm_model_verification_reads_only_V2_training_labels':True,'new_fits':0,'packages':5})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['rank','release','cold']);p.add_argument('--output',type=Path)
    a=p.parse_args();root=Path.cwd().resolve()
    if a.action=='rank':
        spec=yaml.safe_load((root/'configs/round2_final_top5/release.yaml').read_text())
        print(json.dumps(ranking(root,spec)[:10],ensure_ascii=False,indent=2));return
    if a.output is None:p.error('--output required')
    out=a.output.resolve()
    if a.action=='cold':infer(root,out);return
    initialize(root,out)
    try:generate(root,out)
    except Exception as e:
        write_json(out/'FAILED.json',{'error':str(e)});raise


if __name__=='__main__':main()
