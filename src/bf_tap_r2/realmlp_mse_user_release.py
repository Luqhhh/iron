"""Explicit user exploration release; original quality-gated publisher unchanged."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import realmlp_time_release as native
from .data import TARGETS
from .ema_nested_residual import read,write,sha,verify
from .ema_silu_release import package_rows
from .realmlp_time_development import setup

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
SPEC='configs/realmlp_mse_user_release/SPEC.json'
CONF=native.CONF
DEV=native.DEV
SCOPE=dict(candidate='REALMLP_SINGLE_A20',display_name='RealMLP MSE A20',full_official_procedures=1,native_adam=2,native_states=2,
    new_cv=0,new_confirmation_seeds=0,packages=1,desktop_copies=1,agent_uploads=0,blend_weight=.2,formal_promoted=False)


def sources():
    return {str(p.relative_to(WORK)):sha(p) for p in [*list((WORK/'src').rglob('*.py')),WORK/SPEC,
        WORK/'docs/realmlp_mse_user_release/PREREGISTRATION.md',WORK/'tests/test_realmlp_mse_user_release.py',WORK/'uv.lock',WORK/'pyproject.toml']}


def require_scope(spec,grant):
    expected=dict(SCOPE,training_seed=42,inner_seed=42,release_kind='explicit_user_platform_exploration',workers=1,numerical_threads=1,
        max_rss_mib=1536,minimum_available_mib=3072,monitor_seconds=600,time_budget_seconds=None,automatic_retries=False,
        cold_full_batch_atol=0.,cold_order_chunk_atol=.0005,scalar_atol=1e-10,original_conditional_release_not_started=True)
    if any(spec.get(k)!=v or type(spec.get(k)) is not type(v) for k,v in expected.items()):raise ValueError('Fixed user exploration scope changed')
    if (grant.get('source')!='explicit_user_task' or grant.get('user_authorized') is not True
            or grant.get('user_request')!='- **RealMLP MSE A20** 写桌面' or grant.get('scope')!=SCOPE
            or grant.get('desktop_directory')!=spec['desktop_directory'] or grant.get('historical_quality_failure_preserved') is not True):
        raise ValueError('Exact one-package/desktop user authority required')


def authority(spec):
    if sha(spec['explicit_grant'])!=spec['explicit_grant_sha256']:raise ValueError('Explicit grant changed')
    require_scope(spec,read(spec['explicit_grant']))


def require_closed_history(report,audit,terminal):
    if (report['candidate']!=SCOPE['candidate'] or set(report['gains'])!={'42','3407','271828','314159'}
            or not all(math.isfinite(x) for x in report['gains'].values()) or report['gains']['271828']>=0
            or report['paired']['passed'] is not False or audit['status']!='passed' or audit['four_seed_gate_passed'] is not False
            or terminal['status']!='passed' or terminal['actual_supervisor_exit_code']!=0 or terminal['four_seed_gate_passed'] is not False):
        raise ValueError('Closed original evidence and failed quality decision must be preserved')


def prepare(checks):
    from .v5_library import load_v5_training_frame
    from .v2_release import load_v2
    original=setup();spec=read(WORK/SPEC);authority(spec);run=Path(spec['run_directory']);c=read(checks)
    if run.exists():raise FileExistsError('Consumed user release directory')
    if Path(spec['desktop_directory']).exists():raise FileExistsError('Desktop destination already exists')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():raise ValueError('Committed isolated source required')
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['sources']!=sources() or c['junit_sha256']!=sha(c['junit']):raise ValueError('Exact locked source checks required')
    r=read(CONF/'report.json');a=read(CONF/'independent-audit.json');t=read(CONF/'terminal-reconciliation.json');old=read(CONF/'manifest.json')
    require_closed_history(r,a,t)
    if (sha(CONF/'terminal-reconciliation.json')!=spec['confirmation_terminal_sha256'] or t['independent_audit_sha256']!=sha(CONF/'independent-audit.json')
            or a['report_sha256']!=t['report_sha256'] or t['report_sha256']!=sha(CONF/'report.json') or a['manifest_sha256']!=sha(CONF/'manifest.json')):
        raise ValueError('Closed historical receipt chain changed')
    best=read(MAIN/'EVIDENCE_STATUS.json')['round2_current_platform_best']
    if any(best.get(k)!=v for k,v in spec['reference'].items()):raise ValueError('Latest parent changed')
    if old['realmlp_recipe']!=original['recipes']['realmlp_td']:raise ValueError('Original default MSE recipe changed')
    for name in ['v9_realmlp.py','realmlp_state_adapter.py','realmlp_native_audit.py']:
        if sha(WORK/'src/bf_tap_r2'/name)!=sha(Path(old['source_directory'])/'src/bf_tap_r2'/name):raise ValueError('Admitted native source changed')
    if (MAIN/'local/runs/realmlp-time-release-20261004/release-r1').exists():raise ValueError('Original conditional release unexpectedly started; inspect before any fit')
    files=dict(old['files']);paths=[WORK/p for p in sources()]+[Path(checks),Path(c['junit']),Path(spec['explicit_grant']),MAIN/best['package'],MAIN/best['platform_feedback_record']]
    paths += [p for p in CONF.rglob('*') if p.is_file()]
    paths += [MAIN/f'复赛_{stage}/{stage}_{kind}.csv' for stage in ['train','test'] for kind in ['samples','features']]+[MAIN/'复赛_test/result_template.csv']
    paths += [DEV/f's{seed}-f{fold}-i42/predictions.npz' for seed in [42,3407] for fold in range(5)]
    for p in paths:
        p=p.resolve();h=sha(p)
        if str(p) in files and files[str(p)]!=h:raise ValueError('Frozen dependency changed')
        files[str(p)]=h
    verify(files);frame=load_v5_training_frame(MAIN);query=load_v2(MAIN/'复赛_test','test',322)
    ids=pd.read_csv(MAIN/'复赛_test/result_template.csv',dtype={'sample_id':str}).sample_id.tolist()
    if len(frame)!=2754 or query.sample_id.tolist()!=ids or any(t in query for t in TARGETS):raise ValueError('Official data identity differs')
    pd.testing.assert_frame_equal(frame,pd.read_pickle(CONF/'training.pkl'))
    package_rows(MAIN/best['package'],ids)
    if sha(MAIN/best['package'])!=spec['reference']['zip_sha256']:raise ValueError('Parent package identity differs')
    run.mkdir();frame.to_pickle(run/'training.pkl');query.to_pickle(run/'query.pkl')
    files.update({str(run/n):sha(run/n) for n in ['training.pkl','query.pkl']})
    write(run/'manifest.json',dict(spec=spec,files=files,source_directory=str(WORK),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        recipe=original['recipes']['realmlp_td'],parent_zip=str(MAIN/best['package']),confirmation_terminal_sha256=sha(CONF/'terminal-reconciliation.json'),
        local_four_seed_gate_passed=False,formal_promoted=False,release_kind=spec['release_kind'],explicit_grant_sha256=spec['explicit_grant_sha256'],platform_score=None))
    print(json.dumps(dict(status='frozen_user_exploration',full_procedures=1,native_adam=2,packages=1,desktop_copies=1,formal_promoted=False)),flush=True)


def context(inference=False):
    if inference:sys.addaudithook(native.block_training)
    setup();spec=read(WORK/SPEC);authority(spec);run=Path(spec['run_directory']);m=read(run/'manifest.json')
    if inference:
        paths=[*[str(WORK/p) for p in sources()],str(run/'query.pkl'),m['parent_zip'],spec['explicit_grant']]
        verify({p:m['files'][p] for p in paths})
    else:verify(m['files'])
    if m['source_directory']!=str(WORK) or m['spec']!=spec or m['local_four_seed_gate_passed'] is not False or m['formal_promoted'] is not False:
        raise ValueError('Frozen user exploration context changed')
    return run,m


def desktop():
    run,m=context(inference=True);r=read(run/'release.json');a=read(run/'package-audit.json');t=read(run/'terminal-reconciliation.json')
    if (t['status']!='passed' or t['actual_supervisor_exit_code']!=0 or t['packages']!=1 or t['native_adam']!=2 or a['status']!='passed'
            or t['audit_sha256']!=sha(run/'package-audit.json') or a['release_sha256']!=sha(run/'release.json')
            or t['report_sha256']!=sha(run/'release.json') or t['zip_sha256']!=r['zip_sha256'] or sha(r['zip'])!=r['zip_sha256']):
        raise ValueError('Actually completed native release and independent package audit required')
    query=pd.read_pickle(run/'query.pkl');parent,_=package_rows(m['parent_zip'],query.sample_id.tolist());source,data=package_rows(r['zip'],query.sample_id.tolist())
    if any(x['pred_tap_iron']!=y['pred_tap_iron'] for x,y in zip(parent,source)):raise ValueError('Parent iron strings changed')
    dest=Path(m['spec']['desktop_directory']);dest.mkdir(exist_ok=False);hashes={}
    for p in [Path(r['zip']),Path(r['zip']).with_name('result.csv')]:
        with (dest/p.name).open('xb') as f:f.write(p.read_bytes())
        if sha(dest/p.name)!=sha(p):raise ValueError('Desktop copy differs')
        hashes[p.name]=sha(dest/p.name)
    note='RealMLP MSE A20：用户指定的平台探索包\n\n上传同目录中的 ZIP；ZIP 内仅含 result.csv。\n时长 = 0.8 × 当前最佳时长 + 0.2 × RealMLP 默认 MSE 全量预测。\n铁量逐行保留父包 CSV 原字符串。父包：DE3_IRON_EMA_MEAN3_Q100，用户回传96.3979。\nG0：全量原生状态、冷推理、包回读及桌面一致性通过。\nG1：四切分有一个负收益，未正式晋级；本包尚无平台成绩，不承诺96.45。\n\nZIP SHA256：'+r['zip_sha256']+'\n'
    with (dest/'README.txt').open('x',encoding='utf-8') as f:f.write(note)
    rows,copied=package_rows(dest/Path(r['zip']).name,query.sample_id.tolist())
    if copied!=data or copied!=(dest/'result.csv').read_bytes():raise ValueError('Desktop ZIP/CSV readback differs')
    hashes['README.txt']=sha(dest/'README.txt')
    receipt=dict(status='passed',candidate=SCOPE['candidate'],display_name=SCOPE['display_name'],desktop=str(dest),zip=str(dest/Path(r['zip']).name),hashes=hashes,
        rows=322,unique_ids=322,official_order=True,crc_passed=True,iron_string_mismatches=0,terminal_reconciliation_sha256=sha(run/'terminal-reconciliation.json'),
        explicit_grant_sha256=m['explicit_grant_sha256'],source_release_sha256=sha(run/'release.json'),formal_promoted=False,platform_score=None,desktop_copies=1,agent_uploads=0,new_fits=0,new_optimizers=0,pid=os.getpid(),peak_rss_mib=native.memory())
    write(run/'desktop-delivery.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','full','cold','build','audit','desktop']);p.add_argument('--checks');a=p.parse_args()
    if a.action=='prepare':prepare(a.checks)
    elif a.action=='desktop':desktop()
    else:
        with patch.object(native,'context',context):
            if a.action in ['full','cold']:native.full(a.action=='cold')
            elif a.action=='build':native.build()
            else:native.audit()
