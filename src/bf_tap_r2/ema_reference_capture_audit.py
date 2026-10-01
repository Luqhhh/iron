"""Cold audit of an externally bound, completed reference capture.

Run the CLI in a new process after the original warm worker has exited.
No fitting, weight choice or scientific execution admission happens here.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

from .ema_reference_artifacts import audit_witness,sha,write_new
from .ema_reference_ledger import KINDS
from .v7_periodic import digest


def _read(path,expected,hashes):
    path=Path(path)
    if sha(path)!=expected:raise ValueError('Reference cold artifact identity changed: '+str(path))
    hashes[str(path.resolve())]=expected
    return json.loads(path.read_text())


def audit_capture(directory,expected_receipt_sha256):
    directory=Path(directory).resolve()
    if (directory/'cold-complete.json').exists():
        raise FileExistsError('Reference capture already has a completed cold audit')
    hashes={}
    warm=_read(directory/'complete.json',expected_receipt_sha256,hashes)
    if warm['status']!='warm_original_factory_capture_closed_cold_audits_pending' or warm['pipelines']!=32:
        raise ValueError('Reference warm factory capture has not closed all 32 roles')
    start=_read(directory/'start.json',warm['start_sha256'],hashes)
    for key in ('source_directory','split_seed','fold','plan_digest'):
        if start[key]!=warm[key]:raise ValueError('Reference warm source/split/plan identity changed')
    if digest(start['plan'])!=start['plan_digest']:
        raise ValueError('Reference warm plan changed')
    roles={r['name']:r for r in start['plan']['roles']}
    if len(roles)!=32 or set(roles)!=set(warm['roles']):
        raise ValueError('Reference cold role coverage changed')
    if any(not isinstance(n,str) or not n or Path(n).name!=n or n in {'.','..'} for n in roles):
        raise ValueError('Reference cold role name is invalid')
    if {p.name for p in directory.iterdir() if p.is_dir()}!=set(roles):
        raise ValueError('Reference cold directory coverage changed')
    source_directory=start['source_directory']
    if not Path(source_directory).is_absolute() or str(Path(source_directory).resolve())!=source_directory:
        raise ValueError('Reference cold original fit origin is invalid')
    if type(start['split_seed']) is not int or type(start['fold']) is not int:
        raise ValueError('Reference cold split identity is invalid')
    sources=start['source_hashes']
    if not sources:raise ValueError('Reference cold source catalogue is empty')
    hashes.update(sources)
    requests=[];native_totals={k:0 for k in KINDS};warm_pids=set()
    for name,plan in roles.items():
        role=directory/name;summary=warm['roles'][name]
        complete=_read(role/'complete.json',summary['receipt_sha256'],hashes)
        role_start=_read(role/'start.json',complete['start_sha256'],hashes)
        identity=dict(source_directory=source_directory,split_seed=start['split_seed'],fold=start['fold'],trial_id=name)
        if (complete['identity']!=identity or role_start['identity']!=identity
                or complete['plan_digest']!=digest(plan) or role_start['plan']!=plan
                or role_start['training_ids']!=start['training_ids'] or role_start['query_ids']!=start['query_ids']
                or role_start['query_frame_digest']!=start['query_frame_digest']
                or (role/'failure.json').exists()):
            raise ValueError('Reference cold pipeline identity, partition or query changed')
        native=_read(role/'native/scope-complete.json',complete['native_receipt_sha256'],hashes)
        native_start=_read(role/'native/scope-start.json',native['scope_start_sha256'],hashes)
        warm_pids.add(native_start['pid'])
        if (native['status']!='passed' or native['identity']!=identity
                or native_start['identity']!=identity or native_start['source_hashes']!=sources
                or native_start['training_ids']!=start['training_ids'] or native_start['query_ids']!=start['query_ids']
                or native_start['expected']!=plan['expected_native_calls']
                or native['counts']!=complete['native_counts'] or native['counts']!=plan['expected_native_calls']
                or (role/'native/scope-failure.json').exists()):
            raise ValueError('Reference cold native budget or partition changed')
        call_files={str(p.relative_to(role/'native')) for d in (role/'native').glob('call-*') for p in d.iterdir() if p.is_file()}
        if call_files!=set(native['call_hashes']) or len(list((role/'native').glob('call-*')))!=sum(native['counts'].values()):
            raise ValueError('Reference cold native call coverage changed')
        for relative,h in native['call_hashes'].items():
            path=role/'native'/relative
            if not path.resolve().is_relative_to((role/'native').resolve()):
                raise ValueError('Reference native receipt escapes its scope')
            if sha(path)!=h:raise ValueError('Reference cold native call artifact changed')
            hashes[str(path.resolve())]=h
        for k,n in native['counts'].items():native_totals[k]+=n
        keys={'final'}
        if plan['family']=='legacy_inner_select_refit_tree':keys.add('tree-selector')
        if plan['family']=='matching_selector_fresh_refit_network':keys.add('selector-terminal/model-witness')
        if set(complete['witnesses'])!=keys or summary['witnesses']!=complete['witnesses']:
            raise ValueError('Reference cold retained model coverage changed')
        for key,h in complete['witnesses'].items():
            witness=_read(role/key/'complete.json',h,hashes)
            fit_call_id='selector-terminal' if key=='selector-terminal/model-witness' else key
            if (witness['identity']!=dict(identity,fit_call_id=fit_call_id)
                    or witness['source_hashes']!=sources
                    or witness['full_batch_atol']!=start['full_batch_atol'] or witness['row_atol']!=start['row_atol']):
                raise ValueError('Reference cold model identity, sources or tolerance changed')
            fit_ids=witness['training_ids'];query_ids=witness['query_ids']
            if key=='final':
                if fit_ids!=start['training_ids'] or query_ids!=start['query_ids']:
                    raise ValueError('Reference cold final model outer partition changed')
            elif (set(fit_ids)&set(query_ids) or set(fit_ids)|set(query_ids)!=set(start['training_ids'])):
                raise ValueError('Reference cold selector partition does not close outer training')
            requests.append((name,key,role/key,h))
    if native_totals!=warm['native_counts'] or native_totals!=start['plan']['expected_native_calls_per_factory']:
        raise ValueError('Reference cold factory native totals changed')
    if os.getpid() in warm_pids:
        raise ValueError('Reference factory cold audit must use a different process from every warm fit')
    for path,h in hashes.items():
        if sha(path)!=h:raise ValueError('Reference cold source/artifact changed before inference')
    audits={};differences={k:0. for k in ('full','reverse','chunk','singleton')}
    for name,key,path,h in requests:
        result=audit_witness(path,h)
        audits[name+'/'+key]=dict(receipt_sha256=h,cold_audit_sha256=sha(path/'cold-audit.json'),
            differences=result['differences'],rows=result['rows'],identity=result['identity'])
        for k,v in result['differences'].items():differences[k]=max(differences[k],v)
        gc.collect()
    for path,h in hashes.items():
        if sha(path)!=h:raise ValueError('Reference cold source/artifact changed during inference')
    payload=dict(status='passed',warm_capture_receipt_sha256=expected_receipt_sha256,
        source_directory=source_directory,split_seed=start['split_seed'],fold=start['fold'],
        plan_digest=start['plan_digest'],pipelines=32,retained_states=len(audits),native_counts=native_totals,
        cold_pid=os.getpid(),differences=differences,audits=audits,new_fits=0,training_csv_reads=0)
    write_new(directory/'cold-complete.json',payload)
    return payload


def main():
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Reference cold numerical thread contract mismatch')
    import torch
    if str(torch.__version__)!='2.14.0+cpu':raise ValueError('Reference cold CPU runtime identity mismatch')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path);parser.add_argument('--receipt-sha256',required=True)
    args=parser.parse_args()
    print(json.dumps(audit_capture(args.directory,args.receipt_sha256),allow_nan=False))


if __name__=='__main__':main()
