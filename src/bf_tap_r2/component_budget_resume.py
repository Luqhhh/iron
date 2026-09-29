"""Explicit user-authorized runtime readmission; no repeated synthetic fits."""
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import psutil
import yaml
from .component_regularization_run import sources
from .v49_run import verify_hashes, append_event
from .v7_periodic import file_hash, write_new

OLD_SPEC = 'configs/component_augmentation_representation/SPEC.yaml'
NEW_SPEC = 'configs/component_augmentation_representation/BUDGET8.yaml'
REPORT_SHA = '69656f20918a25127f9c0e308e55d6edc758b1942dfa6d91874ad81205b9a02d'


def validate_amendment(old, new, report):
    expected=deepcopy(old)
    expected['preflight']['max_projected_hours']=8.0
    expected['run_root']='local/runs/component-augmentation-representation-budget8'
    expected['source_plan']='docs/component_augmentation_representation/BUDGET8.md'
    if new != expected: raise ValueError('Only the authorized runtime amendment is allowed')
    if report['status']!='failed' or [k for k,v in report['checks'].items() if not v]!=['runtime']:
        raise ValueError('Only a runtime-only admission failure is reusable')
    if not 0 < report['projected_hours'] <= 8.0:
        raise ValueError('Measured projection exceeds authorized budget')


def admit(root):
    old=yaml.safe_load((root/OLD_SPEC).read_text())
    new=yaml.safe_load((root/NEW_SPEC).read_text())
    prior=root/old['run_root']/'preflight-r1'
    report_path=prior/'report.json'
    if file_hash(report_path)!=REPORT_SHA: raise ValueError('Original report changed')
    report=json.loads(report_path.read_text())
    validate_amendment(old,new,report)
    if report['spec_sha256']!=file_hash(root/OLD_SPEC): raise ValueError('Original spec changed')
    verify_hashes(root,report['source_hashes'])
    # Repeat fresh-process cold arithmetic, without training or tolerance changes.
    subprocess.run([sys.executable,'-m','bf_tap_r2.component_regularization_preflight',
                    '--cold',str(prior),'--spec',OLD_SPEC],cwd=root,check=True)
    peak=max(r['peak_rss_mib'] for r in report['results'])
    if 4*peak+1024 >= psutil.virtual_memory().available/1024**2:
        raise ValueError('Current available memory fails original rule')
    artifacts={str(p.relative_to(root)):file_hash(p) for p in prior.rglob('*') if p.is_file()}
    directory=root/new['run_root'];out=directory/'preflight-r1'
    out.mkdir(parents=True,exist_ok=False)
    manifest={'source_hashes':sources(root,new),'spec_sha256':file_hash(root/NEW_SPEC),
              'original_report_sha256':REPORT_SHA,'reused_artifact_hashes':artifacts,
              'user_authorization':'允许7.93，8小时上限','new_fits':0}
    write_new(out/'manifest.json',manifest)
    updated=deepcopy(report)
    updated.update(status='passed',source_hashes=manifest['source_hashes'],
        spec_sha256=manifest['spec_sha256'],synthetic_new_refits=0,synthetic_reused_refits=6,
        original_report_sha256=REPORT_SHA,readmission_manifest_sha256=file_hash(out/'manifest.json'))
    updated['checks']['runtime']=True
    write_new(out/'report.json',updated)
    return new,directory


def main():
    root=Path.cwd();spec,directory=admit(root);run_root=spec['run_root']
    write_new(directory/'workflow-start.json',{'pid':os.getpid(),'started_time_ns':time.time_ns(),
        'interval_seconds':600,'explicit_budget_amendment':True})
    stages=[('development','bf_tap_r2.component_regularization_run',['--output',f'{run_root}/development-r1']),
        ('development_audit','bf_tap_r2.component_regularization_audit',['--output',f'{run_root}/development-r1']),
        ('conditional_confirmation','bf_tap_r2.component_regularization_run',['--output',f'{run_root}/confirmation-r1','--development',f'{run_root}/development-r1']),
        ('confirmation_audit','bf_tap_r2.component_regularization_audit',['--output',f'{run_root}/confirmation-r1'])]
    events=[]
    for name,module,args in stages:
        if name=='confirmation_audit' and not (directory/'confirmation-r1').exists():
            event={'stage':name,'skipped':'no_development_finalist','new_fits':0,'time_ns':time.time_ns()}
            events.append(event);append_event(directory/'workflow-events.jsonl',event);continue
        with (directory/f'{name}.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-m',module,*args,'--spec',NEW_SPEC],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
            event={'event':'start','stage':name,'pid':child.pid}
            append_event(directory/'workflow-events.jsonl',event);print(json.dumps(event),flush=True)
            while True:
                try:code=child.wait(timeout=600);break
                except subprocess.TimeoutExpired:
                    status={'event':'scheduled_observation','stage':name,'pid':child.pid,'time_ns':time.time_ns(),'status':'running'}
                    run=directory/('confirmation-r1' if 'confirmation' in name else 'development-r1')
                    ledger=run/'fit_ledger.jsonl'
                    if ledger.exists():
                        rows=[json.loads(line) for line in ledger.read_text().splitlines()]
                        status['completed_units']=sum(r.get('event')=='complete' for r in rows)
                        status['failed_units']=sum(r.get('event')=='failed' for r in rows)
                    append_event(directory/'scheduled-observations.jsonl',status);print(json.dumps(status),flush=True)
        event={'stage':name,'returncode':code,'time_ns':time.time_ns()}
        events.append(event);append_event(directory/'workflow-events.jsonl',event);print(json.dumps(event),flush=True)
        if code:
            write_new(directory/'completion-event.json',{'status':'failed','events':events,'no_automatic_restart':True})
            raise SystemExit(code)
    write_new(directory/'completion-event.json',{'status':'completed','events':events,'packages':0,'uploads':0})


if __name__=='__main__':main()
