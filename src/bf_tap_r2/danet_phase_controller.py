"""One-shot serial development/audit/arithmetic, then earned confirmation only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .danet_phase_freeze import verify_manifest
from .danet_ledger import write_new


def sequence(invoke):
    result={}
    for phase in ('development','confirmation'):
        prior=[] if phase=='development' else ['--development-audit-sha256',result['development_audit']['audit_sha256'],
            '--development-arithmetic-sha256',result['development_arithmetic']['arithmetic_sha256']]
        fit=invoke(phase+'-run','bf_tap_r2.danet_phase_run',['run','--phase',phase,*prior])
        audit=invoke(phase+'-audit','bf_tap_r2.danet_phase_run',['audit','--phase',phase,
            '--complete-sha256',fit['complete_sha256'],*prior])
        arithmetic=invoke(phase+'-arithmetic','bf_tap_r2.danet_phase_arithmetic',
            ['--phase',phase,'--audit-sha256',audit['audit_sha256'],*prior])
        result[phase+'_audit']=audit;result[phase+'_arithmetic']=arithmetic
        if phase=='development' and not arithmetic['selected_pairs']:
            result['confirmation_skipped']='no_development_finalist';break
    return result


def controller(manifest_path,manifest_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256);root=Path(manifest_path).resolve().parent
    write_new(root/'orchestration-started.json',dict(manifest_sha256=manifest_sha256,time_ns=time.time_ns(),release_authorized=False))
    events=[]
    def invoke(name,module,args):
        verify_manifest(manifest_path,manifest_sha256)
        command=[sys.executable,'-m',module,*args,'--manifest',str(manifest_path),'--manifest-sha256',manifest_sha256]
        write_new(root/(name+'-started.json'),dict(command=command,time_ns=time.time_ns()))
        with (root/(name+'.log')).open('x') as stream:
            result=subprocess.run(command,cwd=manifest['workspace'],stdout=stream,stderr=subprocess.STDOUT)
        event=dict(stage=name,returncode=result.returncode,time_ns=time.time_ns());events.append(event)
        write_new(root/(name+'-event.json'),event)
        if result.returncode:raise subprocess.CalledProcessError(result.returncode,command)
        value=json.loads((root/(name+'.log')).read_text().splitlines()[-1]);write_new(root/(name+'-result.json'),value)
        return value
    try:
        results=sequence(invoke);verify_manifest(manifest_path,manifest_sha256)
        terminal=dict(status='completed',events=events,results=results,new_full_data_fits=0,packages=0,uploads=0)
    except BaseException as exc:
        write_new(root/'completion-event.json',dict(status='failed',events=events,type=type(exc).__name__,message=str(exc),packages=0,uploads=0))
        raise
    write_new(root/'completion-event.json',terminal)
    return terminal


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
    print(json.dumps(controller(a.manifest,a.manifest_sha256),sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
