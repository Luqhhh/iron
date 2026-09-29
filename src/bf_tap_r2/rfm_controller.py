"""One-shot serial RFM state machine; scheduling is delegated to bounded phases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from .rfm_freeze import verify_manifest, private_path
from .rfm_preflight import verify_admission
from .rfm_protocol import write_new, file_hash


def sequence(invoke):
    """Each nonzero subprocess exit propagates before any successor is called."""
    result={}
    for phase in ('development','confirmation'):
        previous=[] if phase=='development' else ['--development-sha256',result['development_audit']['audit_sha256']]
        fit=invoke(phase+'-run','bf_tap_r2.rfm_run',['run','--phase',phase,*previous])
        audit=invoke(phase+'-audit','bf_tap_r2.rfm_run',
            ['audit','--phase',phase,'--complete-sha256',fit['complete_sha256'],*previous])
        extra=[] if phase=='development' else ['--development-arithmetic-sha256',result['development_arithmetic']['arithmetic_sha256']]
        arithmetic=invoke(phase+'-arithmetic','bf_tap_r2.rfm_arithmetic',
            ['--phase',phase,'--audit-sha256',audit['audit_sha256'],*extra])
        result[phase+'_audit']=audit;result[phase+'_arithmetic']=arithmetic
        if phase=='development' and not arithmetic['selected_targets']:
            result['confirmation_skipped']='no_development_finalist';break
    return result


def controller(manifest_path,manifest_sha256,preflight_sha256):
    manifest=verify_manifest(manifest_path,manifest_sha256)
    root=private_path(manifest['workspace'],Path(manifest_path).resolve().parent)
    verify_admission(root/'preflight/admission.json',preflight_sha256,manifest_path,manifest_sha256)
    write_new(root/'orchestration-started.json',dict(manifest_sha256=manifest_sha256,
              preflight_sha256=preflight_sha256,time_ns=time.time_ns()))
    events=[]
    def invoke(name,module,args):
        verify_manifest(manifest_path,manifest_sha256)
        command=[sys.executable,'-m',module,*args,'--manifest',str(manifest_path),
                 '--manifest-sha256',manifest_sha256]
        if module.endswith('rfm_run'):command+=['--preflight-sha256',preflight_sha256]
        write_new(root/(name+'-started.json'),dict(command=command,time_ns=time.time_ns()))
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.run(command,cwd=manifest['workspace'],stdout=log,stderr=subprocess.STDOUT)
        event=dict(stage=name,returncode=proc.returncode,time_ns=time.time_ns())
        events.append(event);write_new(root/(name+'-event.json'),event)
        if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,command)
        result=json.loads((root/(name+'.log')).read_text().splitlines()[-1])
        write_new(root/(name+'-result.json'),result)
        return result
    try:
        results=sequence(invoke)
        verify_manifest(manifest_path,manifest_sha256)
        terminal=dict(status='completed',events=events,results=results,packages=0,uploads=0)
    except BaseException as exc:
        terminal=dict(status='failed',events=events,type=type(exc).__name__,message=str(exc),packages=0,uploads=0)
        write_new(root/'completion-event.json',terminal)
        raise
    write_new(root/'completion-event.json',terminal)
    return terminal


def monitor(root):
    """One read-only observation; caller's systemd timer provides 600s cadence."""
    root=Path(root)
    terminal=root/'completion-event.json'
    if terminal.exists():return json.loads(terminal.read_text())
    result={'status':'running' if (root/'orchestration-started.json').exists() else 'not_started','phases':{}}
    for phase in ('development','confirmation'):
        events=root/phase/'ledger/events'
        result['phases'][phase]={s:len(list(events.glob('outer_fit-*.'+s+'.json'))) for s in ('started','complete','failed')}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--preflight-sha256',required=True)
    a=p.parse_args();print(json.dumps(controller(a.manifest,a.manifest_sha256,a.preflight_sha256),sort_keys=True))


if __name__=='__main__':main()
