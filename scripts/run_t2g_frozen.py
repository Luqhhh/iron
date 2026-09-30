"""One-shot frozen T2G orchestration; only missing references permit resumption."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from bf_tap_r2.data import TARGETS
from bf_tap_r2.t2g_freeze import verify_manifest,attach_reference,write_new,read_bound
from bf_tap_r2.t2g_model import canonical,file_hash
from bf_tap_r2.t2g_preflight import run_preflight
from bf_tap_r2.t2g_reference import verify_reference
from bf_tap_r2.t2g_run import run_phase
from bf_tap_r2.t2g_verify import decide

def event(root,status,**details):
    record={"status":status,"at":datetime.now(timezone.utc).isoformat(),**details}
    with (root/"events.jsonl").open("a") as f:
        f.write(canonical(record)+"\n");f.flush();os.fsync(f.fileno())
    return record

def audit_fresh(manifest,output):
    with (Path(output)/"cold-audit.log").open("x") as stream:
        subprocess.run([sys.executable,"-m","bf_tap_r2.t2g_audit","--manifest",str(manifest),
            "--output",str(output)],check=True,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())

def run_controller(engineering_manifest,native_root,overlay_root,root):
    engineering_manifest=Path(engineering_manifest).resolve()
    m=verify_manifest(engineering_manifest)
    base=Path(m["output"]);root=Path(root).resolve()
    if root!=base/"controller-r1":raise ValueError("Canonical controller directory required")
    root.mkdir(parents=True,exist_ok=True)
    with (root/"controller.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {"status":"already_running"}
        if (root/"orchestration-finished.json").exists():
            raise ValueError("Terminal controller cannot retry")
        history=[json.loads(line) for line in (root/"events.jsonl").read_text().splitlines()] if (root/"events.jsonl").exists() else []
        if history and history[-1]["status"]!="waiting_reference":
            raise ValueError("Claimed or interrupted controller cannot retry")
        try:
            if history:
                anchor=json.loads((root/"g0-observed.complete.json").read_text())
                g0=read_bound(root/"g0-observed.json",anchor["sha256"])
                actual=base/"g0-r1/preflight.json"
                marker=json.loads((actual.parent/"preflight.complete.json").read_text())
                if read_bound(actual,marker["sha256"])!=g0:
                    raise ValueError("Saved G0 evidence differs")
            else:
                event(root,"running_g0",engineering_sha256=file_hash(engineering_manifest))
                g0=run_preflight(engineering_manifest,base/"g0-r1")
                write_new(root/"g0-observed.json",g0)
                write_new(root/"g0-observed.complete.json",{"sha256":file_hash(root/"g0-observed.json")})
            if g0["status"]!="passed":
                result=event(root,"failed_g0",official_fits=0,g0=g0)
                write_new(root/"orchestration-finished.json",result);return result
            try:reference=verify_reference(Path(native_root),Path(overlay_root))
            except FileNotFoundError as e:
                return event(root,"waiting_reference",official_fits=0,message=str(e))
            formal=attach_reference(engineering_manifest,reference,base/"formal.json")
            development=base/"development-r1"
            event(root,"running_development")
            run_phase(formal,"development",development,list(TARGETS))
            event(root,"auditing_development")
            audit_fresh(formal,development)
            decision=decide(development,None,reference)
            write_new(root/"independent-development-decision-r1.json",decision)
            selected=decision["selected_for_confirmation"]
            if selected:
                confirmation=base/"confirmation-r1"
                event(root,"running_confirmation",selected_targets=selected)
                run_phase(formal,"confirmation",confirmation,selected)
                event(root,"auditing_confirmation")
                audit_fresh(formal,confirmation)
                decision=decide(development,confirmation,reference)
                write_new(root/"independent-four-seed-decision-r1.json",decision)
            result=event(root,"completed",decision=decision)
            write_new(root/"orchestration-finished.json",result);return result
        except BaseException as e:
            result=event(root,"failed",type=type(e).__name__,message=str(e),retry_authorized=False)
            write_new(root/"orchestration-finished.json",result)
            raise

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--engineering-manifest",required=True,type=Path)
    p.add_argument("--reference-native",required=True,type=Path)
    p.add_argument("--reference-overlay",required=True,type=Path)
    p.add_argument("--root",required=True,type=Path)
    a=p.parse_args(argv)
    print(canonical(run_controller(a.engineering_manifest,a.reference_native,a.reference_overlay,a.root)))

if __name__=="__main__":main()
