"""One-shot execute then fresh-process zero-fit audit; never resume a fit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from bf_tap_r2.incumbent_reference_ledger import file_hash,write_new
from bf_tap_r2.incumbent_reference_phase import verify_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--sha256',required=True)
    args=p.parse_args();m=verify_manifest(args.manifest,args.sha256);out=Path(m['output'])
    write_new(out/'controller.started.json',dict(manifest_sha256=args.sha256))
    base=[sys.executable,'-m','bf_tap_r2.incumbent_reference_phase','--manifest',str(args.manifest),'--sha256',args.sha256]
    try:
        for phase in ('execute','audit'):
            command=[*base,'--phase',phase]
            if phase=='audit':command+=['--run-sha256',file_hash(out/'run.finished.json')]
            with (out/f'{phase}.log').open('xb') as log:
                subprocess.run(command,cwd=m['workspace'],stdout=log,stderr=subprocess.STDOUT,check=True)
        write_new(out/'controller.complete.json',dict(status='passed',complete_sha256=file_hash(out/'complete.json')))
    except BaseException as exc:
        write_new(out/'controller.failed.json',dict(type=type(exc).__name__,message=str(exc)));raise
    print(json.dumps(dict(status='passed',complete_sha256=file_hash(out/'complete.json'))),flush=True)


if __name__=='__main__':main()
