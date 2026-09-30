"""One-shot durable synthetic resource supervisor; no retry or formal fit."""
import argparse
from pathlib import Path
import subprocess
import sys

from bf_tap_r2.danet_ledger import write_new


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--monitor-timer',required=True);args=parser.parse_args();root=args.manifest.resolve().parent
    command=[sys.executable,'-m','bf_tap_r2.danet_preflight','run','--manifest',str(args.manifest),
        '--manifest-sha256',args.manifest_sha256]
    with (root/'supervisor-live.log').open('x') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    write_new(root/'supervisor-terminal.json',dict(exit_code=result.returncode,manifest_sha256=args.manifest_sha256))
    subprocess.run(['systemctl','--user','stop',args.monitor_timer],check=False)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
