"""Durable supervisor command: one-shot controller; stop its monitor at exit."""
import argparse
from pathlib import Path
import subprocess
import sys

from bf_tap_r2.dnnr_ledger import write_new


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--monitor-timer',required=True);a=p.parse_args();root=a.manifest.resolve().parent
    command=[sys.executable,'-m','bf_tap_r2.dnnr_phase_controller','--manifest',str(a.manifest),'--manifest-sha256',a.manifest_sha256]
    with (root/'controller-live.log').open('x') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    write_new(root/'supervisor-terminal.json',dict(exit_code=result.returncode,manifest_sha256=a.manifest_sha256))
    subprocess.run(['systemctl','--user','stop',a.monitor_timer],check=False)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
