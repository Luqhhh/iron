"""One read-only ledger/service observation; timer enforces the 600s cadence."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from bf_tap_r2.danet_ledger import ReservationLedger,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--service',required=True)
    a=p.parse_args();status=subprocess.check_output(['systemctl','--user','show',a.service,
        '--property=MainPID,ActiveState,SubState,Result,ExecMainStatus'],text=True)
    phases={}
    for phase in ('development','confirmation'):
        ledger=a.root/phase/'ledger'
        if (ledger/'policy.json').exists():phases[phase]=ReservationLedger.open(ledger,file_hash(ledger/'policy.json')).inspect()
    event=dict(observed_ns=time.time_ns(),scheduled_interval_seconds=600,service_status=status,phases=phases)
    with (a.root/'scheduled-monitor.jsonl').open('a') as stream:json.dump(event,stream,sort_keys=True);stream.write('\n');stream.flush()
    print(json.dumps(event,sort_keys=True))


if __name__=='__main__':main()
