"""Read-only observation invoked by an external 600-second timer."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from bf_tap_r2.ptarl_protocol import ReservationLedger,file_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--service',required=True);args=parser.parse_args()
    service=subprocess.check_output(['systemctl','--user','show',args.service,
        '--property=MainPID,ActiveState,SubState,Result,ExecMainStatus,RuntimeMaxUSec'],text=True)
    phases={}
    for phase in ('development','confirmation'):
        ledger=args.root/phase/'ledger'
        if (ledger/'policy.json').exists():phases[phase]=ReservationLedger.open(ledger,file_hash(ledger/'policy.json')).inspect()
    event=dict(observed_ns=time.time_ns(),scheduled_interval_seconds=600,service_status=service,phases=phases)
    with (args.root/'scheduled-monitor.jsonl').open('a') as stream:
        json.dump(event,stream,sort_keys=True);stream.write('\n');stream.flush()
    print(json.dumps(event,sort_keys=True))


if __name__=='__main__':main()
