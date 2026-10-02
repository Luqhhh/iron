"""Record the actual child exit; inspect running work only every600 seconds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();out=Path(args.output).resolve()
    supervision=out.parent/(out.name+'-supervision-r1')
    supervision.mkdir(parents=True,exist_ok=False)
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    started=time.time_ns()
    with (supervision/'controller.log').open('x') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while True:
            try:
                result=process.wait(timeout=600);break
            except subprocess.TimeoutExpired:
                with (supervision/'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(time_ns=time.time_ns(),pid=process.pid,interval_seconds=600,
                        complete_units=sum(1 for p in out.glob('s*-f*/complete.json'))))+'\n')
                    stream.flush();os.fsync(stream.fileno())
    record=dict(command=command,pid=process.pid,observer_pid=os.getpid(),exit_code=result,
                started_ns=started,ended_ns=time.time_ns(),seconds_descriptive=(time.time_ns()-started)/1e9,
                monitor_interval_seconds=600,between_check_polling=False)
    for p in [supervision/'actual-exit.json',out/'actual-exit.json']:
        if p.parent.exists():
            with p.open('x') as stream:
                json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(record))
    return result


if __name__=='__main__':sys.exit(main())
