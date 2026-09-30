"""Read one progress snapshot; healthy training is silent."""
import argparse
import hashlib
import json
from pathlib import Path
from bf_tap_r2.t2g_model import canonical
from bf_tap_r2.t2g_freeze import write_new

def read_progress(root):
    root=Path(root)
    terminal=root/"orchestration-finished.json"
    if terminal.exists():return json.loads(terminal.read_text())
    events=root/"events.jsonl"
    if not events.exists():return {"status":"not_started"}
    # A crash during append leaves the previous complete snapshot readable.
    rows=events.read_text().splitlines()
    for line in reversed(rows):
        try:return json.loads(line)
        except json.JSONDecodeError:continue
    return {"status":"not_started"}

def claim_notification(root,state):
    root=Path(root)
    notices=root/"notification-state";notices.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(canonical(state).encode()).hexdigest()
    try:write_new(notices/(digest+".json"),state)
    except FileExistsError:return False
    return True

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--root",required=True,type=Path)
    state=read_progress(p.parse_args(argv).root)
    if state["status"] in {"failed","failed_g0","completed","waiting_reference"}:
        print(canonical(state))
    return state

if __name__=="__main__":main()
