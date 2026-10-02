from pathlib import Path
import json
p=Path(__file__).resolve().parent;root=p.parents[1];run=root/"local/runs/tabm-time-swa-confirm-271828-314159-20261001"
for name in ["controller-finished.json","controller-failed.json"]:
 if (run/name).exists():print(json.dumps({"terminal":name,"receipt":json.loads((run/name).read_text())}));raise SystemExit
if not (p/"launch.json").exists():
 for name in ["start-failed-r1.json","start-finished-r1.json"]:
  if (p/name).exists():print(name,(p/name).read_text());raise SystemExit
 print(json.dumps({"status":"admission_in_progress"}));raise SystemExit
launch=json.loads((p/"launch.json").read_text());proc=Path(f"/proc/{launch['pid']}/stat")
if not proc.exists():print(json.dumps({"status":"stopped_without_terminal"}));raise SystemExit
stat=proc.read_text();fields=stat[stat.rfind(")")+2:].split()
alive=fields[19]==launch["start_ticks"] and Path("/proc/sys/kernel/random/boot_id").read_text().strip()==launch["boot_id"] and fields[0]!="Z"
print(json.dumps({"status":"running" if alive else "stopped_identity_mismatch","complete_units":len(list((run/"confirmation-r1").glob("s*-f*/complete.json"))),"expected_units":10,"pid":launch["pid"]}))
