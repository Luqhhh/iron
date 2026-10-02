from pathlib import Path
import subprocess,sys,os,json,time,traceback
from bf_tap_r2.tabm_swa_protocol import write_new,sha
root=Path.cwd();p=root/"local/swa-confirmation-20261002"
write_new(p/"start-reserved-r1.json",dict(time=time.time(),authorization="User confirmed 271828/314159 substitution; existing 40 optimizer allocation only"))
try:
 logs={}
 for name,python in [("neural",root/".venv/bin/python"),("locked",root/"local/envs/amf-locked/bin/python")]:
  path=p/(name+"-controller-tests-r1.log")
  with path.open("x") as log:subprocess.run([str(python),"-m","pytest","-q",str(p/"test_controller.py")],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  logs[str(path.relative_to(root))]=sha(path)
 write_new(p/"tests-complete-r1.json",dict(status="passed",logs=logs,source_sha256=sha(p/"controller.py"),new_fits=0))
 subprocess.run([sys.executable,str(p/"controller.py"),"admit"],cwd=root,check=True)
 run=root/"local/runs/tabm-time-swa-confirm-271828-314159-20261001"
 write_new(p/"launch-reserved-r1.json",dict(time=time.time(),preflight_sha256=sha(run/"preflight.json")))
 with (p/"execute-r1.log").open("x") as log:process=subprocess.Popen([sys.executable,"-u",str(p/"controller.py"),"execute"],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
 stat=Path(f"/proc/{process.pid}/stat").read_text();fields=stat[stat.rfind(")")+2:].split()
 write_new(p/"launch.json",dict(pid=process.pid,start_ticks=fields[19],boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),preflight_sha256=sha(run/"preflight.json")))
 write_new(p/"start-finished-r1.json",dict(status="confirmation_controller_launched",time=time.time(),pid=process.pid))
except BaseException as e:
 write_new(p/"start-failed-r1.json",dict(status="failure_preserved",time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
