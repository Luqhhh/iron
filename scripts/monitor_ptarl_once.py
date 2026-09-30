"""Read one PTaRL snapshot; schedule externally once every 600 seconds."""
import argparse
import json
from pathlib import Path
from bf_tap_r2.ptarl_controller import monitor
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True)
    a=p.parse_args();print(json.dumps(monitor(a.control_root),sort_keys=True))
