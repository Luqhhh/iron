"""Run the prospectively frozen Q75 error relocation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bf_tap_r2.q75_error_relocation import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=ROOT/'configs/q75_error_relocation/SPEC.json')
    args = parser.parse_args()
    report = run(args.spec, ROOT)
    print(json.dumps({'G0': report['G0'], 'new_fits': report['new_fits'], 'seeds': list(report['seeds'])}))
