"""Locked Joint MAE checks with zero optimizer construction enforced."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest
import torch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK / 'src'))
from bf_tap_r2.ema_evaluation_diagnostics import sha, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    out = parser.parse_args().output.resolve()
    if sys.version_info[:2] != (3,12) or out.exists():
        raise ValueError('Python3.12 and fresh check directory required')
    out.mkdir(parents=True,exist_ok=False)
    attempts = []
    def blocked(*args,**kwargs):
        attempts.append('Optimizer.__init__')
        raise RuntimeError('Joint MAE checks forbid optimizer construction')
    original = torch.optim.Optimizer.__init__
    torch.optim.Optimizer.__init__ = blocked
    xml = out / 'locked-tests.xml'
    try:
        code = int(pytest.main(['-q','tests/test_q75_joint_mae.py','tests/test_q75_gaussian_time.py',f'--junitxml={xml}']))
    finally:
        torch.optim.Optimizer.__init__ = original
    sources = [WORK / p for p in ('configs/q75_joint_mae/SPEC.json',
        'docs/q75_joint_mae/PREREGISTRATION.md','src/bf_tap_r2/q75_joint_mae.py','src/bf_tap_r2/joint_mae_model.py',
        'scripts/q75_joint_mae.py','scripts/check_q75_joint_mae.py','tests/test_q75_joint_mae.py')]
    tests = ET.parse(xml).getroot().find('testsuite')
    receipt = dict(exit_code=code if not attempts else 1,python_version=sys.version,
        optimizer_runs=0,optimizer_constructor_attempts=len(attempts),passed=int(tests.attrib['tests']),
        junit_path=str(xml),junit_sha256=sha(xml),source_hashes={str(p):sha(p) for p in sources})
    write_new(out / 'receipt.json',receipt)
    print(json.dumps(receipt),flush=True)
    raise SystemExit(receipt['exit_code'])


if __name__ == '__main__':
    main()
