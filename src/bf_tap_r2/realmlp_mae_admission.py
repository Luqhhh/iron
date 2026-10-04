"""Synthetic-only native MAE admission after the preceding MSE batch closes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import pandas as pd

from . import realmlp_native_admission as harness
from .ema_nested_residual import read,write,sha,verify,save_arrays
from .realmlp_development_audit import arrays
from .realmlp_mae_recipe import mae_recipe
from .realmlp_mae_observer import observe_mae_native
from .realmlp_time_development import setup

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
SPEC='configs/realmlp_mae_admission/SPEC.json'


def sources():
    return {str(p.relative_to(WORK)):sha(p) for p in [*list((WORK/'src').rglob('*.py')),WORK/SPEC,
        WORK/'tests/test_realmlp_mae_recipe.py',WORK/'tests/test_realmlp_mae_observer.py',WORK/'tests/test_realmlp_native_audit.py',WORK/'tests/test_realmlp_state_adapter.py',
        WORK/'docs/realmlp_mae/ADMISSION.md',WORK/'configs/round2_v9/SPEC.yaml',WORK/'uv.lock',WORK/'pyproject.toml']}


def predecessors():
    confirmation=MAIN/'local/runs/realmlp-time-confirmation-20261004/confirmation-r1'
    p=confirmation/'terminal-reconciliation.json';c=read(p)
    if c['status']!='passed' or c['actual_supervisor_exit_code']!=0:raise ValueError('Prior MSE confirmation must close first')
    files={str(p):sha(p)}
    if c['four_seed_gate_passed']:
        p=MAIN/'local/runs/realmlp-time-release-20261004/release-r1/terminal-reconciliation.json';r=read(p)
        if r['status']!='passed' or r['actual_supervisor_exit_code']!=0:raise ValueError('Qualified MSE release must close first')
        files[str(p)]=sha(p)
    return files


def prepare(checks):
    import pytabkit
    original=setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);before=predecessors()
    if run.exists():raise FileExistsError('Consumed native MAE admission directory')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():raise ValueError('Committed source required')
    c=read(checks)
    if c['status']!='passed' or c['actual_exit_code']!=0 or c['sources']!=sources() or c['junit_sha256']!=sha(c['junit']):raise ValueError('Exact locked no-fit checks required')
    files={str(WORK/p):h for p,h in sources().items()};files.update(before)
    for p in [Path(checks),Path(c['junit']),*Path(pytabkit.__file__).parent.rglob('*.py')]:files[str(p.resolve())]=sha(p)
    verify(files);run.mkdir(parents=True);training,query,y=harness.synthetic(spec)
    training.to_pickle(run/'training.pkl');query.to_pickle(run/'query.pkl');save_arrays(run/'targets.npz',y=y)
    files.update({str(run/p):sha(run/p) for p in ['training.pkl','query.pkl','targets.npz']})
    write(run/'manifest.json',dict(spec=spec,recipe=mae_recipe(original['recipes']['realmlp_td']),files=files,source_directory=str(WORK),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),official_data_reads=0,scientific_fits=0))
    print(json.dumps(dict(status='frozen',synthetic_procedures=2,native_adam=4,official_data_reads=0)),flush=True)


def context():
    original=setup();spec=read(WORK/SPEC);run=Path(spec['run_directory']);m=read(run/'manifest.json');verify(m['files'])
    if m['spec']!=spec or m['source_directory']!=str(WORK) or m['recipe']!=mae_recipe(original['recipes']['realmlp_td']):raise ValueError('Frozen MAE engineering context changed')
    return run,m,pd.read_pickle(run/'training.pkl'),pd.read_pickle(run/'query.pkl'),arrays(run/'targets.npz')['y']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','control','capture','cold','parity']);p.add_argument('--checks');a=p.parse_args()
    if a.action=='prepare':prepare(a.checks)
    else:
        with patch.object(harness,'context',context),patch.object(harness,'observe_native',observe_mae_native):
            if a.action in ['control','capture']:harness.worker(a.action)
            elif a.action=='cold':harness.cold()
            else:harness.parity()
