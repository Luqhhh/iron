"""Explicit zero-fit reader for two immutable pre-fix synthetic snapshots."""
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from . import realmlp_state_adapter as adapter
from .ema_nested_residual import read,write,sha,verify
from .realmlp_native_audit import cold_audit

WORK=Path(__file__).resolve().parents[2]
MAIN=Path('/home/lux1/iron')
ORIGINAL=MAIN/'local/runs/realmlp-native-admission-20261004/engineering-r1'
RUN=MAIN/'local/runs/realmlp-native-admission-20261004/recovery-r1'


def load_original(path,*,expected_identity,expected_role,expected_recipe):
    path=Path(path).resolve()
    if path!=ORIGINAL/'capture'/(expected_role+'.pkl'):
        raise ValueError('Recovery only admits two exact original synthetic states')
    m=read(RUN/'manifest.json');verify(m['files'])
    adapter.validate_identity(expected_identity,path.parent)
    record=read(path.with_suffix('.pkl.json'));h=record['header']
    if (h['version']!=adapter.VERSION or h['identity']!=expected_identity or h['role']!=expected_role
            or h['recipe']!=expected_recipe or h['adapter_source_sha256']!=m['original_writer_sha256']
            or h['original_source_sha256']!=sha(adapter.original.__file__)
            or h['runtime_versions']!={p:importlib.metadata.version(p) for p in adapter.DEPENDENCIES}
            or record['state_sha256']!=sha(path)):
        raise ValueError('Original snapshot identity/hash/runtime mismatch')
    with path.open('rb') as f:payload=pickle.load(f)
    if set(payload)!={'header','estimator','encoder'} or payload['header']!=h:
        raise ValueError('Original snapshot payload differs')
    adapter.validate_native(payload['estimator'],h['recipe'],h['role'],h['metadata']['selected_epoch'])
    np.testing.assert_array_equal(payload['encoder'].medians_,h['encoder_medians'])
    np.testing.assert_array_equal(payload['encoder'].categories_,h['encoder_categories'])
    return payload


def run():
    import psutil
    import torch
    import pytorch_lightning.core.module as lightning_module
    if sys.version_info[:2]!=(3,12) or any(os.environ.get(k)!='1' for k in
        ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']):
        raise ValueError('Locked Python3.12 and serial numerical environment required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if psutil.virtual_memory().available/2**20<3072:raise ValueError('Entry memory gate failed')
    if RUN.exists():raise FileExistsError('Recovery is append-only')
    if subprocess.check_output(['git','status','--porcelain'],cwd=WORK,text=True).strip():
        raise ValueError('Clean committed recovery snapshot required')
    m=read(ORIGINAL/'manifest.json');files=dict(m['files'])
    from .realmlp_native_admission import source_hashes
    checks=MAIN/'local/runs/realmlp-native-admission-20261004/checks-r2/checks.json'
    c=read(checks)
    if (c['status']!='passed' or c['actual_exit_code']!=0 or c['source_hashes']!=source_hashes()
            or c['junit_sha256']!=sha(c['junit'])):
        raise ValueError('Exact-source recovery tests required')
    files[str(checks)]=sha(checks);files[c['junit']]=sha(c['junit'])
    for p in list(ORIGINAL.rglob('*'))+[WORK/'docs/realmlp_native_admission/RECOVERY.md',Path(lightning_module.__file__)]:
        if p.is_file():files[str(p)]=sha(p)
    for p in (WORK/'src').rglob('*.py'):files[str(p)]=sha(p)
    original_writer=Path(m['source_directory'])/'src/bf_tap_r2/realmlp_state_adapter.py'
    if sha(original_writer)!=m['spec']['adapter_source_sha256']:
        raise ValueError('Original writer source changed')
    failure=read(ORIGINAL/'failure.json')
    if failure['exits']!={'control':0,'capture':0,'cold':1}:
        raise ValueError('Unexpected original failure stage')
    if 'Native post-fit cleanup incomplete' not in (ORIGINAL/'cold.log').read_text():
        raise ValueError('Recovery does not apply to this failure')
    verify(files);RUN.mkdir(parents=True)
    write(RUN/'manifest.json',dict(files=files,original_writer_sha256=sha(original_writer),
        recovery_reader_sha256=sha(__file__),new_validator_sha256=sha(adapter.__file__),
        original_manifest_sha256=sha(ORIGINAL/'manifest.json'),source_directory=str(WORK),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        new_fits=0,new_optimizers=0))
    frame,query=pd.read_pickle(ORIGINAL/'training.pkl'),pd.read_pickle(ORIGINAL/'query.pkl')
    with np.load(ORIGINAL/'targets.npz',allow_pickle=False) as a:y=a['y'].copy()
    control=read(ORIGINAL/'control/complete.json');capture=read(ORIGINAL/'capture/complete.json')
    for k in ['metadata','native_trace','selected_epoch','native_updates','native_optimizer_runs']:
        if control[k]!=capture[k]:raise ValueError('Original/capture parity failed: '+k)
    if control['parameter_states']['refit']!=capture['parameter_states']['refit']:
        raise ValueError('Original/capture parameter parity failed')
    with np.load(ORIGINAL/'control/predictions.npz',allow_pickle=False) as a,np.load(ORIGINAL/'capture/predictions.npz',allow_pickle=False) as b:
        np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(a['refit'],b['refit'])
        np.testing.assert_array_equal(a['ids'],query.sample_id.to_numpy(str))
        result=cold_audit(ORIGINAL/'capture',frame,query,y,m['recipe'],capture['identity'],capture,
            {k:b[k].copy() for k in ['selection','refit']},loader=load_original)
    import resource
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if peak>1536:raise ValueError('Peak memory gate failed')
    verify(files)
    write(RUN/'complete.json',dict(status='passed',native_cold=result,original_capture_prediction_difference=0,
        original_optimizer_runs=control['native_optimizer_runs']+capture['native_optimizer_runs'],
        original_native_updates=control['native_updates']+capture['native_updates'],
        original_failure_preserved=True,new_fits=0,new_optimizers=0,peak_rss_mib=peak,pid=os.getpid(),
        manifest_sha256=sha(RUN/'manifest.json')))
    print(json.dumps(dict(status='passed',cold=result,new_fits=0)),flush=True)


if __name__=='__main__':run()
