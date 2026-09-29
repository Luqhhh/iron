from concurrent.futures import Future, ThreadPoolExecutor
import json

import numpy as np
import pytest

from bf_tap_r2 import rfm_run as run
from bf_tap_r2.rfm_protocol import file_hash, write_new
from test_rfm_model import sample


def test_scheduler_complete_coverage_and_worker_bound():
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sorted(run.bounded_map(pool,lambda x:x*x,range(12)))==[x*x for x in range(12)]
        with pytest.raises(ValueError):run.bounded_map(pool,lambda x:x,[],workers=5)


def test_scheduler_does_not_refill_a_group_containing_failure():
    class ReadyPool:
        def __init__(self):self.started=[]
        def submit(self,worker,job):
            self.started.append(job);f=Future()
            if job==3:f.set_exception(RuntimeError('frozen failure'))
            else:f.set_result(job)
            return f
    pool=ReadyPool()
    with pytest.raises(RuntimeError,match='frozen failure'):
        run.bounded_map(pool,lambda x:x,range(40))
    assert pool.started==[0,1,2,3]


def test_failed_phase_retains_evidence_and_refuses_reentry(tmp_path,monkeypatch):
    manifest={'workspace':str(tmp_path)}
    frame=sample(30);folds={s:np.arange(30)%5 for s in (42,3407)}
    monkeypatch.setattr(run,'_context',lambda *args:(manifest,tmp_path))
    monkeypatch.setattr(run,'reload_references',lambda m:(frame,folds,{}, {},{}))
    monkeypatch.setattr(run,'ProcessPoolExecutor',lambda **kwargs:ThreadPoolExecutor(max_workers=4))
    def failure(job):raise RuntimeError('worker stopped')
    monkeypatch.setattr(run,'_worker',failure)
    with pytest.raises(RuntimeError,match='worker stopped'):
        run.run_phase(tmp_path/'manifest.json','manifest','preflight','development')
    output=tmp_path/'development'
    assert json.loads((output/'failed.json').read_text())['type']=='RuntimeError'
    assert (output/'started.json').exists() and not (output/'complete.json').exists()
    with pytest.raises(FileExistsError):
        run.run_phase(tmp_path/'manifest.json','manifest','preflight','development')


def test_confirmation_requires_anchored_development_audit(tmp_path):
    output=tmp_path/'development';output.mkdir()
    write_new(output/'complete.json',{'test':'synthetic'})
    write_new(output/'audit.json',dict(status='passed',phase='development',manifest_sha256='manifest',
        new_audit_fits=0,phase_complete_sha256=file_hash(output/'complete.json'),records=[]))
    sha=file_hash(output/'audit.json')
    assert run.development_records(tmp_path,'manifest',sha)==[]
    with pytest.raises(ValueError,match='manifest'):
        run.development_records(tmp_path,'other',sha)
    (output/'complete.json').write_text('{}')
    with pytest.raises(ValueError,match='external'):
        run.development_records(tmp_path,'manifest',sha)
