"""Read-only final-bundle first-timestamp equality check; no challenger generation."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from bf_tap.artifacts import atomic_write_json,file_sha256
from bf_tap.config import load_yaml
from bf_tap.io import read_csv
from bf_tap.models.baseline import DualTargetBaseline
from bf_tap.offline import _load_process_sources
from bf_tap.optimization.component_export import ComponentFeatures,META,PRED,forbid_fit,read_json
from bf_tap.optimization.pseudo_history import rollout


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    active=load_yaml('configs/optimization_v0_4/active_release.yaml');root=Path(active['bundle'])
    assert file_sha256(root/'composite.json')==active['composite_sha256']
    assert file_sha256(active['test_a_zip'])==active['test_a_zip_sha256']
    config='local/runs/optimization-v0.4-r2-challenger-r1/cold_data.yaml'
    paths=load_yaml(config)['paths'];m=read_json(root/'composite.json');a=m['algorithm']
    samples=read_csv(paths['test_a_samples'],usecols=META+['tap_no'],time_columns=['reference_time'])
    samples=samples.loc[samples.reference_time==samples.reference_time.min()].copy()
    op,burden,_=_load_process_sources(paths,a['semantic'],a['features'])
    entries={};loaded={};anchor=m['anchors'][0];counter={'attempted_target_fits':0}
    with forbid_fit(counter):
        for r,c in [('OR','E09_PROCESS_CHANGE_E02'),('HR','E04')]:
            identity=anchor['components'][c];directory=root/identity['path']
            assert file_sha256(directory/'bundle.json')==identity['sha256']
            model=DualTargetBaseline.load(directory)
            entries[r]=dict(cutoff=anchor['cutoff'],component=c,variant='R2',
                bundle_sha256=identity['sha256'],history_snapshot_sha256=file_sha256(directory/'history_snapshot.csv'))
            loaded[r]=(model,model.load_history_snapshot())
        u0,u1,audit,_=rollout(samples,entries,loaded,ComponentFeatures(a,op,burden))
        spec=importlib.util.spec_from_file_location('old_cold','scripts/optimization_v4_cold_predict.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        old.sample_metadata=lambda path:samples[META].copy()
        expected=old.predict(root,config,'test_a').set_index('sample_id').sort_index()
        for p in (u0,u1):
            assert np.array_equal(expected[PRED].to_numpy(),p.set_index('sample_id').sort_index()[PRED].to_numpy())
    assert file_sha256(active['test_a_zip'])==active['test_a_zip_sha256']
    atomic_write_json(output/'validation.json',dict(status='PASS',first_timestamp=str(samples.reference_time.min()),
        rows=len(samples),exact_U0_U1_original_loader=True,base_composite_sha256=active['composite_sha256'],
        R2_zip_sha256=active['test_a_zip_sha256'],**counter,test_labels_read=False,challenger_generated=False,audit=audit))
    print('PASS final R2 first test timestamp: exact U0/U1/original-loader equality, zero fit',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);main(p.parse_args().output)
