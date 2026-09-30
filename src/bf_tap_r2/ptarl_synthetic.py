"""Exclusive tiny synthetic paired-unit smoke and fresh-process zero-fit audit.

This entrypoint cannot load competition data or launch an official phase.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES, TARGETS
from .ptarl_execution import execute_unit, audit_unit
from .ptarl_model import clean
from .ptarl_protocol import ReservationLedger, file_hash, write_new


def private_directory(root, directory):
    root=Path(root).resolve()
    output=(root/directory).resolve()
    if not output.is_relative_to(root/'local/research') or output == root/'local/research':
        raise ValueError('Private research subdirectory required')
    return root,output


def run(root, directory):
    root,output=private_directory(root,directory)
    output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(54009)
    x=rng.normal(size=(100,len(FEATURES)))
    f=pd.DataFrame(x,columns=FEATURES)
    f['sample_id']=[f'ptarl-private-synthetic-{i}' for i in range(len(x))]
    f['spout_no']=np.arange(len(x))%2+1
    f['tap_iron']=40+4*x[:,0]+2*x[:,1]*x[:,2]
    f['tap_time_len']=12+2*np.sin(x[:,3])+x[:,4]
    f.iloc[:80].to_csv(output/'synthetic-training.csv',index=False,mode='x')
    clean(f.iloc[80:]).to_csv(output/'synthetic-query.csv',index=False,mode='x')
    training=pd.read_csv(output/'synthetic-training.csv',float_precision='round_trip')
    query=pd.read_csv(output/'synthetic-query.csv',float_precision='round_trip')
    settings=dict(random_seed=42,width=32,blocks=2,tabm_k=4,dropout=.1,embedding_dim=4,
        n_frequencies=4,lite=True,frequency_init_scale=.01,learning_rate=.001,weight_decay=.0001,
        batch_size=16,max_epochs=4,patience=25,min_delta_standardized_mae=1e-5,
        gradient_norm_clip=5.,prototype_count=3,norm_epsilon=1e-12,
        auxiliary_weights=dict(projection=.25,diversity=.25,orthogonalization=.25))
    paths=list((root/'src/bf_tap_r2').glob('*.py'))+list((root/'tests').glob('test_ptarl*.py'))
    paths += [root/'uv.lock',root/'pyproject.toml',root/'configs/ptarl_space_calibration/SPEC.yaml']
    manifest=dict(scope='tiny_synthetic_unit_preparation_not_formal_admission',settings=settings,
        data_hashes={p.name:file_hash(p) for p in output.glob('*.csv')},
        source_hashes={str(p.relative_to(root)):file_hash(p) for p in sorted(paths)},
        tasks=[dict(target=t,seed=42,fold=0) for t in TARGETS],
        budget=dict(pair_unit=2,optimizer=12,kmeans=4),official_label_reads=0,
        official_fits=0,full_size_probes=0,packages=0,uploads=0)
    write_new(output/'manifest.json',manifest)
    ledger=ReservationLedger.create(output/'ledger',manifest['budget'])
    try:
        units=[execute_unit(task,training,query,settings,output/'units'/task['target'],ledger.root,ledger.policy_sha256)
               for task in manifest['tasks']]
        complete=dict(manifest_sha256=file_hash(output/'manifest.json'),units=units,
            ledger_policy_sha256=ledger.policy_sha256,counts=ledger.inspect())
        write_new(output/'complete.json',complete)
    except BaseException as exc:
        write_new(output/'failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(status='completed',directory=str(output),complete_sha256=file_hash(output/'complete.json'),
                counts=complete['counts'])


def audit(root, directory, expected_sha256):
    root,output=private_directory(root,directory)
    if not expected_sha256 or file_hash(output/'complete.json') != expected_sha256:
        raise ValueError('External smoke completion hash required')
    if (output/'failed.json').exists():
        raise ValueError('Failed smoke cannot pass audit')
    write_new(output/'audit-started.json',dict(complete_sha256=expected_sha256))
    try:
        complete=json.loads((output/'complete.json').read_text())
        if file_hash(output/'manifest.json') != complete['manifest_sha256']:
            raise ValueError('Synthetic manifest changed')
        manifest=json.loads((output/'manifest.json').read_text())
        for name,sha in manifest['source_hashes'].items():
            if file_hash(root/name) != sha:raise ValueError('Synthetic source changed: '+name)
        for name,sha in manifest['data_hashes'].items():
            if file_hash(output/name) != sha:raise ValueError('Synthetic inputs changed')
        training=pd.read_csv(output/'synthetic-training.csv',float_precision='round_trip')
        query=pd.read_csv(output/'synthetic-query.csv',float_precision='round_trip')
        ledger=ReservationLedger.open(output/'ledger',complete['ledger_policy_sha256'])
        counts=ledger.inspect()
        if (ledger.limits != manifest['budget'] or counts != complete['counts']
                or counts['started'] != manifest['budget'] or counts['completed'] != manifest['budget']
                or any(counts[s][k] for s in ('failed','incomplete') for k in ledger.limits)):
            raise ValueError('Synthetic reservation budget/count mismatch')
        if len(complete['units']) != 2 or {p.name for p in (output/'units').iterdir()} != set(TARGETS):
            raise ValueError('Incomplete synthetic paired coverage')
        reports=[]
        for task,anchor in zip(manifest['tasks'],complete['units'],strict=True):
            _,report=audit_unit(output/'units'/task['target'],task,training,query,manifest['settings'],
                expected_sha256=anchor['complete_sha256'],ledger_root=ledger.root)
            if report['name'] != anchor['name']:raise ValueError('Synthetic task anchor mismatch')
            reports.append(report)
        report=dict(status='passed',scope=manifest['scope'],units=reports,models_checked=12,
            maximum_difference=max(r['maximum_difference'] for r in reports),counts=counts,
            new_optimizer_calls=0,new_kmeans_calls=0,official_fits=0,packages=0,uploads=0,
            complete_sha256=expected_sha256)
        write_new(output/'audit.json',report)
    except BaseException as exc:
        write_new(output/'audit-failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','audit'])
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--complete-sha256')
    a=p.parse_args()
    result=run(Path.cwd(),a.directory) if a.action=='run' else audit(Path.cwd(),a.directory,a.complete_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
