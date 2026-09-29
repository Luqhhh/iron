"""Zero-fit native V46 reference cache verification, including Git byte identity."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import yaml
from .data import TARGETS
from .v46_beta_nll import verify_source_bytes
from .v5_library import fold_vector,load_v5_training_frame,load_column_reference
from .v5_spec import load_v5_spec
from .v5_replicate import _identity
from .v7_confirm import read_reference_fold
from .v7_periodic import digest,file_hash


def read_json(path):
    return json.loads(path.read_text())


def unique_events(path,key):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    completed=[r for r in rows if r['event']=='complete']
    result={key(row):row for row in completed}
    if len(result)!=len(completed): raise ValueError('Duplicate reference completion')
    return result


def validate_native_audits(a7, a12):
    """Original audits bind development AND confirmation: four seeds, 20 files."""
    expected7 = set()
    expected12 = set()
    for seed in [42, 3407, 7777, 12011]:
        for fold in range(5):
            if seed in [42, 3407]:
                p7 = f'development-r1/tap_time_len-tabm_plr001-s{seed}-f{fold}.npy'
                p12 = f'development-r1/joint-joint_plr001-s{seed}-f{fold}.npy'
            else:
                p7 = p12 = f'confirmation-r1/seed-{seed}-fold-{fold}.npy'
            expected7.add('local/runs/round2-v7-periodic-networks/' + p7)
            expected12.add('local/runs/round2-v12-joint-tabm/' + p12)
    if (a7.get('status') != 'PASS' or a12.get('status') != 'passed'
            or a12.get('prediction_count') != 20
            or set(a7.get('prediction_hashes', {})) != expected7
            or set(a12.get('hashes', {})) != expected12):
        raise ValueError('Passed complete original confirmation audits required')


def load_reference_cache(root):
    root=Path(root).resolve(); hashes={}; source_variants=[]
    def remember(path):
        path=Path(path); hashes[str(path.relative_to(root))]=file_hash(path)
    def verify(path,expected):
        path=Path(path); relative=str(path.relative_to(root))
        working=path.read_bytes()
        if hashlib.sha256(working).hexdigest()==expected:
            remember(path)
            return
        if relative.startswith(('src/','configs/')) and path.suffix in ('.py','.yaml','.yml'):
            blob=subprocess.check_output(['git','show','HEAD:'+relative],cwd=root)
            variant=verify_source_bytes(working,blob,expected)
            if variant!='exact': source_variants.append(dict(path=relative,frozen_sha256=expected,working_sha256=file_hash(path),git_sha256=hashlib.sha256(blob).hexdigest(),reason=variant))
        elif file_hash(path)!=expected:
            raise ValueError('Reference evidence changed: '+relative)
        remember(path)
    frame=load_v5_training_frame(root)
    seeds=[42,3407,7777,12011]; legacy=load_v5_spec(root)
    folds={s:fold_vector(root,frame,s,legacy) for s in seeds}
    data_hash=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    def manifest(directory,spec=None,producer=None,confirmation=None):
        path=root/directory/'manifest.json'; m=read_json(path); remember(path)
        if m.get('data_digest')!=data_hash: raise ValueError('Reference data identity mismatch: '+directory)
        for s,fv in folds.items():
            if str(s) in m.get('fold_digests',{}) and m['fold_digests'][str(s)]!=digest(fv.tolist()):
                raise ValueError('Reference fold identity mismatch')
        for key,value in m.get('versions',{}).items():
            if importlib.metadata.version(key)!=value: raise ValueError('Reference runtime mismatch: '+key)
        for key in ['source_hashes','dependency_code_hashes']:
            for name,sha in m.get(key,{}).items():
                p=root/name if '/' in name else root/'src/bf_tap_r2'/name
                verify(p,sha)
        for key in ['reference_hashes','reference_cache_hashes','primary_reference_hashes','v7_reference_hashes']:
            for name,sha in m.get(key,{}).items(): verify(root/name,sha)
        if spec and 'spec_sha256' in m: verify(root/spec,m['spec_sha256'])
        if producer and 'code_sha256' in m: verify(root/producer,m['code_sha256'])
        if confirmation:
            key='confirmation_sha256' if 'confirmation_sha256' in m else 'confirmation_spec_sha256'
            if key in m: verify(root/confirmation,m[key])
        return m
    # The original development loader independently checks original audits/data/
    # fold identities/fit IDs/prediction hashes and released-column arithmetic.
    from .v17_run import context
    spec17=yaml.safe_load((root/'configs/round2_v17/SPEC.yaml').read_text())
    _,dev_folds,a35,b0,old_hashes=context(root,spec17,[42,3407])
    for name,sha in old_hashes.items(): verify(root/name,sha)
    for s in dev_folds:
        if not np.array_equal(dev_folds[s],folds[s]): raise ValueError('Development fold mismatch')
    d7='local/runs/round2-v7-periodic-networks/development-r1'
    d12='local/runs/round2-v12-joint-tabm/development-r1'
    c7='local/runs/round2-v7-periodic-networks/confirmation-r1'
    c12='local/runs/round2-v12-joint-tabm/confirmation-r1'
    c8='local/runs/round2-v8-feature-attention/confirmation-r1'
    manifest(d7,'configs/round2_v7/SPEC.yaml','src/bf_tap_r2/v7_periodic.py')
    manifest(d12,'configs/round2_v12/SPEC.yaml','src/bf_tap_r2/v12_joint.py')
    m7=manifest(c7,'configs/round2_v7/SPEC.yaml','src/bf_tap_r2/v7_confirm.py')
    m12=manifest(c12,'configs/round2_v12/SPEC.yaml',confirmation='configs/round2_v12/CONFIRMATION.yaml')
    m8=manifest(c8,confirmation='configs/round2_v8/CONFIRMATION.yaml')
    verify(root/'configs/round2_v8/SPEC.yaml',m8['parent_spec_sha256'])
    for m,parent in [(m7,d7),(m12,d12),(m8,'local/runs/round2-v8-feature-attention/development-r1')]:
        verify(root/parent/'summary.json',m['development_summary_sha256'])
    verify(root/d12/'audit-r1.json',m12['development_audit_sha256'])
    if m7['selected']['target']!='tap_time_len' or m7['selected']['recipe']!='tabm_plr001': raise ValueError('V7 recipe mismatch')
    if m12['selected']!={'target':'tap_iron','recipe':'joint_plr001'}: raise ValueError('V12 recipe mismatch')
    a7,a12=read_json(root/c7/'audit-r2.json'),read_json(root/c12/'audit-r1.json')
    for p in [root/c7/'audit-r2.json',root/c12/'audit-r1.json',root/c8/'audit-r1.json',root/c8/'control.json']: remember(p)
    validate_native_audits(a7, a12)
    for name,sha in a7['prediction_hashes'].items(): verify(root/name,sha)
    for name,sha in a12['hashes'].items(): verify(root/name,sha)
    control=read_json(root/c8/'control.json')
    if control['status']!='passed': raise ValueError('V8 original baseline control failed')
    verify(root/c8/'control-baseline.npz',control['baseline_sha256'])
    with np.load(root/c8/'control-baseline.npz',allow_pickle=False) as c:
        if not np.allclose(c['time'],c['expected_time'],rtol=0,atol=1e-9): raise ValueError('V8 baseline control mismatch')
    e7=unique_events(root/c7/'fit_ledger.jsonl',lambda e:(e['seed'],e['fold']))
    e12=unique_events(root/c12/'fit_ledger.jsonl',lambda e:(e['seed'],e['fold']))
    e8=unique_events(root/c8/'fit_ledger.jsonl',lambda e:(e['seed'],e['fold']))
    for p in [root/c7/'fit_ledger.jsonl',root/c12/'fit_ledger.jsonl',root/c8/'fit_ledger.jsonl']: remember(p)
    required={(s,f) for s in [7777,12011] for f in range(5)}
    if set(e7)!=required or set(e12)!=required or set(e8)!=required: raise ValueError('Reference ledger coverage mismatch')
    ref=load_column_reference(root,frame,legacy)
    current={}; historical={}; components={}
    meta=dict(kind='v36',trial_id='v36-s1-N-0048',target='tap_time_len',line='N')
    for seed,fv in folds.items():
        parts={k:np.full(len(frame),np.nan) for k in ['v36_time','n_time','v7_time','v36_iron','v12_iron']}
        if seed in [42,3407]:
            parts['v36_time']=ref.base_for('tap_time_len',seed)
            parts['v36_iron']=ref.base_for('tap_iron',seed)
            npath=root/f'local/runs/round2-v5-error-covariance/time-n-family-r1/seed-{seed}/pred-v36-s1-N-0048.npy'
            if str(npath.relative_to(root)) not in hashes: raise ValueError('N0048 is not bound to the original primary audit')
            parts['n_time']=np.load(npath,allow_pickle=False)
        for fold in range(5):
            mask=fv==fold; ids=digest(frame.loc[~mask,'sample_id'].tolist()); rows=int(mask.sum())
            if seed in [42,3407]:
                p=root/d7/f'tap_time_len-tabm_plr001-s{seed}-f{fold}.npy'
                parts['v7_time'][mask]=np.load(p,allow_pickle=False); remember(p)
                p=root/d12/f'joint-joint_plr001-s{seed}-f{fold}.npy'
                parts['v12_iron'][mask]=np.load(p,allow_pickle=False)[:,0]; remember(p)
            else:
                p=root/f'local/runs/round2-v5-error-covariance/replication-r1/seed-{seed}/fold-{fold}.npz'
                base,n=read_reference_fold(p,_identity(frame,fv,seed,fold,meta),mask)
                # Original V7 confirmation manifest binds these V5 byte identities.
                if str(p.relative_to(root)) not in m7['reference_cache_hashes']:
                    raise ValueError('Unbound V5 derived-seed cache')
                parts['v36_time'][mask],parts['n_time'][mask]=base,n; remember(p)
                for directory,events,audit,column,name in [(c7,e7,a7['prediction_hashes'],None,'v7_time'),(c12,e12,a12['hashes'],0,'v12_iron')]:
                    event=events[seed,fold]; info=event['metadata']
                    if info['fit_ids_digest']!=ids or info['fit_rows']!=len(frame)-rows or info['optimizer_runs']!=2 or not info['selected_epoch']>0:
                        raise ValueError('Reference training/refit/epoch identity failure')
                    p=root/directory/f'seed-{seed}-fold-{fold}.npy'
                    if str(p.relative_to(root)) not in audit: raise ValueError('Unbound reference prediction')
                    values=np.load(p,allow_pickle=False)
                    values=values[:,column] if column is not None else values
                    if values.shape!=(rows,) or not np.isfinite(values).all(): raise ValueError('Reference coverage failure')
                    parts[name][mask]=values; remember(p)
                from .v9_confirm import read_reference
                p=root/c8/f'seed-{seed}-fold-{fold}.npz'
                parts['v36_iron'][mask]=read_reference(p,e8[seed,fold],np.flatnonzero(mask),len(frame)); remember(p)
        if any(v.shape!=(len(frame),) or not np.isfinite(v).all() for v in parts.values()): raise ValueError('Incomplete four-seed components')
        iron=.5*parts['v36_iron']+.5*parts['v12_iron']
        oldtime=.325*parts['v36_time']+.175*parts['n_time']+.5*parts['v7_time']
        if seed in b0 and (not np.allclose(iron,b0[seed]['tap_iron'],rtol=0,atol=1e-12)
                or not np.allclose(oldtime,b0[seed]['tap_time_len'],rtol=0,atol=1e-12)):
            raise ValueError('Original B0 component arithmetic failure')
        current[seed]=dict(tap_iron=iron,tap_time_len=.2*parts['v36_time']+.3*parts['n_time']+.5*parts['v7_time'])
        historical[seed]=dict(tap_iron=iron.copy(),tap_time_len=oldtime)
        components[seed]=parts
    for p in (root/'复赛_train').glob('*.csv'): remember(p)
    return frame,folds,current,historical,dict(status='passed',hashes=hashes,
        source_byte_variants=source_variants,all_four_seeds_verified=True,new_reference_fits=0,
        data_digest=data_hash,fold_hashes={str(s):digest(fv.tolist()) for s,fv in folds.items()})

def cache_audit(root,output):
    """Record a reference-only preflight; failure never schedules an optimizer."""
    import sys
    import traceback
    root=Path(root).resolve(); output=Path(output).resolve()
    if not output.is_relative_to(root/'local/runs/round2-v46-beta-nll'):
        raise ValueError('Private V46 run output required')
    output.mkdir(parents=True,exist_ok=False)
    spec=root/'configs/round2_v46_beta_nll/SPEC.yaml'
    record=dict(identity='V46_TIME_BETA_NLL',stage='all_four_seed_reference_cache_identity',
        specification_sha256=file_hash(spec),python=sys.version,
        official_fits=0,optimizer_runs=0,new_reference_fits=0,
        resource_probes=0,learnability_runs=0,confirmation_fits=0,packages=0,uploads=0)
    origin=root/'local/runs/round2-v8-feature-attention/confirmation-r1/manifest.json'
    if origin.is_file():
        record['historical_v8_manifest_sha256']=file_hash(origin)
        record['missing_frozen_sources']=[dict(path=name,required_sha256=sha)
            for name,sha in read_json(origin).get('source_hashes',{}).items()
            if not (root/name).is_file()]
    try:
        _,_,_,_,audit=load_reference_cache(root)
        record.update(status='reference_cache_passed',reference_audit=audit)
    except Exception as exc:
        record.update(status='stopped_cache_verification_failed',error=repr(exc),
            traceback=traceback.format_exc(),G0='cache_identity_failed',G1='not_evaluated')
    with (output/'report.json').open('x') as stream:
        json.dump(record,stream,indent=2,allow_nan=False); stream.write('\n')
    return record


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=cache_audit(Path.cwd(),args.output)
    print(json.dumps({key:result[key] for key in ['status','official_fits','optimizer_runs','new_reference_fits']}))
    return 0 if result['status']=='reference_cache_passed' else 2


if __name__=='__main__':
    raise SystemExit(main())