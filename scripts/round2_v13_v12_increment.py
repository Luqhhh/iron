"""Supplement V13's frozen comparisons with the subsequently authorized V12 iron recipe.

No selection, classification, fits, releases, or alterations to original summaries.
Run only after the complete V13 development audit has passed.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from bf_tap_r2.v5_library import load_v5_training_frame,fold_vector
from bf_tap_r2.v5_spec import load_v5_spec
from bf_tap_r2.v5_resolution import nested_blend
from bf_tap_r2.v7_periodic import digest,file_hash,write_new
from bf_tap_r2.v11_quantile import load_references,load_oof

root=Path.cwd(); directory=root/'local/runs/round2-v13-ple-repair/development-r1'
spec=yaml.safe_load((root/'configs/round2_v13/SPEC.yaml').read_text())
audit=json.loads((directory/'audit-r1.json').read_text())
assert audit['status']=='passed' and audit['prediction_count']==40
frame=load_v5_training_frame(root)
folds={s:fold_vector(root,frame,s,load_v5_spec(root)) for s in spec['split_seeds']}
base,_,_,_,hashes=load_references(root,frame,folds,spec)
joint_dir=root/'local/runs/round2-v12-joint-tabm/development-r1'
joint_manifest=json.loads((joint_dir/'manifest.json').read_text())
joint_audit=json.loads((joint_dir/'audit-r1.json').read_text())
assert joint_audit['status']=='passed' and joint_audit['prediction_count']==20
assert joint_manifest['data_digest']==hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
reference={}
for seed,fv in folds.items():
    assert digest(fv.tolist())==joint_manifest['fold_digests'][str(seed)]
    p=np.full(len(frame),np.nan)
    for fold in range(5):
        key=f'joint-joint_plr001-s{seed}-f{fold}'; path=joint_dir/(key+'.npy')
        assert file_hash(path)==joint_audit['hashes'][key]
        values=np.load(path,allow_pickle=False); mask=fv==fold
        assert values.shape==(int(mask.sum()),2) and np.isfinite(values).all()
        p[mask]=values[:,0]; hashes[str(path.relative_to(root))]=file_hash(path)
    reference[seed]=.5*base[seed]['tap_iron']+.5*p
records=[]
for recipe in spec['recipes']:
    prediction,h=load_oof(root,directory.relative_to(root),frame,folds,'tap_iron',recipe)
    hashes.update(h)
    result=nested_blend(frame.tap_iron.values,folds,reference,prediction,spec['blend_grid'])
    records.append({'target':'tap_iron','recipe':recipe,'v12_iron_a50_comparison':result})
write_new(directory/'v12-increment-r1.json',{'status':'complete','role':'supplemental_only_no_frozen_decision_change',
    'reference':'0.5*A35_iron+0.5*V12_joint_plr001_iron','records':records,'hashes':hashes,
    'original_v13_audit_sha256':file_hash(directory/'audit-r1.json'),'script_sha256':file_hash(__file__),
    'model_fits':0,'selection_changed':False,'packages':0})
print(json.dumps(records,indent=2))
