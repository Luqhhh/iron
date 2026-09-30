"""Current DE3 incumbent overlay on four-seed native references; zero fitting.

A completed, externally anchored overlay audit is mandatory. The historical
two-seed DE3 result cannot silently stand in for derived-seed reference columns.
"""
from pathlib import Path

import numpy as np

from .rfm_reference import load_reference_cache as native_loader
from .rfm_freeze import load_anchored
from .ptarl_protocol import file_hash
from .v7_periodic import digest

INCUMBENT='DE3_IRON_USER_REQUESTED'
SCORE=96.3749
SEEDS=(42,3407,7777,12011)


def load_current_reference(native_root,overlay_root,overlay_sha256):
    root=Path(overlay_root).resolve()
    complete=load_anchored(root/'complete.json',overlay_sha256)
    if (complete.get('candidate')!=INCUMBENT or complete.get('parent')!='V32_TIME_A60V7_50'
            or complete.get('endpoint')!='maximum(parent_iron+0.5*(mean(J42,J104729,J130363)-J42),0)'
            or set(complete.get('iron_columns',{}))!={str(s) for s in SEEDS}
            or complete.get('training_seeds')!=[42,104729,130363]):
        raise ValueError('Complete four-seed current DE3 reference required')
    audit=load_anchored(root/'audit.json',complete['audit_sha256'])
    if (audit.get('status')!='passed' or audit.get('candidate')!=INCUMBENT
            or audit.get('verified_split_seeds')!=list(SEEDS) or audit.get('new_audit_fits')!=0
            or audit.get('scope')!='incumbent_reference_only_not_DE3_promotion'):
        raise ValueError('Passed four-seed incumbent reference audit required')
    # Inspect overlay endpoints before the native loader can read training rows.
    hashes={'complete.json':overlay_sha256,'audit.json':complete['audit_sha256']}
    values={}
    for seed in SEEDS:
        record=complete['iron_columns'][str(seed)]
        path=root/record['path']
        if not path.resolve().is_relative_to(root) or file_hash(path)!=record['sha256']:
            raise ValueError('Incumbent endpoint artifact changed')
        values[seed]=np.load(path,allow_pickle=False)
        hashes[record['path']]=record['sha256']
    for name,sha in audit['artifact_hashes'].items():
        path=root/name
        if not path.resolve().is_relative_to(root) or file_hash(path)!=sha:
            raise ValueError('Incumbent reference audited artifact changed')
        hashes[name]=sha
    frame,folds,current,historical,native_audit=native_loader(native_root)
    if (complete['row_ids_digest']!=digest(frame.sample_id.tolist())
            or complete['native_data_digest']!=native_audit['data_digest']
            or complete['fold_digests']!={str(s):digest(folds[s].tolist()) for s in SEEDS}):
        raise ValueError('Current/native reference data or split identity mismatch')
    for seed in SEEDS:
        iron=values[seed]
        if iron.shape!=(len(frame),) or not np.isfinite(iron).all() or (iron<0).any():
            raise ValueError('Incomplete current incumbent iron column')
        current[seed]=dict(current[seed],tap_iron=iron.copy())
    audit=dict(native_audit,incumbent_candidate=INCUMBENT,incumbent_score_user_reported=SCORE,
        incumbent_overlay_root=str(root),incumbent_overlay_sha256=overlay_sha256,
        incumbent_overlay_hashes=hashes,incumbent_overlay_audit=audit,
        original_DE3_no_finalist_and_no_confirmation_unchanged=True)
    return frame,folds,current,historical,audit
