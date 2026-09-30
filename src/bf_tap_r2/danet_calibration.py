"""Group-safe raw-MAE epoch selection and two fresh whole-training refits."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import TARGETS
from .dnnr_model import validate_frame
from .dnnr_ledger import write_new,file_hash
from .v3_4_bags import group_safe_inner_folds
from .danet_model import ARMS,Regressor,Settings


@dataclass
class Pair:
    inner:dict
    outer:dict
    receipt:dict

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);hashes={}
        for role,members in (('inner',self.inner),('outer',self.outer)):
            for arm,model in members.items():
                path=directory/f'{role}-{arm}.npz';hashes[path.name]=model.save(path)
        path=directory/'receipt.json'
        write_new(path,dict(format='danet-matched-pair-v1',receipt=self.receipt,model_hashes=hashes))
        hashes[path.name]=file_hash(path)
        return hashes


def fit_pair(training,y,target,*,settings=None,observer=None,resource_upper_bound=False):
    validate_frame(training);y=np.asarray(y,dtype=np.float64)
    if target not in TARGETS or y.shape!=(len(training),) or not np.isfinite(y).all():raise ValueError('Finite aligned single target required')
    if type(resource_upper_bound) is not bool:raise ValueError('Explicit resource-only upper-bound mode required')
    settings=settings or Settings();observer=observer or (lambda *_:None)
    assignment=group_safe_inner_folds(training,n_splits=5,seed=42);mask=assignment['fold']!=0
    fitting,calibration=training.loc[mask].reset_index(drop=True),training.loc[~mask].reset_index(drop=True)
    inner,outer={},{ }
    def reserve(role,arm):
        return lambda kind,payload:observer(kind,dict(payload,role=role,arm=arm,target=target))
    for arm in ARMS:
        inner[arm]=Regressor(arm,settings).fit(fitting,y[mask],calibration=(calibration,y[~mask]),observer=reserve('inner',arm))
    selected={arm:model.selected_epoch_ for arm,model in inner.items()}
    for arm in ARMS:
        epochs=settings.max_epochs if resource_upper_bound else selected[arm]
        outer[arm]=Regressor(arm,settings).fit(training,y,epochs=epochs,observer=reserve('outer',arm))
    if inner[ARMS[0]].initial_state_digest_!=inner[ARMS[1]].initial_state_digest_ or outer[ARMS[0]].initial_state_digest_!=outer[ARMS[1]].initial_state_digest_:
        raise ValueError('Matched arms did not share initial state')
    return Pair(inner,outer,dict(target=target,inner_seed=42,held_fold=0,inner_folds=assignment['fold'].tolist(),
        inner_group_hash=assignment['group_hash'],inner_fold_hash=assignment['inner_fold_hash'],
        fitting_ids=fitting.sample_id.astype(str).tolist(),calibration_ids=calibration.sample_id.astype(str).tolist(),
        outer_ids=training.sample_id.astype(str).tolist(),selected_epochs=selected,resource_upper_bound=resource_upper_bound,
        outer_epochs={a:m.actual_epochs_ for a,m in outer.items()},estimator_runs=4,optimizer_runs=4,
        optimizer_epochs=sum(m.actual_epochs_ for members in (inner,outer) for m in members.values()),
        selection='first_minimum_raw_MAE',outer_refit='fresh_encoder_target_scale_network_and_optimizer',outer_query_labels='absent'))
