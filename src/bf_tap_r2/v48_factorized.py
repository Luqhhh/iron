"""V48 changes only the epoch cap; actual old checkpoints captured in new fits."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import TARGETS
from .v30_deep_kernel import select_weight
from .v47_factorized import FactorRegressor, unlabeled


class LongFactorRegressor(FactorRegressor):
    def train(self, epochs, validation=None, prefix_path=None):
        if epochs < 1: raise ValueError('Positive epoch count required')
        previous = json.loads(Path(prefix_path).read_text()) if prefix_path is not None else None
        if previous is not None:
            restored = dict(self.settings, max_epochs=previous['settings']['max_epochs'])
            if restored != previous['settings'] or previous['recipe'] != self.recipe:
                raise ValueError('Prefix model differs beyond epoch cap')
            if epochs < previous['metadata']['selected_epoch']:
                raise ValueError('New fit cannot reach the old selected prefix')
        prefix_state = None
        rng=np.random.default_rng(self.settings['random_seed'])
        best,stale,best_epoch,best_state=float('inf'),0,0,None
        self.history_=[]
        for epoch in range(1,epochs+1):
            order=rng.permutation(len(self.y_train_));loss_sum=0.
            for start in range(0,len(order),self.settings['batch_size']):
                idx=order[start:start+self.settings['batch_size']]
                self.optimizer_.zero_grad(set_to_none=True)
                pred=self.model_(*(v[idx] for v in self.x_train_))
                mae=(pred-self.y_train_[idx]).abs().mean()
                loss=mae+self.settings['coefficient_second_difference_penalty']*self.model_.roughness()
                if not torch.isfinite(loss): raise ValueError('Nonfinite factorized loss')
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(),self.settings['gradient_norm_clip'],error_if_nonfinite=True)
                self.optimizer_.step();loss_sum+=float(mae.detach())*len(idx)
            row={'epoch':epoch,'standardized_training_mae':loss_sum/len(order)}
            if validation is not None:
                value=float(np.abs(self.predict(validation[0])-validation[1]).mean()/self.std_)
                row['calibration_standardized_mae']=value
                if value < best-self.settings['min_delta_standardized_mae']:
                    best,best_epoch,stale=value,epoch,0
                    best_state=deepcopy(self.model_.state_dict())
                else: stale+=1
            self.history_.append(row)
            if previous is not None and epoch == previous['metadata']['selected_epoch']:
                prefix_state = {k:v.detach().numpy().copy().tolist() for k,v in self.model_.state_dict().items()}
            if validation is not None and stale>=self.settings['patience']: break
        if validation is not None:
            if best_state is None: raise ValueError('No finite selected epoch')
            self.model_.load_state_dict(best_state)
        else: best_epoch=epoch
        self.selected_epoch_,self.stopped_epoch_=best_epoch,epoch
        if previous is not None:
            old_meta = previous['metadata']
            if len(self.history_) < old_meta['stopped_epoch'] or prefix_state is None:
                raise ValueError('Old trace prefix not reached')
            prefix_meta = self.metadata()
            prefix_meta.update(selected_epoch=old_meta['selected_epoch'], stopped_epoch=old_meta['stopped_epoch'],
                               history=deepcopy(self.history_[:old_meta['stopped_epoch']]))
            if prefix_state != previous['state']:
                raise ValueError('Old parameter prefix not reproduced exactly')
            clean = lambda m: {k:v for k,v in m.items() if k != 'peak_rss_mib'}
            if clean(prefix_meta) != clean(old_meta):
                raise ValueError('Old preprocessing/normalization/trace prefix not reproduced exactly')
            self.prefix_payload_ = {'recipe':self.recipe, 'settings':dict(self.settings,max_epochs=previous['settings']['max_epochs']),
                                    'metadata':prefix_meta, 'state':prefix_state}
        return best_epoch

    def save_prefix(self, path):
        if not hasattr(self, 'prefix_payload_'):
            raise ValueError('No captured old prefix')
        with Path(path).open('x') as stream:
            json.dump(self.prefix_payload_, stream, allow_nan=False)


def fit_partition(fitting,calibration,outer_training,query,target,recipe,settings,calibration_base,grid,old_directory=None):
    for part in (fitting,calibration,outer_training,query):
        if part.sample_id.duplicated().any(): raise ValueError('Duplicate sample IDs')
    if set(fitting.sample_id)&set(calibration.sample_id): raise ValueError('Fitting/calibration overlap')
    if set(outer_training.sample_id)!=set(fitting.sample_id)|set(calibration.sample_id):
        raise ValueError('Incorrect training union')
    if set(outer_training.sample_id)&set(query.sample_id): raise ValueError('Outer query overlaps training')
    if any(t in query for t in TARGETS): raise ValueError('Query labels must be removed')
    old_directory = Path(old_directory) if old_directory is not None else None
    selector=LongFactorRegressor(recipe,settings).initialize(fitting,fitting[target].to_numpy())
    clean_cal=unlabeled(calibration)
    epoch=selector.train(settings['max_epochs'],(clean_cal,calibration[target].to_numpy()),
                         prefix_path=old_directory/'calibration_model.json' if old_directory is not None else None)
    cp=selector.predict(clean_cal)
    weight,losses=select_weight(calibration[target],calibration_base,cp,grid)
    final=LongFactorRegressor(recipe,settings).initialize(outer_training,outer_training[target].to_numpy())
    final.train(epoch,prefix_path=old_directory/'model.json' if old_directory is not None else None);prediction=final.predict(query);final.calibration_model_=selector
    return final,prediction,{'weight':weight,'calibration_mae_by_weight':losses,
                            'calibration':selector.metadata(),'refit':final.metadata()},cp
