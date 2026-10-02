"""Matched fixed/adaptive Laplace location heads on the frozen V33 backbone."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .v33_mixture import MixtureRegressor

ARMS = ('LAPLACE_FIXED', 'LAPLACE_SCALE')


def laplace_nll(y, location, scale):
    if y.ndim != 1 or location.shape != (len(y),1) or scale.shape != location.shape:
        raise ValueError('Aligned scalar Laplace location and scale required')
    if not all(torch.isfinite(v).all() for v in (y,location,scale)) or not (scale>0).all():
        raise ValueError('Finite Laplace inputs and positive scale required')
    return ((y[:,None]-location).abs()/scale + scale.log() + math.log(2)).mean()


class LaplaceRegressor(MixtureRegressor):
    def __init__(self, arm, settings):
        if arm not in ARMS:
            raise ValueError('Unknown frozen Laplace arm')
        super().__init__('GAUSS1',settings)
        self.arm = arm

    def loss(self, y, output):
        _, location, scale = output
        if self.arm == 'LAPLACE_FIXED':
            scale = torch.full_like(scale,self.settings['initial_scale'])
        return laplace_nll(y,location,scale)

    def train(self, epochs, validation=None):
        # Same data order, optimizer, stopping and fresh-refit rules as V33.
        rng = np.random.default_rng(self.settings['random_seed'])
        best, stale, best_epoch, best_state = float('inf'), 0, 0, None
        self.history_ = []
        for epoch in range(1,epochs+1):
            self.model_.train()
            order, loss_sum = rng.permutation(len(self.y_train_)), 0.
            for start in range(0,len(order),self.settings['batch_size']):
                idx = order[start:start+self.settings['batch_size']]
                self.optimizer_.zero_grad(set_to_none=True)
                loss = self.loss(self.y_train_[idx],self.model_(self.x_train_[idx]))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite Laplace likelihood')
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(),self.settings['gradient_norm_clip'],error_if_nonfinite=True)
                self.optimizer_.step()
                loss_sum += float(loss.detach())*len(idx)
            row = dict(epoch=epoch,negative_log_likelihood=loss_sum/len(order))
            if validation is not None:
                value = float(np.abs(self.predict(validation[0])-validation[1]).mean()/self.std_)
                row['calibration_standardized_mae'] = value
                if value < best-self.settings['min_delta_standardized_mae']:
                    best,best_epoch,stale = value,epoch,0
                    best_state = deepcopy(self.model_.state_dict())
                else:
                    stale += 1
            self.history_.append(row)
            if validation is not None and stale >= self.settings['patience']:
                break
        if validation is not None:
            if best_state is None:
                raise ValueError('No finite selected epoch')
            self.model_.load_state_dict(best_state)
        else:
            best_epoch = epoch
        self.selected_epoch_,self.stopped_epoch_ = best_epoch,epoch
        return best_epoch

    def metadata(self):
        return dict(super().metadata(),laplace_arm=self.arm,likelihood='Laplace',
                    point_prediction='location_equals_conditional_median')

    def save(self,path):
        payload = dict(recipe=self.recipe,laplace_arm=self.arm,likelihood='Laplace',
            settings=self.settings,preprocessing=self.preprocessor_.metadata(),
            mean=self.mean_,std=self.std_,state=self.model_.state_dict())
        with Path(path).open('xb') as stream:
            torch.save(payload,stream)

    @classmethod
    def load(cls,path):
        payload = torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('likelihood') != 'Laplace' or payload.get('laplace_arm') not in ARMS:
            raise ValueError('Explicit Laplace checkpoint identity required')
        base = MixtureRegressor.load(path)
        model = cls(payload['laplace_arm'],base.settings)
        model.__dict__.update(base.__dict__)
        return model
