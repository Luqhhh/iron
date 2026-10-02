"""Paired shared/separate V33 trunks with train-standardized two-target MAE."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch
from torch import nn

from .data import FEATURES
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest
from .v33_mixture import MixtureNetwork, MixtureRegressor

ARMS = ('SHARED_MAE', 'SEPARATE_MAE')
TARGET_ORDER = ('tap_iron', 'tap_time_len')


def joint_loss(y, prediction):
    if y.ndim != 2 or y.shape[1] != 2 or prediction.shape != y.shape:
        raise ValueError('Two aligned targets required')
    if not torch.isfinite(y).all() or not torch.isfinite(prediction).all():
        raise ValueError('Finite joint inputs required')
    # Sum of two fixed b=.5 Laplace NLLs; log(2*b) is zero.
    return ((y-prediction).abs()/.5).sum(dim=1).mean()


class JointNetwork(nn.Module):
    def __init__(self, arm, settings, categories):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown joint MAE arm')
        self.arm = arm
        base = MixtureNetwork('GAUSS1', settings, categories)
        if arm == 'SHARED_MAE':
            # Original location row becomes both initial task heads.
            with torch.no_grad():
                base.head.weight[0].copy_(base.head.weight[1])
                base.head.bias[0].copy_(base.head.bias[1])
            self.networks = nn.ModuleList([base])
        else:
            self.networks = nn.ModuleList([base, deepcopy(base)])

    def forward(self, x):
        if self.arm == 'SHARED_MAE':
            iron, time, _ = self.networks[0](x)
            return torch.cat([iron, time], dim=1)
        return torch.cat([net(x)[1] for net in self.networks], dim=1)


class JointMAERegressor:
    _inputs = MixtureRegressor._inputs

    def __init__(self, arm, settings):
        if arm not in ARMS:
            raise ValueError('Unknown joint MAE arm')
        self.arm, self.settings = arm, dict(settings)

    def initialize(self, frame, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame), 2) or not np.isfinite(y).all() or not (y.std(0)>0).all():
            raise ValueError('Finite variable two-target training matrix required')
        torch.set_num_threads(1)
        torch.manual_seed(self.settings['random_seed'])
        self.preprocessor_ = NumericPreprocessor(structure='raw_mlp').fit(frame)
        self.mean_, self.std_ = y.mean(0), y.std(0)
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        self.model_ = JointNetwork(self.arm, self.settings, self.preprocessor_.n_spout_categories_).double()
        self.x_train_, self.y_train_ = self._inputs(frame), torch.as_tensor((y-self.mean_)/self.std_)
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(), lr=self.settings['learning_rate'],
                                            weight_decay=self.settings['weight_decay'])
        return self

    def train(self, epochs, validation=None):
        rng = np.random.default_rng(self.settings['random_seed'])
        best, stale, best_epoch, best_state = float('inf'), 0, 0, None
        self.history_ = []
        for epoch in range(1, epochs+1):
            self.model_.train()
            order, loss_sum = rng.permutation(len(self.y_train_)), 0.
            for start in range(0, len(order), self.settings['batch_size']):
                idx = order[start:start+self.settings['batch_size']]
                self.optimizer_.zero_grad(set_to_none=True)
                loss = joint_loss(self.y_train_[idx], self.model_(self.x_train_[idx]))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite joint loss')
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), self.settings['gradient_norm_clip'], error_if_nonfinite=True)
                self.optimizer_.step()
                loss_sum += float(loss.detach())*len(idx)
            row = dict(epoch=epoch, negative_log_likelihood=loss_sum/len(order))
            if validation is not None:
                value = float((np.abs(self.predict(validation[0])-validation[1])/self.std_).mean())
                row['calibration_standardized_mae'] = value
                if value < best-self.settings['min_delta_standardized_mae']:
                    best, best_epoch, stale = value, epoch, 0
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
        self.selected_epoch_, self.stopped_epoch_ = best_epoch, epoch
        return best_epoch

    def predict(self, frame):
        self.model_.eval()
        with torch.no_grad():
            prediction = self.model_(self._inputs(frame)).numpy()*self.std_+self.mean_
        if prediction.shape != (len(frame),2) or not np.isfinite(prediction).all():
            raise ValueError('Invalid joint predictions')
        return prediction

    def metadata(self):
        return dict(joint_mae_arm=self.arm, target_order=list(TARGET_ORDER),
            selected_epoch=self.selected_epoch_, stopped_epoch=self.stopped_epoch_,
            fit_ids_digest=digest(self.fit_ids_), fit_rows=len(self.fit_ids_),
            preprocessing=self.preprocessor_.metadata(), target_mean=self.mean_.tolist(),
            target_std=self.std_.tolist(), history=self.history_,
            parameter_count=sum(p.numel() for p in self.model_.parameters()))

    def save(self, path):
        with Path(path).open('xb') as stream:
            torch.save(dict(joint_mae_arm=self.arm, target_order=list(TARGET_ORDER), settings=self.settings,
                preprocessing=self.preprocessor_.metadata(), mean=self.mean_.tolist(), std=self.std_.tolist(),
                state=self.model_.state_dict()), stream)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('joint_mae_arm') not in ARMS or payload.get('target_order') != list(TARGET_ORDER):
            raise ValueError('Explicit joint arm and target order required')
        model = cls(payload['joint_mae_arm'], payload['settings'])
        info = payload['preprocessing']
        model.preprocessor_ = NumericPreprocessor(structure='raw_mlp')
        model.preprocessor_.means_ = np.array(info['means'])
        model.preprocessor_.stds_ = np.array(info['stds'])
        model.preprocessor_.spout_to_index_ = {int(k): v for k,v in info['spout_vocabulary'].items()}
        model.preprocessor_.n_spout_categories_ = info['n_spout_categories']
        model.model_ = JointNetwork(model.arm, model.settings, info['n_spout_categories']).double()
        model.model_.load_state_dict(payload['state'])
        model.mean_, model.std_ = np.array(payload['mean']), np.array(payload['std'])
        return model


def numpy_predict(payload, frame):
    """Independent array arithmetic; no torch forward or preprocessor methods."""
    info = payload['preprocessing']
    numeric = ((frame[list(FEATURES)].to_numpy(float)-np.array(info['means']))/np.array(info['stds'])).astype(np.float32).astype(float)
    vocabulary = {int(k):v for k,v in info['spout_vocabulary'].items()}
    cat = np.eye(info['n_spout_categories'])[[vocabulary.get(int(v),0) for v in frame.spout_no]]
    state = {k:v.detach().cpu().numpy() for k,v in payload['state'].items()}
    predictions = []
    for index in range(1 if payload['joint_mae_arm']=='SHARED_MAE' else 2):
        prefix = f'networks.{index}.'
        phase = 2*np.pi*numeric[:,:,None]*state[prefix+'embedding.periodic.weight'][None,:,:]
        periodic = np.concatenate([np.cos(phase),np.sin(phase)],axis=2)
        embedded = np.maximum(periodic @ state[prefix+'embedding.linear.weight'].T + state[prefix+'embedding.linear.bias'],0.)
        x = np.concatenate([embedded.reshape(len(frame),-1),cat],axis=1)
        for layer in (0,2):
            x = x @ state[prefix+f'trunk.{layer}.weight'].T + state[prefix+f'trunk.{layer}.bias']
            x = x*expit(x)
        heads = x @ state[prefix+'head.weight'].T + state[prefix+'head.bias']
        predictions.append(heads[:,:2] if payload['joint_mae_arm']=='SHARED_MAE' else heads[:,1:2])
    return np.concatenate(predictions,axis=1)*np.array(payload['std'])+np.array(payload['mean'])
