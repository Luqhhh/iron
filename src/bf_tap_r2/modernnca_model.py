"""Partition-local ModernNCA with an exact default author PLR/linear encoder.

Adapted from TALENT 1b973adffa4203c3f62c4949957b1ba3efbe60d3;
Copyright (c) 2024 Siyang Liu, Haorun Cai, Qile Zhou, Han-Jia Ye.
See licenses/TALENT-MIT.txt. Group masks extend the author's self mask.
This module contains no official-data runner or release entrypoint.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import FEATURES, TARGETS
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest, file_hash, write_new

ARMS = ('FIXED_ENCODER', 'LEARNED_ENCODER')


@dataclass(frozen=True)
class Settings:
    dim: int = 128
    frequencies: int = 77
    embedding: int = 34
    frequency_scale: float = .04431360576139521
    temperature: float = 1.
    sample_rate: float = .5
    learning_rate: float = .01
    weight_decay: float = .0002
    random_seed: int = 42
    batch_size: int = 256
    max_epochs: int = 240
    patience: int = 25
    min_delta: float = 0.

    def __post_init__(self):
        for key in ('dim', 'frequencies', 'embedding', 'batch_size', 'max_epochs', 'patience'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('Positive integer model/training settings required')
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError('Nonnegative integer seed required')
        numeric = [self.frequency_scale,self.temperature,self.sample_rate,self.learning_rate,self.weight_decay,self.min_delta]
        if not np.isfinite(numeric).all() or min(numeric[:4]) <= 0 or not self.sample_rate <= 1 or min(numeric[4:]) < 0:
            raise ValueError('Invalid finite geometry/training settings')


class Periodic(nn.Module):
    def __init__(self, features, frequencies, scale):
        super().__init__()
        self.frequencies = nn.Parameter(torch.normal(0., scale, (features, frequencies)))

    def forward(self, x):
        angles = 2 * torch.pi * self.frequencies[None] * x[..., None]
        return torch.cat([angles.cos(), angles.sin()], -1)


class NeighborNetwork(nn.Module):
    def __init__(self, settings, categories):
        super().__init__()
        self.settings = settings
        # Preserve the author's initialization order: encoder before PLR.
        self.encoder = nn.Linear(len(FEATURES)*settings.embedding + categories, settings.dim)
        self.num_embeddings = nn.Sequential(Periodic(len(FEATURES),settings.frequencies,settings.frequency_scale),
            nn.Linear(2*settings.frequencies,settings.embedding),nn.ReLU())
        self.double()

    def encode(self, x):
        numeric = self.num_embeddings(x[:, :len(FEATURES)]).flatten(1)
        return self.encoder(torch.cat([numeric, x[:, len(FEATURES):]], 1))

    def forward(self, x, y, candidates, labels, *, training, groups=None):
        if training:
            count = int(len(candidates)*self.settings.sample_rate)
            if count < 1:
                raise ValueError('Nonempty sampled external neighbor pool required')
            indices = torch.randperm(len(candidates))[:count]
            candidates, labels = candidates[indices], labels[indices]
        q, bank = self.encode(x), self.encode(candidates)
        if training:
            if y is None or len(y) != len(x):
                raise ValueError('Training batch responses required')
            bank, labels = torch.cat([q,bank]), torch.cat([y,labels])
        distances = torch.cdist(q,bank,p=2)/self.settings.temperature
        if training:
            mask = torch.eye(len(x),dtype=torch.bool) if groups is None else groups[:,None] == groups[None,:]
            # Avoid an autograd-visible in-place mutation of a distance tensor.
            distances = distances.masked_fill(torch.cat([mask,torch.zeros((len(x),len(bank)-len(x)),dtype=torch.bool)],1),torch.inf)
        weights = (-distances).softmax(1)
        return weights @ labels


def clean(frame):
    return frame[['sample_id','spout_no',*FEATURES]].copy()


def validate_features(frame):
    if (not len(frame) or frame.sample_id.isna().any() or frame.sample_id.astype(str).duplicated().any()
            or not np.isfinite(frame[list(FEATURES)].to_numpy(float)).all()
            or frame.spout_no.isna().any()):
        raise ValueError('Nonempty unique finite feature rows required')
    if any(t in frame for t in TARGETS):
        raise ValueError('Feature input must not carry target columns')


def feature_groups(frame):
    return pd.util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy().view(np.int64)


def validate_pair(training, query):
    validate_features(training);validate_features(query)
    if set(training.sample_id.astype(str)) & set(query.sample_id.astype(str)):
        raise ValueError('Training/query IDs overlap')
    if set(feature_groups(training)) & set(feature_groups(query)):
        raise ValueError('Training/query feature groups overlap')


def external_pool(groups, batch):
    keep = ~np.isin(groups, groups[batch])
    return np.flatnonzero(keep)


class NeighborRegressor:
    def __init__(self, arm, settings=Settings()):
        if arm not in ARMS:
            raise ValueError('Unknown neighbor arm')
        self.arm, self.settings = arm, settings

    def initialize(self, frame, y):
        validate_features(frame)
        y = np.asarray(y,float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or not y.std() > 0:
            raise ValueError('Finite variable response vector required')
        self.fit_frame_ = frame.copy();self.fit_y_ = y.copy()
        self.preprocessor_ = NumericPreprocessor(structure='raw_tabm').fit(frame)
        self.mean_, self.std_ = float(y.mean()), float(y.std())
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.settings.random_seed)
            self.model_ = NeighborNetwork(self.settings,self.preprocessor_.n_spout_categories_)
            self.training_rng_ = torch.get_rng_state().clone()
        self.initial_state_ = {k:v.detach().numpy().copy() for k,v in self.model_.state_dict().items()}
        self.model_.requires_grad_(self.arm == 'LEARNED_ENCODER')
        self.optimizer_ = (torch.optim.AdamW(self.model_.parameters(),lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay) if self.arm == 'LEARNED_ENCODER' else None)
        self.x_ = self.inputs(frame);self.y_ = torch.as_tensor((y-self.mean_)/self.std_,dtype=torch.float64)
        self.groups_ = feature_groups(frame)
        self.actual_epochs_ = 0;self.optimizer_steps_ = 0;self.history_ = [];self.calibration_predictions_ = []
        return self

    def inputs(self, frame):
        numeric, cat = self.preprocessor_.transform_mlp(frame)
        return torch.as_tensor(np.concatenate([numeric,cat],1),dtype=torch.float64)

    def predict(self, frame):
        validate_features(frame)
        self.model_.eval()
        with torch.no_grad():
            p = self.model_(self.inputs(frame),None,self.x_,self.y_,training=False)
        values = p.numpy()*self.std_ + self.mean_
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError('Invalid neighbor prediction')
        return values

    def train(self, epochs, calibration=None):
        if self.actual_epochs_ or self.history_:
            raise ValueError('Partition training is one-shot')
        if type(epochs) is not int or not 1 <= epochs <= self.settings.max_epochs:
            raise ValueError('Epochs exceed fixed recipe')
        if calibration is not None:
            query, truth = calibration;validate_pair(self.fit_frame_,query)
            truth = np.asarray(truth,float)
            if truth.shape != (len(query),) or not np.isfinite(truth).all():
                raise ValueError('Invalid calibration response')
        best, selected, stale = float('inf'), 0, 0
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.training_rng_)
            for epoch in range(1,epochs+1):
                if self.optimizer_ is not None:
                    self.model_.train();order = torch.randperm(len(self.x_)).numpy()
                    for start in range(0,len(order),self.settings.batch_size):
                        batch = order[start:start+self.settings.batch_size];pool = external_pool(self.groups_,batch)
                        self.optimizer_.zero_grad(set_to_none=True)
                        pred = self.model_(self.x_[batch],self.y_[batch],self.x_[pool],self.y_[pool],training=True,
                            groups=torch.as_tensor(self.groups_[batch]))
                        loss = (pred-self.y_[batch]).square().mean()
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite neighbor training loss')
                        loss.backward();self.optimizer_.step();self.optimizer_steps_ += 1
                self.actual_epochs_ = epoch
                if calibration is not None:
                    prediction = self.predict(query);error = float(np.abs(prediction-truth).mean())
                    self.calibration_predictions_.append(prediction.copy())
                    self.history_.append(dict(epoch=epoch,mae=error))
                    if error < best-self.settings.min_delta:
                        best, selected, stale = error,epoch,0
                        self.selected_state_ = {k:v.detach().numpy().copy() for k,v in self.model_.state_dict().items()}
                    else:stale += 1
                    if stale >= self.settings.patience or self.optimizer_ is None:break
        self.selected_epoch_ = selected if calibration is not None else epochs
        if not self.selected_epoch_:
            raise ValueError('No finite selected epoch')
        if calibration is not None:
            self.model_.load_state_dict({k:torch.as_tensor(v) for k,v in self.selected_state_.items()})
        return self.selected_epoch_

    def save(self, path):
        path = Path(path)
        if not self.actual_epochs_ or not self.selected_epoch_:
            raise ValueError('Only completed partition states may be saved')
        metadata = dict(format='modernnca-partition-v1',arm=self.arm,settings=asdict(self.settings),
            preprocessor=self.preprocessor_.metadata(),fit_ids=self.fit_ids_,mean=self.mean_,std=self.std_,
            actual_epochs=self.actual_epochs_,selected_epoch=self.selected_epoch_,optimizer_steps=self.optimizer_steps_,
            history=self.history_,fit_frame_digest=digest(pd.util.hash_pandas_object(self.fit_frame_,index=True).tolist()),
            fit_y_digest=digest(self.fit_y_.tolist()))
        arrays = {f'state::{k}':v.detach().numpy().copy() for k,v in self.model_.state_dict().items()}
        arrays.update({f'initial::{k}':v for k,v in self.initial_state_.items()})
        arrays.update(bank_x=self.x_.numpy(),bank_y=self.y_.numpy(),calibration=np.asarray(self.calibration_predictions_,dtype=float),
            metadata=np.asarray(json.dumps(metadata,sort_keys=True)))
        with path.open('xb') as stream:np.savez(stream,**arrays)
        return file_hash(path)


def fit_pair(fitting, calibration, outer_training, target, settings):
    if target not in TARGETS:
        raise ValueError('Unknown response')
    ids = [set(p.sample_id.astype(str)) for p in (fitting,calibration,outer_training)]
    if ids[0]&ids[1] or ids[2] != ids[0]|ids[1]:
        raise ValueError('Invalid inner/outer row partition')
    combined=pd.concat([fitting,calibration]).set_index('sample_id').sort_index()
    if not combined.equals(outer_training.set_index('sample_id').sort_index()):
        raise ValueError('Inner/outer feature/response content mismatch')
    validate_pair(clean(fitting),clean(calibration))
    inner, outer = {}, {}
    for arm in ARMS:
        inner[arm] = NeighborRegressor(arm,settings).initialize(clean(fitting),fitting[target].to_numpy())
        epoch = inner[arm].train(settings.max_epochs,(clean(calibration),calibration[target].to_numpy()))
        outer[arm] = NeighborRegressor(arm,settings).initialize(clean(outer_training),outer_training[target].to_numpy())
        outer[arm].train(epoch)
    return inner,outer
