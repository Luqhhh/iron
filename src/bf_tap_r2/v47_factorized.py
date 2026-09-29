"""Signed low-rank spline ANOVA; distinct original feature groups only."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import resource

import numpy as np
import torch
from torch import nn

from .data import FEATURES, TARGETS
from .v7_periodic import digest
from .v30_deep_kernel import select_weight


def unlabeled(frame):
    return frame.drop(columns=[t for t in TARGETS if t in frame])


def hat_basis(values, knots):
    values, knots = np.asarray(values, float), np.asarray(knots, float)
    if not np.isfinite(values).all() or not len(knots) or not np.isfinite(knots).all():
        raise ValueError('Invalid spline inputs')
    if len(knots) == 1:
        return np.ones((len(values), 1))
    if not (np.diff(knots) > 0).all():
        raise ValueError('Knots must be unique and increasing')
    clipped = np.clip(values, knots[0], knots[-1])
    right = np.clip(np.searchsorted(knots, clipped, side='right'), 1, len(knots)-1)
    left = right-1
    t = (clipped-knots[left])/(knots[right]-knots[left])
    result = np.zeros((len(values), len(knots)))
    result[np.arange(len(values)), left] = 1-t
    result[np.arange(len(values)), right] = t
    return result


class FactorBasis:
    def __init__(self, knot_count=16):
        self.knot_count = knot_count

    def fit(self, frame):
        x = frame[list(FEATURES)].to_numpy(dtype=float)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError('Invalid training inputs')
        self.knots_ = [np.unique(np.quantile(x[:, j], np.linspace(0,1,self.knot_count)))
                       for j in range(len(FEATURES))]
        self.categories_ = sorted(int(v) for v in frame.spout_no.unique())
        self.sizes_ = [len(k) for k in self.knots_] + [len(self.categories_)]
        self.width_ = max(self.sizes_)
        self.mean_, self.std_ = x.mean(0), x.std(0)
        self.std_[self.std_ < 1e-12] = 1.
        self.centers_ = self._raw(frame).mean(0)
        self.fit_ids_digest_ = digest(frame.sample_id.astype(str).tolist())
        return self

    def _raw(self, frame):
        result = np.zeros((len(frame),len(FEATURES)+1,self.width_))
        for j, name in enumerate(FEATURES):
            result[:,j,:self.sizes_[j]] = hat_basis(frame[name].to_numpy(float), self.knots_[j])
        for k, category in enumerate(self.categories_):
            result[:,-1,k] = frame.spout_no.to_numpy() == category
        return result

    def transform(self, frame):
        if any(t in frame for t in TARGETS):
            raise ValueError('Prediction/transform frame must not carry targets')
        linear = (frame[list(FEATURES)].to_numpy(float)-self.mean_)/self.std_
        if not np.isfinite(linear).all():
            raise ValueError('Nonfinite query inputs')
        return self._raw(frame)-self.centers_, linear

    def metadata(self):
        return {'knot_count':self.knot_count,'knots':[k.tolist() for k in self.knots_],
                'categories':self.categories_,'sizes':self.sizes_,'width':self.width_,
                'mean':self.mean_.tolist(),'std':self.std_.tolist(),'centers':self.centers_.tolist(),
                'fit_ids_digest':self.fit_ids_digest_}

    @classmethod
    def restore(cls, info):
        obj=cls(info['knot_count'])
        obj.knots_=[np.array(k) for k in info['knots']]
        obj.categories_,obj.sizes_,obj.width_=info['categories'],info['sizes'],info['width']
        obj.mean_,obj.std_,obj.centers_=[np.asarray(info[k]) for k in ('mean','std','centers')]
        obj.fit_ids_digest_=info['fit_ids_digest']
        return obj


def elementary_newton(phi, degree):
    """phi is rows x original-groups x rank, never flattened basis columns."""
    power_sums = [(phi**k).sum(dim=1) for k in range(1,degree+1)]
    values = [torch.ones_like(power_sums[0])]
    for k in range(1,degree+1):
        value = torch.zeros_like(values[0])
        for i in range(1,k+1):
            value = value + ((-1)**(i+1))*values[k-i]*power_sums[i-1]
        values.append(value/k)
    return values[degree]


class FactorNetwork(nn.Module):
    def __init__(self, recipe, settings, sizes, width):
        super().__init__()
        if recipe not in ('AFM2','AHOFM4'): raise ValueError('Unknown factor recipe')
        self.orders = settings['control_orders' if recipe=='AFM2' else 'candidate_orders']
        self.sizes = list(sizes)
        p,rank = len(sizes),settings['rank_per_order']
        self.bias = nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.linear = nn.Parameter(torch.zeros(len(FEATURES),dtype=torch.float64))
        self.additive = nn.Parameter(torch.zeros(p,width,dtype=torch.float64))
        self.factors = nn.ParameterDict()
        self.rank_weights = nn.ParameterDict()
        for order in self.orders:
            generator=torch.Generator().manual_seed(settings['random_seed']+1009*order)
            self.factors[str(order)] = nn.Parameter(torch.randn(p,width,rank,dtype=torch.float64,
                                                   generator=generator)*settings['factor_initial_std'])
            self.rank_weights[str(order)] = nn.Parameter(torch.full((rank,),settings['rank_weight_initial'],
                                                                    dtype=torch.float64))

    def forward(self, basis, linear):
        result=self.bias+(linear*self.linear).sum(1)+(basis*self.additive).sum((1,2))
        for order in self.orders:
            phi=torch.einsum('npj,pjr->npr',basis,self.factors[str(order)])
            result=result+(elementary_newton(phi,order)*self.rank_weights[str(order)]).sum(1)
        return result

    def roughness(self):
        # Numeric coefficients only. Padded and categorical coordinates are excluded.
        parts=[]
        for j,size in enumerate(self.sizes[:-1]):
            if size >= 3:
                parts.append(torch.diff(self.additive[j,:size],n=2,dim=0).reshape(-1))
                for factor in self.factors.values():
                    parts.append(torch.diff(factor[j,:size],n=2,dim=0).reshape(-1))
        return torch.cat(parts).square().mean() if parts else self.bias*0


class FactorRegressor:
    def __init__(self, recipe, settings):
        self.recipe,self.settings=recipe,deepcopy(settings)

    def initialize(self, frame, y):
        y=np.asarray(y,float)
        if y.shape!=(len(frame),) or not np.isfinite(y).all() or y.std()<=0:
            raise ValueError('Invalid training target')
        torch.set_num_threads(1)
        self.preprocessor_=FactorBasis(self.settings['quantile_knots']).fit(unlabeled(frame))
        self.mean_,self.std_=float(y.mean()),float(y.std())
        self.fit_ids_=frame.sample_id.astype(str).tolist()
        self.model_=FactorNetwork(self.recipe,self.settings,self.preprocessor_.sizes_,self.preprocessor_.width_)
        self.x_train_=self._inputs(unlabeled(frame))
        self.y_train_=torch.as_tensor((y-self.mean_)/self.std_)
        self.optimizer_=torch.optim.AdamW(self.model_.parameters(),lr=self.settings['learning_rate'],
                                         weight_decay=self.settings['weight_decay'])
        return self

    def _inputs(self, frame):
        return tuple(torch.as_tensor(a,dtype=torch.float64) for a in self.preprocessor_.transform(frame))

    def train(self, epochs, validation=None):
        if epochs < 1: raise ValueError('Positive epoch count required')
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
            if validation is not None and stale>=self.settings['patience']: break
        if validation is not None:
            if best_state is None: raise ValueError('No finite selected epoch')
            self.model_.load_state_dict(best_state)
        else: best_epoch=epoch
        self.selected_epoch_,self.stopped_epoch_=best_epoch,epoch
        return best_epoch

    def predict(self, frame):
        with torch.no_grad(): prediction=self.model_(*self._inputs(frame)).numpy()*self.std_+self.mean_
        if prediction.shape!=(len(frame),) or not np.isfinite(prediction).all():
            raise ValueError('Invalid factor prediction')
        return prediction

    def metadata(self):
        return {'recipe':self.recipe,'selected_epoch':self.selected_epoch_,'stopped_epoch':self.stopped_epoch_,
                'fit_ids_digest':digest(self.fit_ids_),'fit_rows':len(self.fit_ids_),
                'preprocessing':self.preprocessor_.metadata(),'target_mean':self.mean_,'target_std':self.std_,
                'history':self.history_,'parameter_count':sum(p.numel() for p in self.model_.parameters()),
                'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}

    def save(self,path):
        payload={'recipe':self.recipe,'settings':self.settings,'metadata':self.metadata(),
                 'state':{k:v.detach().numpy().tolist() for k,v in self.model_.state_dict().items()}}
        with Path(path).open('x') as stream: json.dump(payload,stream,allow_nan=False)

    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text());obj=cls(payload['recipe'],payload['settings'])
        meta=payload['metadata'];obj.saved_metadata_=meta
        obj.preprocessor_=FactorBasis.restore(meta['preprocessing'])
        obj.mean_,obj.std_=meta['target_mean'],meta['target_std']
        obj.model_=FactorNetwork(obj.recipe,obj.settings,obj.preprocessor_.sizes_,obj.preprocessor_.width_)
        obj.model_.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in payload['state'].items()})
        return obj


def fit_partition(fitting,calibration,outer_training,query,target,recipe,settings,calibration_base,grid):
    for part in (fitting,calibration,outer_training,query):
        if part.sample_id.duplicated().any(): raise ValueError('Duplicate sample IDs')
    if set(fitting.sample_id)&set(calibration.sample_id): raise ValueError('Fitting/calibration overlap')
    if set(outer_training.sample_id)!=set(fitting.sample_id)|set(calibration.sample_id):
        raise ValueError('Incorrect training union')
    if set(outer_training.sample_id)&set(query.sample_id): raise ValueError('Outer query overlaps training')
    if any(t in query for t in TARGETS): raise ValueError('Query labels must be removed')
    selector=FactorRegressor(recipe,settings).initialize(fitting,fitting[target].to_numpy())
    clean_cal=unlabeled(calibration)
    epoch=selector.train(settings['max_epochs'],(clean_cal,calibration[target].to_numpy()))
    cp=selector.predict(clean_cal)
    weight,losses=select_weight(calibration[target],calibration_base,cp,grid)
    final=FactorRegressor(recipe,settings).initialize(outer_training,outer_training[target].to_numpy())
    final.train(epoch);prediction=final.predict(query);final.calibration_model_=selector
    return final,prediction,{'weight':weight,'calibration_mae_by_weight':losses,
                            'calibration':selector.metadata(),'refit':final.metadata()},cp
