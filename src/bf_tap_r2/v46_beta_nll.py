"""Frozen V46 paired single-Gaussian objectives; torch is optional at import."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math
import numpy as np
from .data import FEATURES, TARGETS
from .v3_6_networks import NumericPreprocessor
from .v5_resolution import paired_summary
from .v7_periodic import digest

ARMS = {'NLL': 0., 'BETA05': .5}


def verify_source_bytes(working, git_blob, frozen_sha):
    """Accept only exact recorded bytes or proven Git-identical CRLF conversion."""
    norm = lambda value: value.replace(b'\r\n', b'\n')
    if norm(working) != norm(git_blob):
        raise ValueError('Working source differs from committed Git source')
    h = lambda value: hashlib.sha256(value).hexdigest()
    if h(working) == frozen_sha:
        return 'exact'
    # Never use this function for data, models, manifests or prediction evidence.
    if frozen_sha in {h(git_blob), h(norm(git_blob)), h(norm(git_blob).replace(b'\n', b'\r\n'))}:
        return 'CRLF_only'
    raise ValueError('Frozen source hash is not an exact Git source byte variant')


def default_settings():
    return dict(dtype='float64', device='cpu', random_seed=42, inner_seed=42,
        hidden_widths=[128,128], periodic_embedding_dim=8, periodic_n_frequencies=4,
        periodic_frequency_init_scale=.01, scale_floor=.05, initial_scale=.5,
        learning_rate=.001, weight_decay=.0001, batch_size=256, max_epochs=240,
        patience=30, min_delta_standardized_mae=1e-5, gradient_norm_clip=10.,
        betas=[.9,.999], eps=1e-8)


def beta_nll(y, mean, variance, beta):
    import torch
    if beta not in (0., .5) or y.shape != mean.shape or variance.shape != y.shape:
        raise ValueError('Unfrozen beta or incompatible likelihood shapes')
    if not all(torch.isfinite(value).all() for value in (y,mean,variance)) or not (variance>0).all():
        raise ValueError('Nonfinite or nonpositive Gaussian variance')
    row = .5*((y-mean).square()/variance + variance.log() + math.log(2*math.pi))
    return (row * variance.detach().pow(beta)).mean()


class GaussianPreprocessor(NumericPreprocessor):
    def transform_mlp(self, frame):
        numeric, categorical = super().transform_mlp(frame)
        unknown = ~frame.spout_no.isin(self.spout_to_index_).to_numpy()
        categorical[unknown] = 0.
        return numeric, categorical


def fit_preprocessor(frame):
    if not np.isfinite(frame[list(FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite numeric input')
    return GaussianPreprocessor(structure='raw_mlp').fit(frame)


def make_network(settings, categories):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from rtdl_num_embeddings import PeriodicEmbeddings
    class SingleGaussian(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = PeriodicEmbeddings(len(FEATURES),
                d_embedding=settings['periodic_embedding_dim'],
                n_frequencies=settings['periodic_n_frequencies'],
                frequency_init_scale=settings['periodic_frequency_init_scale'], lite=True)
            width=len(FEATURES)*settings['periodic_embedding_dim']+categories
            layers=[]
            for hidden in settings['hidden_widths']:
                layers.extend([nn.Linear(width,hidden), nn.SiLU()]); width=hidden
            self.trunk=nn.Sequential(*layers)
            self.head=nn.Linear(width,3)
            nn.init.normal_(self.head.weight,std=.01)
            with torch.no_grad():
                self.head.bias.zero_()
                self.head.bias[2]=math.log(math.expm1(settings['initial_scale']-settings['scale_floor']))
        def forward(self,x):
            z=torch.cat([self.embedding(x[:,:len(FEATURES)]).flatten(1),x[:,len(FEATURES):]],1)
            logits,means,raw=self.head(self.trunk(z)).unbind(1)
            return logits, means, F.softplus(raw)+settings['scale_floor']
    return SingleGaussian()


class GaussianRegressor:
    def __init__(self, arm, settings):
        if arm not in ARMS: raise ValueError('Unknown frozen V46 arm')
        self.arm,self.settings=arm,dict(settings)

    def inputs(self,frame):
        import torch
        numeric,cat=self.preprocessor.transform_mlp(frame)
        values=np.concatenate([numeric,cat],1).astype(np.float64)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite input')
        return torch.as_tensor(values,dtype=torch.float64)

    def initialize(self,frame,y,on_start=None,stage='outer'):
        import torch
        y=np.asarray(y,dtype=float)
        if y.shape!=(len(frame),) or not np.isfinite(y).all() or not y.std()>0:
            raise ValueError('Invalid training target')
        if on_start is None: raise ValueError('Optimizer start accounting required')
        # Count BEFORE creating an optimizer; failures consume the allocation.
        on_start({'event':'optimizer_started','stage':stage,'arm':self.arm})
        torch.set_num_threads(1); torch.manual_seed(self.settings['random_seed'])
        self.preprocessor=fit_preprocessor(frame)
        self.mean,self.std=float(y.mean()),float(y.std())
        self.fit_ids=frame.sample_id.astype(str).tolist()
        self.model=make_network(self.settings,self.preprocessor.n_spout_categories_).double()
        self.x,self.y=self.inputs(frame),torch.as_tensor((y-self.mean)/self.std,dtype=torch.float64)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.settings['learning_rate'],
            weight_decay=self.settings['weight_decay'],betas=tuple(self.settings['betas']),eps=self.settings['eps'])
        return self

    def step(self,indices):
        import torch
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        _,means,scales=self.model(self.x[indices])
        loss=beta_nll(self.y[indices],means,scales.square(),ARMS[self.arm])
        if not torch.isfinite(loss): raise ValueError('Nonfinite V46 loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.settings['gradient_norm_clip'],error_if_nonfinite=True)
        self.optimizer.step()
        return float(loss.detach())

    def train(self,epochs,validation=None):
        rng=np.random.default_rng(self.settings['random_seed'])
        best,stale,best_epoch,best_state=float('inf'),0,0,None
        self.history=[]
        for epoch in range(1,epochs+1):
            order=rng.permutation(len(self.y)); total=0.
            for start in range(0,len(order),self.settings['batch_size']):
                indices=order[start:start+self.settings['batch_size']]
                total+=self.step(indices)*len(indices)
            row={'epoch':epoch,'objective':total/len(order)}
            if validation is not None:
                value=float(np.abs(self.predict(validation[0])-validation[1]).mean()/self.std)
                row['calibration_standardized_mae']=value
                if value<best-self.settings['min_delta_standardized_mae']:
                    best,best_epoch,stale=value,epoch,0; best_state=deepcopy(self.model.state_dict())
                else: stale+=1
            self.history.append(row)
            if validation is not None and stale>=self.settings['patience']: break
        if validation is not None:
            if best_state is None: raise ValueError('No selected epoch')
            self.model.load_state_dict(best_state)
        else: best_epoch=epochs
        self.selected_epoch,self.stopped_epoch=best_epoch,epoch
        return best_epoch

    def predict(self,frame,chunk_rows=2048):
        import torch
        if any(t in frame for t in TARGETS): raise ValueError('Query labels must be removed')
        if chunk_rows<1: raise ValueError('Invalid chunk size')
        self.model.eval(); values=[]
        with torch.no_grad():
            for start in range(0,len(frame),chunk_rows):
                _,means,_=self.model(self.inputs(frame.iloc[start:start+chunk_rows]))
                values.append(means.numpy()*self.std+self.mean)
        result=np.concatenate(values) if values else np.empty(0)
        if result.shape!=(len(frame),) or not np.isfinite(result).all(): raise ValueError('Invalid Gaussian prediction')
        return result

    def metadata(self):
        return dict(arm=self.arm,settings=self.settings,fit_rows=len(self.fit_ids),
            fit_ids_digest=digest(self.fit_ids),target_mean=self.mean,target_std=self.std,
            preprocessing=self.preprocessor.metadata(),selected_epoch=self.selected_epoch,
            stopped_epoch=self.stopped_epoch,history=self.history,
            parameter_count=sum(p.numel() for p in self.model.parameters()))

    def save(self,path):
        import torch
        payload=dict(arm=self.arm,settings=self.settings,preprocessing=self.preprocessor.metadata(),
            mean=self.mean,std=self.std,state=self.model.state_dict())
        with path.open('xb') as stream: torch.save(payload,stream)

    @classmethod
    def load(cls,path):
        import torch
        payload=torch.load(path,map_location='cpu',weights_only=True)
        obj=cls(payload['arm'],payload['settings']); info=payload['preprocessing']
        obj.preprocessor=GaussianPreprocessor(structure='raw_mlp')
        obj.preprocessor.means_=np.array(info['means']); obj.preprocessor.stds_=np.array(info['stds'])
        obj.preprocessor.spout_to_index_={int(k):v for k,v in info['spout_vocabulary'].items()}
        obj.preprocessor.n_spout_categories_=info['n_spout_categories']
        obj.mean,obj.std=payload['mean'],payload['std']
        obj.model=make_network(obj.settings,obj.preprocessor.n_spout_categories_).double()
        obj.model.load_state_dict(payload['state'])
        return obj


def fit_partition(training,query,arm,settings,on_start):
    from .v3_4_bags import group_safe_inner_folds
    if any(t in query for t in TARGETS): raise ValueError('Query labels must be removed')
    if set(training.sample_id)&set(query.sample_id): raise ValueError('Training/query overlap')
    inner=group_safe_inner_folds(training,seed=settings['inner_seed'])['fold']; mask=inner==0
    fitting,held=training.loc[~mask].reset_index(drop=True),training.loc[mask].reset_index(drop=True)
    selected=GaussianRegressor(arm,settings).initialize(fitting,fitting.tap_time_len.to_numpy(),on_start,'inner')
    epoch=selected.train(settings['max_epochs'],(held.drop(columns=list(TARGETS)),held.tap_time_len.to_numpy()))
    final=GaussianRegressor(arm,settings).initialize(training,training.tap_time_len.to_numpy(),on_start,'outer')
    final.train(epoch)
    prediction=final.predict(query)
    metadata=dict(inner=selected.metadata(),outer=final.metadata(),
        calibration_ids_digest=digest(held.sample_id.astype(str).tolist()),
        query_ids_digest=digest(query.sample_id.astype(str).tolist()))
    return selected,final,prediction,metadata


def check_coverage(records,seeds):
    if set(records)!=set(ARMS): raise ValueError('Paired arm coverage required')
    if any(set(v)!=set(map(str,seeds)) for v in records.values()): raise ValueError('Seed coverage required')
    if any(row.get('folds')!=5 or not all(np.isfinite(row[k]) for k in ['gain','candidate_score'])
           for v in records.values() for row in v.values()): raise ValueError('Complete finite fold coverage required')


def eligible(records):
    check_coverage(records,[42,3407])
    candidate=[records['BETA05'][str(s)] for s in [42,3407]]
    contrasts=[r['gain']-records['NLL'][str(s)]['gain'] for r,s in zip(candidate,[42,3407])]
    return bool(all(r['gain']>0 for r in candidate) and np.mean([r['gain'] for r in candidate])>=.01
                and np.mean(contrasts)>0 and np.mean([r['candidate_score'] for r in candidate])>=96.25)


def final_decision(records):
    seeds=[42,3407,7777,12011]; check_coverage(records,seeds)
    gains=[records['BETA05'][str(s)]['gain'] for s in seeds]
    contrasts=[g-records['NLL'][str(s)]['gain'] for g,s in zip(gains,seeds)]
    paired=paired_summary(gains); mechanism=paired_summary(contrasts)
    failures=[]
    if not eligible({a:{str(s):records[a][str(s)] for s in [42,3407]} for a in ARMS}): failures.append('development_gate')
    if not all(g>0 for g in gains): failures.append('not_all_four_seeds_positive')
    if paired['lcb95']<=0: failures.append('nonpositive_seed_LCB')
    if mechanism['mean']<=0: failures.append('nonpositive_mechanism_mean')
    return dict(promoted=not failures,failed_conditions=failures,paired=paired,mechanism=mechanism,release_authorized=False)


def resource_decision(rows,available_mib):
    train=max(r['train_p95'] for r in rows.values()); validation=max(r['validation_p95'] for r in rows.values())
    peak=max(r['peak_rss_mib'] for r in rows.values())
    projected=20/4*1.5*240*(16*train+2*validation)+300
    return dict(passed=bool(projected<=7200 and peak<=1024 and 4*peak+1024<=available_mib),
        projected_development_seconds=projected,peak_worker_rss_mib=peak,available_ram_mib=available_mib)