"""Numeric latent-gradient regularization; equations from TANGOS (ICLR 2023).

Fresh matched periodic-TabM arms, with no reward for output disagreement.
"""
from copy import deepcopy
from pathlib import Path
import resource
import numpy as np
import torch
from torch import nn

from .data import FEATURES, TARGETS
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest, make_network
from .v30_deep_kernel import select_weight


def attribution_terms(attributions, epsilon):
    """Rows x units x numeric features; average unordered absolute cosines."""
    if attributions.ndim != 3 or attributions.shape[1] < 2 or epsilon <= 0:
        raise ValueError('Invalid attribution tensor or epsilon')
    specialization=attributions.abs().sum(-1).mean()
    # Clamp squared norm before sqrt: finite second derivatives at a zero vector.
    norms=attributions.square().sum(-1).clamp_min(epsilon**2).sqrt()
    unit=attributions/norms[...,None]
    pairs=torch.triu_indices(unit.shape[1],unit.shape[1],offset=1,device=unit.device)
    cosine=(unit[:,pairs[0]]*unit[:,pairs[1]]).sum(-1).abs().mean()
    return specialization,cosine


def sampled_units(settings,step):
    member=(step//settings['auxiliary_every_steps'])%settings['tabm_k']
    rng=np.random.default_rng(np.random.SeedSequence([settings['random_seed'],step,49001]))
    units=rng.choice(settings['width'],size=settings['auxiliary_units'],replace=False)
    return member,units.tolist()


def latent_attributions(model,x_num,x_cat,member,units,create_graph=True):
    """Eval-mode attribution, restoring all modes without consuming Torch RNG."""
    modes=[(module,module.training) for module in model.modules()]
    captured=[]
    hook=model.backbone.register_forward_hook(lambda _m,_args,output:captured.append(output))
    x=x_num.detach().clone().requires_grad_(True)
    try:
        model.eval()
        with torch.enable_grad():
            model(x,x_cat)
            hidden=captured[0]
            grads=[torch.autograd.grad(hidden[:,member,u].sum(),x,create_graph=create_graph,
                                       retain_graph=True)[0] for u in units]
        return torch.stack(grads,dim=1)
    finally:
        hook.remove()
        for module,mode in modes:module.training=mode


def clean(frame):
    return frame.drop(columns=[t for t in TARGETS if t in frame])


def architecture(settings,categories):
    return make_network({'backbone':'tabm','frequency':settings['frequency_init_scale']},
                        settings,categories).double()


class GradientRegressor:
    def __init__(self,recipe,settings):
        if recipe not in ('BASE','TANGOS'):raise ValueError('Unknown gradient arm')
        self.recipe,self.settings=recipe,deepcopy(settings)

    def _inputs(self,frame):
        if any(t in frame for t in TARGETS):raise ValueError('Query labels must be removed')
        x,cat=self.preprocessor_.transform_tabm(frame)
        if not np.isfinite(x).all():raise ValueError('Nonfinite query input')
        return torch.tensor(x,dtype=torch.float64),torch.tensor(cat,dtype=torch.long)

    def initialize(self,frame,y):
        y=np.asarray(y,float)
        if y.shape!=(len(frame),) or not np.isfinite(y).all() or y.std()<=0:
            raise ValueError('Invalid training target')
        torch.set_num_threads(1);torch.manual_seed(self.settings['random_seed'])
        self.preprocessor_=NumericPreprocessor(structure='raw_tabm').fit(clean(frame))
        self.mean_,self.std_=float(y.mean()),float(y.std())
        self.fit_ids_=frame.sample_id.astype(str).tolist()
        self.model_=architecture(self.settings,self.preprocessor_.n_spout_categories_)
        self.optimizer_=torch.optim.AdamW(self.model_.parameters(),lr=self.settings['learning_rate'],
                                         weight_decay=self.settings['weight_decay'])
        self.x_train_,self.cat_train_=self._inputs(clean(frame))
        self.y_train_=torch.tensor((y-self.mean_)/self.std_,dtype=torch.float64)
        return self

    def train(self,epochs,validation=None):
        if epochs<1:raise ValueError('Positive epoch count required')
        rng=np.random.default_rng(self.settings['random_seed']);step=0
        best,stale,best_epoch,best_state=float('inf'),0,0,None
        self.history_=[];self.auxiliary_updates_=0
        for epoch in range(1,epochs+1):
            self.model_.train();order=rng.permutation(len(self.y_train_));total=0.
            specs,orths=[],[]
            for start in range(0,len(order),self.settings['batch_size']):
                idx=order[start:start+self.settings['batch_size']]
                self.optimizer_.zero_grad(set_to_none=True)
                pred=self.model_(self.x_train_[idx],self.cat_train_[idx])[:,:,0]
                mse=((pred-self.y_train_[idx,None])**2).mean();loss=mse
                active=(self.recipe=='TANGOS' and step%self.settings['auxiliary_every_steps']==0
                        and (self.settings['lambda_specialization']>0 or self.settings['lambda_orthogonalization']>0))
                if active:
                    sub=idx[:self.settings['auxiliary_rows']]
                    member,units=sampled_units(self.settings,step)
                    attr=latent_attributions(self.model_,self.x_train_[sub],self.cat_train_[sub],member,units)
                    spec,orth=attribution_terms(attr,self.settings['norm_epsilon'])
                    loss=loss+self.settings['auxiliary_scale']*(self.settings['lambda_specialization']*spec+
                                                               self.settings['lambda_orthogonalization']*orth)
                    specs.append(float(spec.detach()));orths.append(float(orth.detach()))
                    self.auxiliary_updates_+=1
                if not torch.isfinite(loss):raise ValueError('Nonfinite gradient-regularized loss')
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(),self.settings['gradient_norm_clip'],error_if_nonfinite=True)
                self.optimizer_.step();step+=1;total+=float(mse.detach())*len(idx)
            row={'epoch':epoch,'standardized_training_mse':total/len(order),'auxiliary_updates':len(specs),
                 'specialization':float(np.mean(specs)) if specs else 0.,
                 'orthogonalization':float(np.mean(orths)) if orths else 0.}
            if validation is not None:
                value=float(np.abs(self.predict(validation[0])-validation[1]).mean()/self.std_)
                row['calibration_standardized_mae']=value
                if value<best-self.settings['min_delta_standardized_mae']:
                    best,best_epoch,stale=value,epoch,0;best_state=deepcopy(self.model_.state_dict())
                else:stale+=1
            self.history_.append(row)
            if validation is not None and stale>=self.settings['patience']:break
        if validation is not None:
            if best_state is None:raise ValueError('No finite selected epoch')
            self.model_.load_state_dict(best_state)
        else:best_epoch=epoch
        self.selected_epoch_,self.stopped_epoch_=best_epoch,epoch
        return best_epoch

    def predict(self,frame):
        self.model_.eval()
        with torch.no_grad():
            pred=self.model_(*self._inputs(frame))[:,:,0].numpy().mean(1)*self.std_+self.mean_
        if pred.shape!=(len(frame),) or not np.isfinite(pred).all():raise ValueError('Invalid prediction')
        return pred

    def metadata(self):
        return {'recipe':self.recipe,'selected_epoch':self.selected_epoch_,'stopped_epoch':self.stopped_epoch_,
                'fit_ids_digest':digest(self.fit_ids_),'fit_rows':len(self.fit_ids_),
                'preprocessing':self.preprocessor_.metadata(),'target_mean':self.mean_,'target_std':self.std_,
                'history':self.history_,'auxiliary_updates':self.auxiliary_updates_,
                'parameter_count':sum(p.numel() for p in self.model_.parameters()),
                'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}

    def save(self,path):
        payload={'recipe':self.recipe,'settings':self.settings,'metadata':self.metadata(),'state':self.model_.state_dict()}
        with Path(path).open('xb') as stream:torch.save(payload,stream)

    @classmethod
    def load(cls,path):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        obj=cls(payload['recipe'],payload['settings']);obj.saved_metadata_=payload['metadata']
        meta=obj.saved_metadata_;info=meta['preprocessing']
        obj.preprocessor_=NumericPreprocessor(structure='raw_tabm')
        obj.preprocessor_.means_=np.asarray(info['means']);obj.preprocessor_.stds_=np.asarray(info['stds'])
        obj.preprocessor_.spout_to_index_={int(k):v for k,v in info['spout_vocabulary'].items()}
        obj.preprocessor_.n_spout_categories_=info['n_spout_categories']
        obj.mean_,obj.std_=meta['target_mean'],meta['target_std']
        with torch.random.fork_rng(devices=[]):
            obj.model_=architecture(obj.settings,info['n_spout_categories'])
        obj.model_.load_state_dict(payload['state']);return obj


def fit_partition(fitting,calibration,outer_training,query,target,recipe,settings,calibration_base,grid):
    for part in (fitting,calibration,outer_training,query):
        if part.sample_id.duplicated().any():raise ValueError('Duplicate sample IDs')
    if set(fitting.sample_id)&set(calibration.sample_id):raise ValueError('Fitting/calibration overlap')
    if set(outer_training.sample_id)!=set(fitting.sample_id)|set(calibration.sample_id):raise ValueError('Incorrect training union')
    if set(outer_training.sample_id)&set(query.sample_id):raise ValueError('Outer query overlaps training')
    if any(t in query for t in TARGETS):raise ValueError('Query labels must be removed')
    selector=GradientRegressor(recipe,settings).initialize(fitting,fitting[target].to_numpy())
    epoch=selector.train(settings['max_epochs'],(clean(calibration),calibration[target].to_numpy()))
    cp=selector.predict(clean(calibration));weight,losses=select_weight(calibration[target],calibration_base,cp,grid)
    final=GradientRegressor(recipe,settings).initialize(outer_training,outer_training[target].to_numpy())
    final.train(epoch);pred=final.predict(query);final.calibration_model_=selector
    return final,pred,{'weight':weight,'calibration_mae_by_weight':losses,
                      'calibration':selector.metadata(),'refit':final.metadata()},cp
