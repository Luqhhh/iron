"""Native joint TabM: frozen LR cycles and sparse cycle-end parameter averaging."""
from collections import deque
import resource
import numpy as np
import torch
from .component_regularization import ComponentRegressor,clone_state,inference_state
from .v12_joint import joint_loss
from .cycle_swa_protocol import epoch_lr,cycle_epochs,MECHANISMS,digest

class CycleSWARegressor(ComponentRegressor):
    def __init__(self,settings,directory,ledger,phase,unit):
        super().__init__({'backbone':'tabm','frequency':.01},settings,'BASE',MECHANISMS,directory)
        self.ledger,self.phase,self.unit=ledger,phase,unit;self.initializations=0
    def _initialize(self,frame,y):
        stage=('selection','refit')[self.initializations];self.initializations+=1;self.active=self.unit+':SWA_CYCLE_TAIL:'+stage
        self.ledger.reserve(self.phase,'state',self.active);self.ledger.reserve(self.phase,'optimizer',self.active)
        super()._initialize(frame,y)
    def fit(self,frame,y):
        result=super().fit(frame,y)
        self.metadata_['budget_limited']=False
        self.metadata_['selection_policy']='complete_frozen_240_epoch_cycle_grid'
        return result
    def _train(self,frame,y,epochs,validation=None):
        x,cat=self._inputs(frame);target=torch.as_tensor((y-self.mean_)/self.std_,dtype=torch.float32)
        rng=np.random.default_rng(self.settings['random_seed']);window=deque(maxlen=5);history=[]
        best=float('inf');best_epoch=0;best_state=None;best_window=None
        if validation is not None:
            if epochs!=240:raise ValueError('Complete frozen240 selection required')
            if set(frame.sample_id)&set(validation[0].sample_id):raise ValueError('Inner calibration overlap')
            vx,vc=self._inputs(validation[0]);vy=torch.as_tensor((validation[1]-self.mean_)/self.std_,dtype=torch.float32)
        else:cycle_epochs(epochs)
        def average():
            first=window[0][1]
            return {k:torch.stack([s[k] for _,s in window]).mean(0) if v.is_floating_point() else v.clone() for k,v in first.items()}
        for epoch in range(1,epochs+1):
            lr=epoch_lr(epoch)
            for group in self.optimizer_.param_groups:group['lr']=lr
            self.model_.train();order=rng.permutation(len(frame));losses=[]
            for start in range(0,len(order),self.settings['batch_size']):
                idx=order[start:start+self.settings['batch_size']];self.optimizer_.zero_grad(set_to_none=True)
                loss=joint_loss(self.model_(x[idx],cat[idx]),target[idx])
                if not torch.isfinite(loss):raise ValueError('Nonfinite joint training loss')
                loss.backward();self.optimizer_.step();losses.append(float(loss.detach()))
            self.model_.eval();row={'epoch':epoch,'learning_rate':lr,'updates':len(losses),'gradient_evaluations':len(losses),'mean_batch_training_loss':float(np.mean(losses))}
            if epoch in range(100,241,20):
                window.append((epoch,clone_state(self.model_)));state=average()
                if validation is not None:
                    with inference_state(self.model_,state),torch.no_grad():value=float((self.model_(vx,vc).mean(1)-vy).abs().mean())
                    row['validation_mae']=value
                    if value<best-self.settings['min_delta']:
                        best,best_epoch=value,epoch;best_state={k:v.clone() for k,v in state.items()}
                        best_window=[(e,{k:v.clone() for k,v in s.items()}) for e,s in window]
            history.append(row)
        phase='selection' if validation is not None else 'refit';selected=best_epoch if validation is not None else epochs
        if validation is not None:
            if best_state is None:raise ValueError('No finite completed-cycle checkpoint')
            self.selection_stopped_epoch_=epochs;self.model_.load_state_dict(best_state);witness=best_window
        else:self.model_.load_state_dict(average());witness=list(window)
        self.model_.eval();self.traces[phase]={'history':history,'selected_epoch':selected,'stopped_epoch':epochs,'fit_ids_digest':digest(frame.sample_id.tolist()),'fit_rows':len(frame),
                                            'updates':sum(r['updates'] for r in history),'gradient_evaluations':sum(r['gradient_evaluations'] for r in history),'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        self.save(self.directory/f'{phase}.pt',self.traces[phase])
        with (self.directory/f'{phase}-window.pt').open('xb') as stream:
            torch.save({'epochs':[e for e,s in witness],'states':[s for e,s in witness],'window':5,'warmup_epochs':80,'cycle_length':20,'selected_epoch':selected},stream)
        self.ledger.complete(self.phase,'optimizer',self.active);self.ledger.complete(self.phase,'state',self.active)
        return selected
