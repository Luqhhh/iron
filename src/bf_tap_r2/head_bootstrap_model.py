"""Original random joint TabM; only fit batch per-row/head MSE weights change."""
import resource
import numpy as np
import torch
from .component_regularization import ComponentRegressor,clone_state
from .head_bootstrap_protocol import BootstrapStream,digest,CANDIDATE,expected_mechanisms
from .data import TARGETS

def bootstrap_loss(prediction,target,weights):
    if prediction.ndim!=3 or prediction.shape[2]!=2 or target.shape!=(len(prediction),2) or weights.shape!=prediction.shape[:2] or not len(prediction):
        raise ValueError("Expected nonempty row/head/two-target and matching weight matrix")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all() or not torch.isfinite(weights).all() or (weights<0).any() or not torch.equal(weights,weights.round()):
        raise ValueError("Finite nonnegative integer bootstrap weights required")
    return ((prediction-target[:,None,:]).square()*weights[:,:,None]).mean()

class BootstrapRegressor(ComponentRegressor):
    def __init__(self,settings,directory,ledger,phase,unit):
        super().__init__({"backbone":"tabm","frequency":.01},settings,"BASE",expected_mechanisms(CANDIDATE,phase),directory)
        self.ledger,self.phase,self.unit=ledger,phase,unit;self.initializations=0
    def _initialize(self,frame,y):
        stage=("selection","refit")[self.initializations];self.initializations+=1
        self.active=self.unit+":"+CANDIDATE+":"+stage
        self.ledger.reserve(self.phase,"state",self.active);self.ledger.reserve(self.phase,"optimizer",self.active)
        super()._initialize(frame,y)
    def _train(self,frame,y,epochs,validation=None):
        selected=self._train_bootstrap(frame,y,epochs,validation)
        self.ledger.complete(self.phase,"optimizer",self.active);self.ledger.complete(self.phase,"state",self.active)
        return selected
    def _train_bootstrap(self,frame,y,epochs,validation=None):
        x,cat=self._inputs(frame);target=torch.as_tensor((y-self.mean_)/self.std_,dtype=torch.float32)
        rng=np.random.default_rng(self.settings["random_seed"]);stream=BootstrapStream(self.settings["random_seed"])
        best,best_epoch,stale=float("inf"),0,0;best_state=None;history=[]
        if validation is not None:
            if set(frame.sample_id)&set(validation[0].sample_id):raise ValueError("Inner train/validation overlap")
            vx,vc=self._inputs(validation[0]);vy=torch.as_tensor((validation[1]-self.mean_)/self.std_,dtype=torch.float32)
        for epoch in range(1,epochs+1):
            self.model_.train();order=rng.permutation(len(frame));losses=[]
            for start in range(0,len(order),self.settings["batch_size"]):
                idx=order[start:start+self.settings["batch_size"]]
                weights=torch.as_tensor(stream.draw(frame.iloc[idx].sample_id.tolist(),self.settings["tabm_k"]),dtype=torch.float32)
                self.optimizer_.zero_grad(set_to_none=True);pred=self.model_(x[idx],cat[idx])
                loss=bootstrap_loss(pred,target[idx],weights)
                if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
                loss.backward();self.optimizer_.step();losses.append(float(loss.detach()))
            self.model_.eval()
            with torch.no_grad():
                training_pred=self.model_(x,cat).mean(1);train_mae=float((training_pred-target).abs().mean());train_mse=float((training_pred-target).square().mean())
                value=float((self.model_(vx,vc).mean(1)-vy).abs().mean()) if validation is not None else None
            row={"epoch":epoch,"updates":len(losses),"gradient_evaluations":len(losses),"training_eval_mae":train_mae,"training_eval_mse":train_mse,"mean_batch_training_loss":float(np.mean(losses)),"learning_rate":self.settings["learning_rate"],"training_objective":"fixed_denominator_per_row_head_poisson_mse","bootstrap":stream.certificate()}
            if validation is not None:
                row["validation_mae"]=value
                if value<best-self.settings["min_delta"]:best,best_epoch,stale=value,epoch,0;best_state=clone_state(self.model_)
                else:stale+=1
            history.append(row)
            if validation is not None and stale>=self.settings["patience"]:break
        phase="selection" if validation is not None else "refit";selected=best_epoch if validation is not None else epochs
        if validation is not None:
            if best_state is None:raise ValueError("No selected state")
            self.selection_stopped_epoch_=epoch;self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.traces[phase]={"history":history,"selected_epoch":selected,"stopped_epoch":epoch,"fit_ids_digest":digest(frame.sample_id.tolist()),"fit_rows":len(frame),"updates":sum(row["updates"] for row in history),"gradient_evaluations":sum(row["gradient_evaluations"] for row in history),"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"bootstrap_fit_ids_digest":digest(frame.sample_id.tolist()),"bootstrap_seed":int(self.settings["random_seed"])+1000003}
        if self.directory is not None:self.save(self.directory/f"{phase}.pt",self.traces[phase])
        return selected
    def predict(self,frame):
        if any(t in frame for t in TARGETS):raise ValueError("Prediction query contains targets")
        return super().predict(frame)
