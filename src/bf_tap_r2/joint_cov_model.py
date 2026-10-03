"""Native two-output TabM; only per-head training loss metric changes."""
from pathlib import Path
import resource
import numpy as np
import torch
from .component_regularization import ComponentRegressor,clone_state,inference_state,update_ema,sam_step
from .joint_cov_protocol import covariance_metric,digest
from .data import TARGETS

def covariance_loss(prediction,target,precision):
    if prediction.ndim!=3 or prediction.shape[2]!=2 or target.shape!=(len(prediction),2) or precision.shape!=(2,2):
        raise ValueError("Expected row/head/two-output and 2x2 precision")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all() or not torch.isfinite(precision).all() or not torch.allclose(precision,precision.T) or torch.linalg.eigvalsh(precision).min()<=0:
        raise ValueError("Finite positive symmetric precision required")
    residual=prediction-target[:,None,:]
    return .5*torch.einsum('bhi,ij,bhj->bh',residual,precision,residual).mean()

class CovarianceRegressor(ComponentRegressor):
    def __init__(self,settings,directory,ledger,phase,unit):
        super().__init__({"backbone":"tabm","frequency":.01},settings,"BASE",{"joint_cov_shrinkage":.5,"loss":"half_quadratic_precision_per_head"},directory)
        self.ledger,self.phase,self.unit=ledger,phase,unit;self.initializations=0
    def _initialize(self,frame,y):
        self.covariance_,self.precision_numpy_=covariance_metric(y)
        self.precision_=torch.tensor(self.precision_numpy_,dtype=torch.float32)
        stage=("selection","refit")[self.initializations];self.initializations+=1;self.active=self.unit+":JOINT_COV_MSE:"+stage
        self.ledger.reserve(self.phase,"state",self.active);self.ledger.reserve(self.phase,"optimizer",self.active)
        super()._initialize(frame,y)
    def _train(self,*args,**kwargs):
        result=self._train_cov(*args,**kwargs)
        self.ledger.complete(self.phase,"optimizer",self.active);self.ledger.complete(self.phase,"state",self.active)
        return result

    def _train_cov(self, frame, y, epochs, validation=None):
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y-self.mean_)/self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings["random_seed"])
        ema = clone_state(self.model_) if self.arm == "EMA" else None
        best, best_epoch, stale = float("inf"), 0, 0
        best_state = None
        history = []
        if validation is not None:
            if set(frame.sample_id) & set(validation[0].sample_id):
                raise ValueError("Inner train/validation overlap")
            vx, vc = self._inputs(validation[0])
            vy = torch.as_tensor((validation[1]-self.mean_)/self.std_, dtype=torch.float32)
        for epoch in range(1, epochs+1):
            self.model_.train()
            order = rng.permutation(len(frame))
            losses, perturb_losses, norms = [], [], []
            for start in range(0, len(order), self.settings["batch_size"]):
                idx = order[start:start+self.settings["batch_size"]]
                if self.arm == "SAM":
                    a,b,c = sam_step(self.model_, self.optimizer_,
                        lambda: covariance_loss(self.model_(x[idx],cat[idx]),target[idx], self.precision_),
                        self.mechanisms["sam_rho"], self.mechanisms["sam_epsilon"])
                    losses.append(a); perturb_losses.append(b); norms.append(c)
                else:
                    self.optimizer_.zero_grad(set_to_none=True)
                    pred = self.model_(x[idx], cat[idx])
                    loss = covariance_loss(pred, target[idx], self.precision_)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    self.optimizer_.step()
                    losses.append(float(loss.detach()))
                    if ema is not None:
                        update_ema(self.model_, ema, self.mechanisms["ema_beta"])
            self.model_.eval()
            # Evaluation consumes no RNG. Restore raw parameters after EMA eval.
            state = ema if ema is not None else clone_state(self.model_)
            with inference_state(self.model_, state), torch.no_grad():
                training_pred = self.model_(x,cat).mean(1)
                train_mae = float((training_pred-target).abs().mean())
                train_mse = float((training_pred-target).square().mean())
                value = float((self.model_(vx,vc).mean(1)-vy).abs().mean()) if validation is not None else None
            row = {"epoch":epoch,"updates":len(losses),
                   "gradient_evaluations":len(losses)*(2 if self.arm=="SAM" else 1),
                   "training_eval_mae":train_mae,"training_eval_mse":train_mse,
                   "mean_batch_training_loss":float(np.mean(losses)), "learning_rate":self.settings["learning_rate"], "training_objective":"half_quadratic_precision_per_head"}
            if self.arm == "SAM":
                row.update(perturbed_loss=float(np.mean(perturb_losses)), gradient_norm=float(np.mean(norms)))
            if validation is not None:
                row["validation_mae"] = value
                if value < best-self.settings["min_delta"]:
                    best,best_epoch,stale = value,epoch,0
                    best_state = {k:v.clone() for k,v in state.items()}
                else:
                    stale += 1
            history.append(row)
            if validation is not None and stale >= self.settings["patience"]:
                break
        phase = "selection" if validation is not None else "refit"
        selected = best_epoch if validation is not None else epochs
        if validation is not None:
            if best_state is None:
                raise ValueError("No selected state")
            self.selection_stopped_epoch_ = epoch
            self.model_.load_state_dict(best_state)
        elif ema is not None:
            self.model_.load_state_dict(ema)
        self.model_.eval()
        self.traces[phase] = {"history":history,"selected_epoch":selected,
            "stopped_epoch":epoch,"fit_ids_digest":digest(frame.sample_id.tolist()),
            "fit_rows":len(frame),"updates":sum(r["updates"] for r in history),
            "gradient_evaluations":sum(r["gradient_evaluations"] for r in history),
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, "covariance":self.covariance_.tolist(), "precision":self.precision_numpy_.tolist(), "covariance_fit_ids_digest":digest(frame.sample_id.tolist())}
        if self.directory is not None:
            self.save(self.directory/f"{phase}.pt", self.traces[phase])
        return selected

    def predict(self, frame):
        if any(t in frame for t in TARGETS):
            raise ValueError("Prediction query contains targets")
        return super().predict(frame)
