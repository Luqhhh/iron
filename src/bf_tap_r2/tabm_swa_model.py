"""Original native TabM optimization with finite uniform epoch-trajectory averaging."""
from pathlib import Path
import resource
import numpy as np
import torch
from .component_regularization import ComponentRegressor,clone_state,inference_state,update_ema,sam_step
from .v12_joint import joint_loss
from .tabm_swa_window import EpochWindow
from .tabm_swa_protocol import digest

class SWARegressor(ComponentRegressor):
    def __init__(self,settings,directory,ledger,phase,unit):
        super().__init__({"backbone":"tabm","frequency":.01},settings,"BASE",{"uniform_epoch_window":10},directory)
        self.ledger,self.phase,self.unit=ledger,phase,unit
        self.initializations=0
    def _initialize(self,frame,y):
        stage=("selection","refit")[self.initializations];self.initializations+=1;self.active=self.unit+":SWA_WINDOW10:"+stage
        self.ledger.reserve(self.phase,"state",self.active);self.ledger.reserve(self.phase,"optimizer",self.active)
        super()._initialize(frame,y)
    def _train(self,*args,**kwargs):
        result=self._train_swa(*args,**kwargs)
        self.ledger.complete(self.phase,"optimizer",self.active);self.ledger.complete(self.phase,"state",self.active)
        return result
    def _train_swa(self, frame, y, epochs, validation=None):
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y-self.mean_)/self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings["random_seed"])
        ema = None
        window=EpochWindow(10)
        best_window=None
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
                        lambda: joint_loss(self.model_(x[idx],cat[idx]),target[idx]),
                        self.mechanisms["sam_rho"], self.mechanisms["sam_epsilon"])
                    losses.append(a); perturb_losses.append(b); norms.append(c)
                else:
                    self.optimizer_.zero_grad(set_to_none=True)
                    pred = self.model_(x[idx], cat[idx])
                    loss = joint_loss(pred, target[idx])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    self.optimizer_.step()
                    losses.append(float(loss.detach()))
                    if ema is not None:
                        update_ema(self.model_, ema, self.mechanisms["ema_beta"])
            self.model_.eval()
            # Evaluation consumes no RNG. Restore raw parameters after averaged-state eval.
            window.append(epoch,clone_state(self.model_))
            state=window.average()
            with inference_state(self.model_, state), torch.no_grad():
                training_pred = self.model_(x,cat).mean(1)
                train_mae = float((training_pred-target).abs().mean())
                train_mse = float((training_pred-target).square().mean())
                value = float((self.model_(vx,vc).mean(1)-vy).abs().mean()) if validation is not None else None
            row = {"epoch":epoch,"updates":len(losses),
                   "gradient_evaluations":len(losses)*(2 if self.arm=="SAM" else 1),
                   "training_eval_mae":train_mae,"training_eval_mse":train_mse,
                   "mean_batch_training_loss":float(np.mean(losses))}
            if self.arm == "SAM":
                row.update(perturbed_loss=float(np.mean(perturb_losses)), gradient_norm=float(np.mean(norms)))
            if validation is not None:
                row["validation_mae"] = value
                if value < best-self.settings["min_delta"]:
                    best,best_epoch,stale = value,epoch,0
                    best_state = {k:v.clone() for k,v in state.items()}
                    best_window=window.snapshot()
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
        else:
            self.model_.load_state_dict(window.average())
        self.model_.eval()
        self.traces[phase] = {"history":history,"selected_epoch":selected,
            "stopped_epoch":epoch,"fit_ids_digest":digest(frame.sample_id.tolist()),
            "fit_rows":len(frame),"updates":sum(r["updates"] for r in history),
            "gradient_evaluations":sum(r["gradient_evaluations"] for r in history),
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        if self.directory is not None:
            self.save(self.directory/f"{phase}.pt", self.traces[phase])
            witness=best_window if validation is not None else window.snapshot()
            with (self.directory/f"{phase}-window.pt").open("xb") as f:
                torch.save(dict(epochs=[e for e,s in witness],states=[s for e,s in witness],window=10,selected_epoch=selected),f)
        return selected

