"""Native joint TabM with a fit-only supervised representation distance objective."""
from pathlib import Path
import resource
import numpy as np
import pandas as pd
import torch
from .component_regularization import ComponentRegressor,clone_state,inference_state,update_ema,sam_step
from .v12_joint import joint_loss
from .data import FEATURES,TARGETS
from .v3_4_bags import group_safe_inner_folds
from .tabm_metric_protocol import Ledger,sha,digest,write_new

def target_metric_loss(h,y):
 if h.ndim!=2 or y.shape!=(len(h),) or not torch.isfinite(h).all() or not torch.isfinite(y).all():raise ValueError("Invalid target metric input")
 if len(h)<2:return h.sum()*0
 z=torch.nn.functional.normalize(h,dim=1,eps=1e-8)
 d=(1-z@z.T)/2;target=1-torch.exp(-torch.abs(y[:,None]-y[None,:]))
 mask=~torch.eye(len(h),dtype=torch.bool,device=h.device)
 return (d[mask]-target[mask]).square().mean()

class MetricRegressor(ComponentRegressor):
    def __init__(self,settings,directory,ledger,phase,unit,arm):
        self.aux_weight=0. if arm=="BASE" else .02
        super().__init__({"backbone":"tabm","frequency":.01},settings,"BASE",{"target_metric_weight":self.aux_weight},directory)
        self.ledger,self.phase,self.unit,self.identity_arm=ledger,phase,unit,arm;self.initializations=0
    def _initialize(self,frame,y):
        stage=("selection","refit")[self.initializations];self.initializations+=1;self.active=self.unit+":"+self.identity_arm+":"+stage
        self.ledger.reserve(self.phase,"state",self.active);self.ledger.reserve(self.phase,"optimizer",self.active)
        super()._initialize(frame,y)
    def _train(self,*args,**kwargs):
        result=self._train_metric(*args,**kwargs)
        self.ledger.complete(self.phase,"optimizer",self.active);self.ledger.complete(self.phase,"state",self.active)
        return result
    def _train_metric(self, frame, y, epochs, validation=None):
        handle=None
        if self.aux_weight:
            def capture(module,args):
                self._representation=args[0] if self.model_.training else None
            handle=self.model_.output.register_forward_pre_hook(capture)
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
                        lambda: joint_loss(self.model_(x[idx],cat[idx]),target[idx]),
                        self.mechanisms["sam_rho"], self.mechanisms["sam_epsilon"])
                    losses.append(a); perturb_losses.append(b); norms.append(c)
                else:
                    self.optimizer_.zero_grad(set_to_none=True)
                    pred = self.model_(x[idx], cat[idx])
                    loss = joint_loss(pred, target[idx])
                    if self.aux_weight:
                        representation=self._representation
                        if representation.shape!=(len(idx),self.settings["tabm_k"],self.settings["width"]):raise ValueError("Unexpected native representation shape")
                        loss=loss+self.aux_weight*target_metric_loss(representation.mean(1),target[idx,0])
                        self._representation=None
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
                   "mean_batch_training_loss":float(np.mean(losses))}
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
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        if self.directory is not None:
            self.save(self.directory/f"{phase}.pt", self.traces[phase])
        if handle is not None:handle.remove()
        self._representation=None
        return selected


def features(frame):return frame.drop(columns=[c for c in TARGETS if c in frame])
def train_unit(frame,train,query,directory,spec,settings,ledger,phase,unit):
 d=Path(directory);d.mkdir(exist_ok=False);training=frame.iloc[train].reset_index(drop=True)
 iv=np.asarray(group_safe_inner_folds(training,seed=settings["inner_seed"])["fold"]);inner=train[iv!=0];cal=train[iv==0]
 groups=pd.util.hash_pandas_object(frame[list(FEATURES)],index=False).to_numpy()
 if set(groups[train])&set(groups[query]) or set(groups[inner])&set(groups[cal]):raise ValueError("Duplicate group leak")
 pred={"ids":frame.iloc[query].sample_id.to_numpy(dtype=str),"cal_ids":frame.iloc[cal].sample_id.to_numpy(dtype=str)}
 for arm in ("BASE","TARGET_METRIC"):
  ad=d/arm;ad.mkdir()
  model=MetricRegressor(settings,ad,ledger,phase,unit,arm).fit(features(training),training[list(TARGETS)].to_numpy(float))
  pred[arm]=model.predict(features(frame.iloc[query]))[:,0]
  write_new(ad/"metadata.json",model.metadata_)
  del model
 with (d/"predictions.npz").open("xb") as f:np.savez(f,**pred)
 with (d/"partitions.npz").open("xb") as f:np.savez(f,train=train,query=query,inner=inner,cal=cal)
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 if rss>1024:raise ValueError("Worker memory exceeded")
 write_new(d/"complete.json",dict(phase=phase,unit=unit,peak_rss_mib=rss,hashes={str(p.relative_to(d)):sha(p) for p in d.rglob("*") if p.is_file()}))
