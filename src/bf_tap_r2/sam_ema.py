"""SAM training followed by EMA of restored, updated parameters.

A separate recipe: the original BASE/EMA/SAM trainers remain byte-identical.
"""
import resource
import numpy as np
import torch
from .component_regularization import ComponentRegressor, clone_state, inference_state, sam_step, update_ema
from .v12_joint import joint_loss
from .v7_periodic import digest


def sam_ema_step(model, optimizer, closure, ema, mechanisms):
    result = sam_step(model, optimizer, closure, mechanisms['sam_rho'], mechanisms['sam_epsilon'])
    update_ema(model, ema, mechanisms['ema_beta'])
    return result


class SAMEMARegressor(ComponentRegressor):
    def __init__(self, recipe, settings, arm, mechanisms, directory=None):
        if arm != 'SAM_EMA': raise ValueError('Only the SAM_EMA recipe is registered')
        super().__init__(recipe, settings, 'SAM', mechanisms, directory)
        self.arm = arm

    def _train(self, frame, y, epochs, validation=None):
        x, cat = self._inputs(frame)
        target = torch.as_tensor((y-self.mean_)/self.std_, dtype=torch.float32)
        rng = np.random.default_rng(self.settings["random_seed"])
        ema = clone_state(self.model_)
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
                a,b,c = sam_ema_step(self.model_, self.optimizer_,
                    lambda: joint_loss(self.model_(x[idx],cat[idx]),target[idx]),
                    ema, self.mechanisms)
                losses.append(a); perturb_losses.append(b); norms.append(c)
            self.model_.eval()
            # Evaluation consumes no RNG. Restore raw parameters after EMA eval.
            state = ema if ema is not None else clone_state(self.model_)
            with inference_state(self.model_, state), torch.no_grad():
                training_pred = self.model_(x,cat).mean(1)
                train_mae = float((training_pred-target).abs().mean())
                train_mse = float((training_pred-target).square().mean())
                value = float((self.model_(vx,vc).mean(1)-vy).abs().mean()) if validation is not None else None
            row = {"epoch":epoch,"updates":len(losses),
                   "gradient_evaluations":2*len(losses), "ema_updates":len(losses),
                   "training_eval_mae":train_mae,"training_eval_mse":train_mse,
                   "mean_batch_training_loss":float(np.mean(losses))}
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
            "ema_updates":sum(r["ema_updates"] for r in history),
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        if self.directory is not None:
            self.save(self.directory/f"{phase}.pt", self.traces[phase])
        return selected

