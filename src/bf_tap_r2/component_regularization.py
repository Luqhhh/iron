"""Native float32 component replay, EMA and same-dropout SAM.

The original component modules remain immutable. This subclass retains their
initialization, data transforms, optimizer and inner-selection/refit protocol.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import copy
import math
import resource

import numpy as np
import torch

from .data import TARGETS
from .v12_joint import JointRegressor, joint_loss, make_network
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def update_ema(model, state, beta):
    if not 0 <= beta < 1:
        raise ValueError("Invalid EMA decay")
    parameters = dict(model.named_parameters())
    for key, value in model.state_dict().items():
        if key in parameters:
            state[key].mul_(beta).add_(value, alpha=1-beta)
        else:
            state[key].copy_(value)


@contextmanager
def inference_state(model, state):
    old = clone_state(model)
    try:
        model.load_state_dict(state)
        yield
    finally:
        model.load_state_dict(old)


def sam_step(model, optimizer, closure, rho, epsilon=1e-12):
    """One AdamW update; restore exact originals even if the closure raises."""
    if rho < 0 or epsilon <= 0:
        raise ValueError("Invalid SAM constants")
    optimizer.zero_grad(set_to_none=True)
    rng_before = torch.get_rng_state()
    first = closure()
    if not torch.isfinite(first):
        raise ValueError("Nonfinite first SAM loss")
    first.backward()
    rng_after = torch.get_rng_state()
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters])) if parameters else torch.tensor(0.)
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite SAM gradient")
    originals = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:
                p.add_(p.grad * (rho/(norm+epsilon)))
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_before)
        second = closure()
        if not torch.isfinite(second):
            raise ValueError("Nonfinite second SAM loss")
        second.backward()
    finally:
        with torch.no_grad():
            for p, value in zip(parameters, originals):
                p.copy_(value)
        torch.set_rng_state(rng_after)
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Nonfinite SAM second gradient")
    optimizer.step()
    return float(first.detach()), float(second.detach()), float(norm)


class ComponentRegressor(JointRegressor):
    def __init__(self, recipe, settings, arm, mechanisms, directory=None):
        super().__init__(recipe, settings)
        if arm not in ("BASE", "EMA", "SAM"):
            raise ValueError("Unknown mechanism")
        self.arm, self.mechanisms = arm, dict(mechanisms)
        self.directory = Path(directory) if directory is not None else None
        self.traces = {}

    def fit(self, frame, y):
        if frame.sample_id.duplicated().any():
            raise ValueError("Duplicate fitting IDs")
        result = super().fit(frame, y)
        self.metadata_["arm"] = self.arm
        self.metadata_["traces"] = self.traces
        return result

    def _train(self, frame, y, epochs, validation=None):
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
        return selected

    def predict(self, frame):
        if any(t in frame for t in TARGETS):
            raise ValueError("Prediction query contains targets")
        return super().predict(frame)

    def save(self, path, trace):
        info = self.preprocessor_.metadata()
        payload = dict(recipe=self.recipe, settings=self.settings, arm=self.arm,
            mechanisms=self.mechanisms, preprocessing=info, mean=self.mean_.tolist(),
            std=self.std_.tolist(), state=clone_state(self.model_), trace=copy.deepcopy(trace))
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        result = cls(saved["recipe"],saved["settings"],saved["arm"],saved["mechanisms"])
        p = NumericPreprocessor(structure="raw_tabm")
        info = saved["preprocessing"]
        p.means_,p.stds_ = np.asarray(info["means"]),np.asarray(info["stds"])
        p.spout_to_index_ = {int(k):v for k,v in info["spout_vocabulary"].items()}
        p.n_spout_categories_ = info["n_spout_categories"]
        result.preprocessor_ = p
        result.mean_,result.std_ = np.asarray(saved["mean"]),np.asarray(saved["std"])
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            result.model_ = make_network(result.recipe,result.settings,p.n_spout_categories_,len(result.mean_))
        result.model_.load_state_dict(saved["state"])
        result.model_.eval()
        result.saved = saved
        return result


def replacement(reference, old, new, weight=.5):
    reference,old,new = (np.asarray(a,float) for a in (reference,old,new))
    if reference.shape!=old.shape or old.shape!=new.shape or reference.ndim!=1:
        raise ValueError("Replacement shape mismatch")
    if not all(np.isfinite(a).all() for a in (reference,old,new)):
        raise ValueError("Nonfinite replacement")
    return reference + weight*(new-old)


def paired_row_bootstrap(y, base, candidate, replicates=2000, seed=52002):
    """Resample row clusters, compute each seed's ratio, then average scores."""
    y=np.asarray(y,float);base=np.asarray(base,float);candidate=np.asarray(candidate,float)
    if base.shape!=candidate.shape or base.ndim!=2 or base.shape[1]!=len(y):
        raise ValueError("Expected seed by row predictions")
    contributions = np.abs(y[None,:]-base)-np.abs(y[None,:]-candidate)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(replicates):
        ids=rng.integers(0,len(y),len(y))
        denominator=np.abs(y[ids]).sum()
        if denominator<=0: raise ValueError("Invalid bootstrap denominator")
        values.append(float(50*contributions[:,ids].sum(axis=1).mean()/denominator))
    return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),
                percentile95=np.quantile(values,[.025,.975]).tolist(),
                replicates=replicates,scope="conditional_descriptive_row_cluster_not_independent_training_sets")
