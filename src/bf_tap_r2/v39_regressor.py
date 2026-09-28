"""Train-only preprocessing, selection and fresh refit of frozen V37 trees."""
from copy import deepcopy
from pathlib import Path
import resource

import numpy as np
from sklearn.preprocessing import QuantileTransformer
import torch

from .data import FEATURES, TARGETS
from .v7_periodic import digest
from .v30_deep_kernel import select_weight
from .v37_hard_tree import HardTreeEnsemble


class TreePreprocessor:
    def fit(self, frame):
        numeric = frame[list(FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite numeric training inputs")
        self.transformer_ = QuantileTransformer(n_quantiles=min(1000, len(frame)),
            output_distribution="normal", subsample=len(frame), random_state=42).fit(numeric)
        self.categories_ = sorted(int(v) for v in frame.spout_no.unique())
        self.fit_ids_digest_ = digest(frame.sample_id.tolist())
        return self

    def transform(self, frame):
        numeric = frame[list(FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite query inputs")
        onehot = np.column_stack([frame.spout_no.to_numpy() == c for c in self.categories_])
        return np.concatenate([self.transformer_.transform(numeric), onehot], axis=1).astype(np.float32)

    def metadata(self):
        return {"quantiles": self.transformer_.quantiles_.tolist(),
                "references": self.transformer_.references_.tolist(),
                "categories": self.categories_, "fit_ids_digest": self.fit_ids_digest_}

    @classmethod
    def from_metadata(cls, info):
        obj = cls()
        obj.categories_ = info["categories"]
        obj.fit_ids_digest_ = info["fit_ids_digest"]
        obj.transformer_ = QuantileTransformer(output_distribution="normal", random_state=42)
        obj.transformer_.quantiles_ = np.asarray(info["quantiles"])
        obj.transformer_.references_ = np.asarray(info["references"])
        obj.transformer_.n_quantiles_ = len(info["references"])
        obj.transformer_.n_features_in_ = len(FEATURES)
        return obj


class TreeRegressor:
    def __init__(self, recipe, settings):
        self.recipe, self.settings = recipe, deepcopy(settings)

    def initialize(self, frame, y):
        torch.set_num_threads(1)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or y.std() <= 0:
            raise ValueError("Invalid target")
        self.preprocessor_ = TreePreprocessor().fit(frame)
        self.mean_, self.std_ = float(y.mean()), max(float(y.std()), 1e-8)
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        self.x_train_ = torch.from_numpy(self.preprocessor_.transform(frame))
        self.y_train_ = torch.tensor((y-self.mean_)/self.std_, dtype=torch.float32)
        self.model_ = HardTreeEnsemble(self.x_train_.shape[1], self.recipe, self.settings["model"])
        self.optimizer_ = torch.optim.Adam(
            [{"params": [getattr(self.model_, name)], "lr": rate}
             for name, rate in self.settings["learning_rates"].items()],
            betas=tuple(self.settings["betas"]), eps=self.settings["eps"],
            weight_decay=self.settings["weight_decay"])
        return self

    def train(self, epochs, validation=None):
        if epochs < 1:
            raise ValueError("Positive epoch count required")
        rng = np.random.default_rng(self.settings["random_seed"])
        best, stale, best_epoch, best_state = float("inf"), 0, 0, None
        self.history_ = []
        for epoch in range(1, epochs+1):
            self.model_.train()
            order, total = rng.permutation(len(self.y_train_)), 0.
            for start in range(0, len(order), self.settings["batch_size"]):
                index = order[start:start+self.settings["batch_size"]]
                self.optimizer_.zero_grad(set_to_none=True)
                loss = (self.model_(self.x_train_[index])-self.y_train_[index]).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model_.parameters()):
                    raise ValueError("Nonfinite or missing gradient")
                self.optimizer_.step()
                total += float(loss.detach())*len(index)
            row = {"epoch": epoch, "standardized_mse": total/len(order)}
            if validation is not None:
                value = float(np.abs(self.predict(validation[0])-validation[1]).mean()/self.std_)
                row["calibration_standardized_mae"] = value
                if value < best-self.settings["min_delta"]:
                    best, best_epoch, stale = value, epoch, 0
                    best_state = deepcopy(self.model_.state_dict())
                else:
                    stale += 1
            self.history_.append(row)
            if validation is not None and stale >= self.settings["patience"]:
                break
        if validation is not None:
            if best_state is None:
                raise ValueError("No selected epoch")
            self.model_.load_state_dict(best_state)
        else:
            best_epoch = epoch
        self.selected_epoch_, self.stopped_epoch_ = best_epoch, epoch
        return best_epoch

    def predict(self, frame):
        self.model_.eval()
        x = torch.from_numpy(self.preprocessor_.transform(frame))
        chunk = self.settings["prediction_chunk_rows"]
        with torch.no_grad():
            prediction = np.concatenate([self.model_(x[i:i+chunk]).numpy().astype(float)
                                         for i in range(0, len(frame), chunk)])
        values = prediction*self.std_+self.mean_
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite prediction")
        return values

    def metadata(self):
        return {"recipe": self.recipe, "selected_epoch": self.selected_epoch_,
                "stopped_epoch": self.stopped_epoch_, "fit_ids_digest": digest(self.fit_ids_),
                "fit_rows": len(self.fit_ids_), "preprocessing": self.preprocessor_.metadata(),
                "target_mean": self.mean_, "target_std": self.std_, "history": self.history_,
                "parameter_count": sum(p.numel() for p in self.model_.parameters()),
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}

    def save(self, path):
        with Path(path).open("xb") as stream:
            torch.save({"recipe": self.recipe, "settings": self.settings,
                        "preprocessing": self.preprocessor_.metadata(), "mean": self.mean_,
                        "std": self.std_, "state": self.model_.state_dict()}, stream)

    @classmethod
    def load(cls, path):
        torch.set_num_threads(1)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(payload["recipe"], payload["settings"])
        obj.preprocessor_ = TreePreprocessor.from_metadata(payload["preprocessing"])
        obj.mean_, obj.std_ = payload["mean"], payload["std"]
        obj.model_ = HardTreeEnsemble(len(FEATURES)+len(obj.preprocessor_.categories_), obj.recipe,
                                     obj.settings["model"])
        obj.model_.load_state_dict(payload["state"])
        return obj


def fit_partition(fitting, calibration, outer_training, query, target, recipe,
                  settings, calibration_base, grid):
    if (set(fitting.sample_id) & set(calibration.sample_id)
            or set(outer_training.sample_id) != set(fitting.sample_id) | set(calibration.sample_id)
            or set(outer_training.sample_id) & set(query.sample_id)):
        raise ValueError("Invalid nested partitions")
    if any(c in query for c in TARGETS):
        raise ValueError("Query labels must be removed")
    selector = TreeRegressor(recipe, settings).initialize(fitting, fitting[target].to_numpy())
    epoch = selector.train(settings["max_epochs"], (calibration, calibration[target].to_numpy()))
    cp = selector.predict(calibration)
    weight, losses = select_weight(calibration[target], calibration_base, cp, grid)
    final = TreeRegressor(recipe, settings).initialize(outer_training, outer_training[target].to_numpy())
    final.train(epoch)
    prediction = final.predict(query)
    final.calibration_model_ = selector
    return final, prediction, {"weight": weight, "calibration_mae_by_weight": losses,
                              "calibration": selector.metadata(), "refit": final.metadata()}, cp
