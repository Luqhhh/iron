"""Normal CRPS boosting with an explicit integrated-CDF gradient metric."""
from __future__ import annotations

import math
from pathlib import Path
import pickle

import numpy as np
from scipy.special import ndtr
from sklearn.tree import DecisionTreeRegressor

from .data import TARGETS
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest
from .v30_deep_kernel import select_weight


def normal_crps(y, params):
    mu, sigma = params[:, 0], np.exp(params[:, 1])
    z = (y-mu)/sigma
    phi = np.exp(-.5*z*z)/math.sqrt(2*math.pi)
    return sigma*(z*(2*ndtr(z)-1)+2*phi-1/math.sqrt(math.pi))


def crps_gradient(y, params):
    sigma = np.exp(params[:, 1])
    z = (y-params[:, 0])/sigma
    phi = np.exp(-.5*z*z)/math.sqrt(2*math.pi)
    return np.column_stack((1-2*ndtr(z), sigma*(2*phi-1/math.sqrt(math.pi))))


def crps_metric_diagonal(params):
    sigma = np.exp(params[:, 1])
    return np.column_stack((1/(sigma*math.sqrt(math.pi)), sigma/(2*math.sqrt(math.pi))))


def advance(params, direction, step, settings):
    result = params - settings["learning_rate"]*step*direction
    result[:, 1] = np.clip(result[:, 1], settings["log_scale_min"], settings["log_scale_max"])
    return result


def training_step(y, params, direction, settings):
    old_score = float(normal_crps(y, params).mean())
    for step in settings["line_search_steps"]:
        candidate = advance(params, direction, step, settings)
        score = float(normal_crps(y, candidate).mean())
        if np.isfinite(score) and score <= old_score:
            return float(step), candidate, score
    raise ValueError("No valid training CRPS step, including frozen zero fallback")


class CRPSRegressor:
    def __init__(self, recipe, settings):
        if recipe not in ("CRPS_FIXED", "CRPS_SCALE"):
            raise ValueError("Unknown CRPS recipe")
        self.recipe, self.settings = recipe, dict(settings)

    def _inputs(self, frame):
        numeric, cat = self.preprocessor_.transform_mlp(frame)
        x = np.concatenate((numeric, cat), 1).astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite CRPS inputs")
        return x

    def initialize(self, frame, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or not y.std() > 0:
            raise ValueError("Invalid training target")
        self.preprocessor_ = NumericPreprocessor(structure="raw_mlp").fit(frame)
        self.mean_, self.std_ = float(y.mean()), float(y.std())
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        self.x_train_, self.y_train_ = self._inputs(frame), (y-self.mean_)/self.std_
        self.trees_, self.steps_ = [], []
        return self

    def train(self, epochs, validation=None):
        params = np.zeros((len(self.y_train_), 2))
        self.history_ = []
        best, stale, best_epoch = float("inf"), 0, 0
        if validation is not None:
            val_x, val_y = self._inputs(validation[0]), np.asarray(validation[1])
            val_params = np.zeros((len(val_y), 2))
        dimensions = 1 if self.recipe == "CRPS_FIXED" else 2
        for epoch in range(1, epochs+1):
            gradient = crps_gradient(self.y_train_, params)/crps_metric_diagonal(params)
            if not np.isfinite(gradient).all():
                raise ValueError("Nonfinite natural gradient")
            trees, direction = [], np.zeros_like(params)
            for dim in range(dimensions):
                tree = DecisionTreeRegressor(criterion="squared_error",
                    random_state=self.settings["random_seed"], max_depth=self.settings["max_depth"],
                    min_samples_leaf=self.settings["min_samples_leaf"],
                    min_samples_split=self.settings["min_samples_split"],
                    max_features=self.settings["max_features"])
                tree.fit(self.x_train_, gradient[:, dim])
                trees.append(tree)
                direction[:, dim] = tree.predict(self.x_train_)
            step, params, score = training_step(self.y_train_, params, direction, self.settings)
            self.trees_.append(trees)
            self.steps_.append(step)
            row = {"epoch":epoch, "training_crps":score, "step":step,
                   "scale_boundary_fraction":float(np.mean((params[:,1] == self.settings["log_scale_min"])
                                                 | (params[:,1] == self.settings["log_scale_max"])))}
            if validation is not None:
                direction = np.zeros_like(val_params)
                for dim, tree in enumerate(trees):
                    direction[:,dim] = tree.predict(val_x)
                val_params = advance(val_params, direction, step, self.settings)
                value = float(np.abs(val_params[:,0]*self.std_+self.mean_-val_y).mean()/self.std_)
                row["calibration_standardized_mae"] = value
                if value < best-self.settings["min_delta_standardized_mae"]:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
            self.history_.append(row)
            if validation is not None and stale >= self.settings["patience"]:
                break
        if validation is None:
            best_epoch = epoch
        elif best_epoch == 0:
            raise ValueError("No finite selected iteration")
        self.trees_, self.steps_ = self.trees_[:best_epoch], self.steps_[:best_epoch]
        self.selected_epoch_, self.stopped_epoch_ = best_epoch, epoch
        self.fitted_tree_count_ = dimensions*epoch
        return best_epoch

    def predict_params(self, frame):
        x = self._inputs(frame)
        params = np.zeros((len(frame),2))
        for trees, step in zip(self.trees_, self.steps_):
            direction = np.zeros_like(params)
            for dim, tree in enumerate(trees):
                direction[:,dim] = tree.predict(x)
            params = advance(params, direction, step, self.settings)
        if not np.isfinite(params).all():
            raise ValueError("Invalid predicted distribution parameters")
        return params

    def predict(self, frame):
        return self.predict_params(frame)[:,0]*self.std_+self.mean_

    def metadata(self):
        return {"recipe":self.recipe, "selected_epoch":self.selected_epoch_, "stopped_epoch":self.stopped_epoch_,
                "fit_ids_digest":digest(self.fit_ids_), "fit_rows":len(self.fit_ids_),
                "preprocessing":self.preprocessor_.metadata(), "target_mean":self.mean_, "target_std":self.std_,
                "history":self.history_, "fitted_tree_count":self.fitted_tree_count_,
                "retained_tree_count":sum(map(len,self.trees_)), "zero_steps":self.steps_.count(0.)}

    def save(self, path):
        payload = {"recipe":self.recipe,"settings":self.settings,"preprocessing":self.preprocessor_.metadata(),
                   "mean":self.mean_,"std":self.std_,"trees":self.trees_,"steps":self.steps_}
        with Path(path).open("xb") as stream:
            pickle.dump(payload,stream,protocol=5)

    @classmethod
    def load(cls, path):
        # Only load our hash-verified local model artifacts, never foreign pickles.
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        model = cls(payload["recipe"],payload["settings"])
        info = payload["preprocessing"]
        model.preprocessor_ = NumericPreprocessor(structure="raw_mlp")
        model.preprocessor_.means_ = np.array(info["means"])
        model.preprocessor_.stds_ = np.array(info["stds"])
        model.preprocessor_.spout_to_index_ = {int(k):v for k,v in info["spout_vocabulary"].items()}
        model.preprocessor_.n_spout_categories_ = info["n_spout_categories"]
        model.mean_,model.std_,model.trees_,model.steps_ = payload["mean"],payload["std"],payload["trees"],payload["steps"]
        return model


def fit_partition(fitting, calibration, outer_training, query, target, recipe, settings, calibration_base, grid):
    if set(fitting.sample_id) & set(calibration.sample_id):
        raise ValueError("Fitting/calibration overlap")
    if set(outer_training.sample_id) != set(fitting.sample_id) | set(calibration.sample_id):
        raise ValueError("Incorrect outer training union")
    if set(outer_training.sample_id) & set(query.sample_id):
        raise ValueError("Outer query overlaps training")
    if any(c in query.columns for c in TARGETS):
        raise ValueError("Query labels must be removed")
    selector = CRPSRegressor(recipe,settings).initialize(fitting,fitting[target].to_numpy())
    epoch = selector.train(settings["max_epochs"],(calibration,calibration[target].to_numpy()))
    cp = selector.predict(calibration)
    weight,losses = select_weight(calibration[target],calibration_base,cp,grid)
    final = CRPSRegressor(recipe,settings).initialize(outer_training,outer_training[target].to_numpy())
    final.train(epoch)
    prediction = final.predict(query)
    final.calibration_model_ = selector
    return final,prediction,{"weight":weight,"calibration_mae_by_weight":losses,
                             "calibration":selector.metadata(),"refit":final.metadata()},cp
