"""Long fixed BART schedule; unchanged V43 kernel and exact old-prefix capture."""
from copy import copy, deepcopy
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .data import FEATURES, TARGETS
from .v7_periodic import digest
from .v30_deep_kernel import select_weight
from .v43_bart import BartRegressor, NodeSpace, Tree, SplinePreprocessor, update_tree


class LongBartRegressor(BartRegressor):
    PREFIX = {"burn_sweeps":200,"thin":2,"sweeps":400}

    def fit(self, frame, y):
        y = np.asarray(y,float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or np.ptp(y) <= 0:
            raise ValueError("Invalid training target")
        self.preprocessor_ = SplinePreprocessor().fit(frame)
        self.center_,self.scale_ = float((y.max()+y.min())/2),float(y.max()-y.min())
        z = (y-self.center_)/self.scale_
        x = self.preprocessor_.transform(frame)
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        self.cuts_ = []
        for feature in range(x.shape[1]):
            cuts = ([self.settings["categorical_knot"]] if feature >= len(FEATURES)
                    else np.unique(np.quantile(x[:,feature],self.settings["knot_quantiles"])))
            self.cuts_.extend([[feature,float(v)] for v in cuts])
        space = NodeSpace(x,self.cuts_,self.settings["max_depth"][self.recipe],self.settings)
        rng = np.random.default_rng(self.settings["seed"])
        m = self.settings["trees"]
        tau2 = (.5/(self.settings["leaf_prior_k"]*math.sqrt(m)))**2
        nu = self.settings["sigma_prior_df"]
        sigma2 = float(z.var(ddof=1))
        lam = sigma2*chi2.ppf(1-self.settings["sigma_prior_upper_probability"],nu)/nu
        self.prior_ = {"tau2":tau2,"nu":nu,"lambda":float(lam),"initial_sigma2":sigma2}
        trees = [Tree(space.root) for _ in range(m)]
        members,total = np.zeros((m,len(z))),np.zeros(len(z))
        self.draws_,self.trace_,self.prefix_draws_ = [],[],[]
        self.prefix_schedule_ = deepcopy(self.PREFIX)
        sweeps = self.settings["burn_sweeps"]+self.settings["retained_draws"]*self.settings["thin"]
        for sweep in range(1,sweeps+1):
            counts = {"grow_proposed":0,"grow_accepted":0,"prune_proposed":0,"prune_accepted":0,"none":0}
            for j in range(m):
                residual = z-total+members[j]
                trees[j],prediction,action,accepted = update_tree(trees[j],space,residual,sigma2,tau2,rng)
                total += prediction-members[j]
                members[j] = prediction
                if action == "none": counts["none"] += 1
                else:
                    counts[action+"_proposed"] += 1
                    counts[action+"_accepted"] += int(accepted)
            rss = float(np.sum((z-total)**2))
            sigma2 = float((rss+nu*lam)/(2*rng.gamma((len(z)+nu)/2)))
            if not np.isfinite(sigma2) or sigma2 <= 0:
                raise ValueError("Invalid sampled noise variance")
            self.trace_.append({"sweep":sweep,"rss":rss,"sigma2":sigma2,
                                "mean_prediction":float(total.mean()),
                                "total_leaves":sum(len(t.leaves) for t in trees),**counts})
            prefix = self.prefix_schedule_
            if prefix["burn_sweeps"] < sweep <= prefix["sweeps"] and (sweep-prefix["burn_sweeps"])%prefix["thin"] == 0:
                self.prefix_draws_.append({"sweep":sweep,"sigma":math.sqrt(sigma2),"trees":[t.snapshot() for t in trees]})
            if sweep > self.settings["burn_sweeps"] and (sweep-self.settings["burn_sweeps"])%self.settings["thin"] == 0:
                self.draws_.append({"sweep":sweep,"sigma":math.sqrt(sigma2),"trees":[t.snapshot() for t in trees]})
        return self

    def predict_prefix(self, frame):
        prefix = copy(self)
        prefix.draws_ = self.prefix_draws_
        return BartRegressor.predict(prefix, frame)

    def metadata(self):
        result = super().metadata()
        result.update(prefix_schedule=self.prefix_schedule_,
                      prefix_trace_digest=digest(self.trace_[:self.prefix_schedule_["sweeps"]]),
                      prefix_draws_digest=digest(self.prefix_draws_))
        return result

    def save(self,path):
        with Path(path).open("x") as stream:
            json.dump({"recipe":self.recipe,"settings":self.settings,"preprocessing":self.preprocessor_.metadata(),
                       "center":self.center_,"scale":self.scale_,"cuts":self.cuts_,"draws":self.draws_,
                       "prior":self.prior_,"trace":self.trace_,"prefix_draws":self.prefix_draws_,
                       "prefix_schedule":self.prefix_schedule_},stream,allow_nan=False)

    @classmethod
    def load(cls,path):
        saved=json.loads(Path(path).read_text());obj=cls(saved["recipe"],saved["settings"])
        obj.preprocessor_=SplinePreprocessor.restore(saved["preprocessing"])
        obj.center_,obj.scale_,obj.cuts_=saved["center"],saved["scale"],saved["cuts"]
        obj.draws_,obj.prior_,obj.trace_=saved["draws"],saved["prior"],saved["trace"]
        obj.prefix_draws_,obj.prefix_schedule_=saved["prefix_draws"],saved["prefix_schedule"]
        return obj


def verify_prefix(model, old_payload, query, expected):
    old_settings=old_payload["settings"]
    old_sweeps=old_settings["burn_sweeps"]+old_settings["retained_draws"]*old_settings["thin"]
    restored_settings=dict(model.settings,burn_sweeps=old_settings["burn_sweeps"],thin=old_settings["thin"])
    checks={"settings":restored_settings==old_settings,
            "preprocessing":model.preprocessor_.metadata()==old_payload["preprocessing"],
            "target":model.center_==old_payload["center"] and model.scale_==old_payload["scale"],
            "cuts":model.cuts_==old_payload["cuts"],"prior":model.prior_==old_payload["prior"],
            "trace":model.trace_[:old_sweeps]==old_payload["trace"],
            "draws":model.prefix_draws_==old_payload["draws"],
            "schedule":model.prefix_schedule_=={"burn_sweeps":old_settings["burn_sweeps"],"thin":old_settings["thin"],"sweeps":old_sweeps}}
    prediction=model.predict_prefix(query)
    difference=float(np.max(np.abs(prediction-expected)))
    checks["prediction_exact"]=np.array_equal(prediction,expected)
    if not all(checks.values()):
        raise ValueError(f"V43 prefix mismatch: {checks}; difference={difference}")
    return {"checks":checks,"max_prediction_difference":difference,"new_fits":0}


def fit_partition(fitting,calibration,outer_training,query,target,recipe,settings,calibration_base,grid):
    if (set(fitting.sample_id)&set(calibration.sample_id)
            or set(outer_training.sample_id)!=set(fitting.sample_id)|set(calibration.sample_id)
            or set(outer_training.sample_id)&set(query.sample_id)):
        raise ValueError("Invalid nested partitions")
    if any(t in query for t in TARGETS):
        raise ValueError("Query labels must be removed")
    selector=LongBartRegressor(recipe,settings).fit(fitting,fitting[target].to_numpy())
    cp=selector.predict(calibration)
    weight,losses=select_weight(calibration[target],calibration_base,cp,grid)
    final=LongBartRegressor(recipe,settings).fit(outer_training,outer_training[target].to_numpy())
    final.calibration_model_=selector
    return final,final.predict(query),{"weight":weight,"calibration_mae_by_weight":losses,
                                      "calibration":selector.metadata(),"refit":final.metadata()},cp
