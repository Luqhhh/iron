"""Finite-grid Gaussian Bayesian backfitting with reversible grow/prune moves."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import resource

import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2

from .data import FEATURES, TARGETS
from .v7_periodic import digest
from .v30_deep_kernel import select_weight
from .v42_splines import SplinePreprocessor


def leaf_log_integral(n, total, sigma2, tau2):
    """Integrated leaf log density minus the common zero-mean Normal density."""
    return -.5*math.log1p(n*tau2/sigma2) + .5*tau2*total**2/(sigma2*(sigma2+n*tau2))


def leaf_conditional(n, total, sigma2, tau2):
    variance = sigma2*tau2/(sigma2+n*tau2)
    return tau2*total/(sigma2+n*tau2), variance


def mixture_median(means, sigmas, steps=60):
    """Rows are independent queries; columns are retained posterior draws."""
    means, sigmas = np.asarray(means, float), np.asarray(sigmas, float)
    if means.ndim != 2 or sigmas.shape != (means.shape[1],) or np.any(sigmas <= 0):
        raise ValueError("Invalid Normal mixture shape or scales")
    if not np.isfinite(means).all() or not np.isfinite(sigmas).all():
        raise ValueError("Nonfinite Normal mixture")
    lo, hi = means.min(axis=1), means.max(axis=1)
    for _ in range(steps):
        mid = (lo+hi)/2
        cdf = ndtr((mid[:, None]-means)/sigmas[None, :]).mean(axis=1)
        lo, hi = np.where(cdf < .5, mid, lo), np.where(cdf >= .5, mid, hi)
    return (lo+hi)/2


@dataclass(frozen=True)
class Node:
    rows: np.ndarray
    choices: np.ndarray
    depth: int
    clauses: tuple


class NodeSpace:
    def __init__(self, x, cuts, max_depth, settings):
        self.x, self.cuts, self.max_depth, self.settings = x, cuts, max_depth, settings
        self.left = np.column_stack([x[:, feature] <= cut for feature, cut in cuts]) if cuts else np.empty((len(x),0), bool)
        self.cache = OrderedDict()
        self.root = self.node(np.arange(len(x)), 0, ())

    def node(self, rows, depth, clauses):
        key = (depth, tuple(sorted(clauses)))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if depth >= self.max_depth:
            choices = np.empty(0, int)
        else:
            counts = self.left[rows].sum(axis=0)
            minimum = self.settings["min_leaf_rows"]
            choices = np.flatnonzero((counts >= minimum)&(len(rows)-counts >= minimum))
        result = Node(rows, choices, depth, key[1])
        self.cache[key] = result
        if len(self.cache) > self.settings["node_cache_max"]:
            self.cache.popitem(last=False)
        return result

    def children(self, parent, cut):
        mask = self.left[parent.rows, cut]
        return (self.node(parent.rows[mask], parent.depth+1, parent.clauses+((cut, True),)),
                self.node(parent.rows[~mask], parent.depth+1, parent.clauses+((cut, False),)))

    def split_probability(self, node):
        return (self.settings["tree_prior_alpha"]/(1+node.depth)**self.settings["tree_prior_beta"]
                if len(node.choices) else 0.)


class Tree:
    def __init__(self, root):
        self.nodes, self.leaves, self.splits, self.values = {0:root}, {0:root}, {}, {0:0.}

    def copy(self):
        result = Tree(self.nodes[0])
        result.nodes, result.leaves = self.nodes.copy(), self.leaves.copy()
        result.splits, result.values = self.splits.copy(), self.values.copy()
        return result

    def options(self):
        grow = [i for i in sorted(self.leaves) if len(self.leaves[i].choices)]
        prune = [i for i in sorted(self.splits) if 2*i+1 in self.leaves and 2*i+2 in self.leaves]
        return grow, prune

    def changed(self, action, index, space, cut=None):
        result = self.copy()
        a,b = 2*index+1,2*index+2
        if action == "grow":
            parent = result.leaves.pop(index)
            if cut not in parent.choices:
                raise ValueError("Inadmissible split")
            left,right = space.children(parent,cut)
            result.nodes.update({a:left,b:right});result.leaves.update({a:left,b:right})
            result.splits[index] = int(cut)
        elif action == "prune":
            result.leaves.pop(a);result.leaves.pop(b)
            result.nodes.pop(a);result.nodes.pop(b)
            result.splits.pop(index)
            result.leaves[index] = result.nodes[index]
        else:
            raise ValueError("Unknown tree move")
        return result

    def log_prior(self, space):
        return sum(math.log(space.split_probability(self.nodes[i]))-math.log(len(self.nodes[i].choices))
                   for i in self.splits) + sum(math.log1p(-space.split_probability(n)) for n in self.leaves.values())

    def log_integral(self, residual, sigma2, tau2):
        return sum(leaf_log_integral(len(n.rows),float(residual[n.rows].sum()),sigma2,tau2)
                   for n in self.leaves.values())

    def sample_leaves(self, residual, sigma2, tau2, rng):
        prediction = np.empty(len(residual))
        self.values = {}
        for i in sorted(self.leaves):
            rows = self.leaves[i].rows
            mean,var = leaf_conditional(len(rows),float(residual[rows].sum()),sigma2,tau2)
            value = float(rng.normal(mean,math.sqrt(var)))
            self.values[i] = value;prediction[rows] = value
        return prediction

    def snapshot(self):
        return {"splits":[[int(i),int(v)] for i,v in sorted(self.splits.items())],
                "values":[[int(i),float(v)] for i,v in sorted(self.values.items())]}


def proposal_probability(tree, action, index):
    grow,prune = tree.options()
    available = int(bool(grow))+int(bool(prune))
    if action == "grow" and index in grow:
        return 1/(available*len(grow)*len(tree.leaves[index].choices))
    if action == "prune" and index in prune:
        return 1/(available*len(prune))
    return 0.


def log_acceptance(tree, proposed, action, index, space, residual, sigma2, tau2):
    reverse = "prune" if action == "grow" else "grow"
    forward_q = proposal_probability(tree,action,index)
    reverse_q = proposal_probability(proposed,reverse,index)
    if min(forward_q,reverse_q) <= 0:
        raise ValueError("Tree proposal lacks reverse support")
    return (proposed.log_prior(space)-tree.log_prior(space)
            +proposed.log_integral(residual,sigma2,tau2)-tree.log_integral(residual,sigma2,tau2)
            +math.log(reverse_q)-math.log(forward_q))


def update_tree(tree, space, residual, sigma2, tau2, rng):
    grow,prune = tree.options()
    actions = (["grow"] if grow else [])+(["prune"] if prune else [])
    action,accepted = "none",False
    if actions:
        action = actions[int(rng.integers(len(actions)))]
        pool = grow if action == "grow" else prune
        index = pool[int(rng.integers(len(pool)))]
        cut = (int(rng.choice(tree.leaves[index].choices)) if action == "grow" else None)
        proposed = tree.changed(action,index,space,cut)
        ratio = log_acceptance(tree,proposed,action,index,space,residual,sigma2,tau2)
        if math.log(max(float(rng.random()),np.finfo(float).tiny)) < min(0.,ratio):
            tree,accepted = proposed,True
    prediction = tree.sample_leaves(residual,sigma2,tau2,rng)
    return tree,prediction,action,accepted


def predict_tree(x, cuts, snapshot):
    split = dict(snapshot["splits"]);values = dict(snapshot["values"])
    prediction = np.empty(len(x))
    pending = [(0,np.arange(len(x)))]
    while pending:
        index,rows = pending.pop()
        if index in split:
            feature,knot = cuts[split[index]]
            mask = x[rows,feature] <= knot
            pending.extend([(2*index+1,rows[mask]),(2*index+2,rows[~mask])])
        else:
            prediction[rows] = values[index]
    return prediction


class BartRegressor:
    def __init__(self, recipe, settings):
        self.recipe,self.settings = recipe,deepcopy(settings)

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
        self.draws_,self.trace_ = [],[]
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
            if sweep > self.settings["burn_sweeps"] and (sweep-self.settings["burn_sweeps"])%self.settings["thin"] == 0:
                self.draws_.append({"sweep":sweep,"sigma":math.sqrt(sigma2),"trees":[t.snapshot() for t in trees]})
        return self

    def predict(self, frame):
        x = self.preprocessor_.transform(frame)
        means = np.empty((len(x),len(self.draws_)))
        for j,draw in enumerate(self.draws_):
            column = np.zeros(len(x))
            for tree in draw["trees"]:
                column += predict_tree(x,self.cuts_,tree)
            means[:,j] = column
        result = mixture_median(means,[d["sigma"] for d in self.draws_],self.settings["median_bisection_steps"])
        return result*self.scale_+self.center_

    def metadata(self):
        return {"recipe":self.recipe,"fit_ids_digest":digest(self.fit_ids_),"fit_rows":len(self.fit_ids_),
                "preprocessing":self.preprocessor_.metadata(),"target_center":self.center_,"target_scale":self.scale_,
                "prior":self.prior_,"trace":self.trace_,"retained_draws":len(self.draws_),
                "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}

    def save(self,path):
        with Path(path).open("x") as stream:
            json.dump({"recipe":self.recipe,"settings":self.settings,"preprocessing":self.preprocessor_.metadata(),
                       "center":self.center_,"scale":self.scale_,"cuts":self.cuts_,"draws":self.draws_,
                       "prior":self.prior_,"trace":self.trace_},stream,allow_nan=False)

    @classmethod
    def load(cls,path):
        saved = json.loads(Path(path).read_text());obj = cls(saved["recipe"],saved["settings"])
        obj.preprocessor_ = SplinePreprocessor.restore(saved["preprocessing"])
        obj.center_,obj.scale_,obj.cuts_ = saved["center"],saved["scale"],saved["cuts"]
        obj.draws_,obj.prior_,obj.trace_ = saved["draws"],saved["prior"],saved["trace"]
        return obj


def fit_partition(fitting, calibration, outer_training, query, target, recipe, settings, calibration_base, grid):
    if (set(fitting.sample_id)&set(calibration.sample_id)
            or set(outer_training.sample_id)!=set(fitting.sample_id)|set(calibration.sample_id)
            or set(outer_training.sample_id)&set(query.sample_id)):
        raise ValueError("Invalid nested partitions")
    if any(t in query for t in TARGETS):
        raise ValueError("Query labels must be removed")
    selector = BartRegressor(recipe,settings).fit(fitting,fitting[target].to_numpy())
    cp = selector.predict(calibration)
    weight,losses = select_weight(calibration[target],calibration_base,cp,grid)
    final = BartRegressor(recipe,settings).fit(outer_training,outer_training[target].to_numpy())
    final.calibration_model_ = selector
    return final,final.predict(query),{"weight":weight,"calibration_mae_by_weight":losses,
                                      "calibration":selector.metadata(),"refit":final.metadata()},cp
