"""Fixed full-original-training-row leaf distribution; no RF mean prediction."""
import numpy as np
from fractions import Fraction
from sklearn.ensemble import RandomForestRegressor

PROTOCOL = 'QRF_FULLTRAIN_LEAF_v1'
PARAMETERS = dict(n_estimators=256, criterion='squared_error', max_depth=None,
    min_samples_split=2, min_samples_leaf=10, min_weight_fraction_leaf=0.,
    max_features=.7, max_leaf_nodes=None, min_impurity_decrease=0., bootstrap=True,
    max_samples=None, oob_score=False, random_state=2026, n_jobs=8, verbose=0,
    warm_start=False, ccp_alpha=0., monotonic_cst=None)


def distribution_weights(members, n):
    weights = np.zeros(n, dtype=np.float64)
    for indices in members:
        if len(indices) == 0 or len(set(indices.tolist())) != len(indices):
            raise ValueError('empty or repeated full-training leaf members')
        weights[indices] += 1. / (len(members)*len(indices))
    mass = weights.sum()
    if not np.isfinite(weights).all() or (weights < 0).any() or abs(mass-1.) > 1e-12:
        raise ValueError('invalid leaf distribution mass')
    return weights


def lower_median(y, weights, members=None):
    order = np.argsort(y, kind='stable')
    cdf = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cdf, .5, side='left')), len(y)-1)
    if members is not None:
        # Resolve only rounding-ambiguous CDF boundaries using the exact rational
        # per-tree masses. This preserves a true lower median without a new threshold.
        bound = len(y)*np.finfo(np.float64).eps
        nearby = np.flatnonzero(np.abs(cdf-.5) <= bound)
        if len(nearby):
            candidates = sorted(set(y[order[nearby]].tolist()+[float(y[order[min(int(nearby[-1])+1,len(y)-1)]])]))
            for value in candidates:
                mass = sum((Fraction(int(np.count_nonzero(y[indices] <= value)), len(indices))
                            for indices in members), Fraction()) / len(members)
                if mass >= Fraction(1, 2): return float(value)
    return float(y[order[index]])


class QRF:
    def fit(self, x, y, ids):
        if len(set(ids)) != len(ids) or len(y) != len(ids) or len(x) != len(ids):
            raise ValueError('training response/ID alignment differs')
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('forest requires finite float32 input')
        y = np.asarray(y, dtype=np.float64)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('training responses must be finite nonnegative')
        self.forest = RandomForestRegressor(**PARAMETERS)
        self.forest.fit(x, y)
        self.y, self.ids = y, list(ids)
        self.leaves = []
        for tree in self.forest.estimators_:
            leaf_ids = tree.apply(x)
            mapping = {int(leaf): np.flatnonzero(leaf_ids == leaf) for leaf in np.unique(leaf_ids)}
            if not np.array_equal(np.sort(np.concatenate(list(mapping.values()))), np.arange(len(ids))):
                raise ValueError('original training rows not partitioned exactly once')
            self.leaves.append(mapping)
        return self

    def predict(self, x, training_months=None):
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('prediction requires finite float32 input')
        leaf_ids = np.column_stack([tree.apply(x) for tree in self.forest.estimators_])
        median, mean, diagnostics = [], [], []
        for row in leaf_ids:
            members = [self.leaves[b][int(leaf)] for b, leaf in enumerate(row)]
            w = distribution_weights(members, len(self.y))
            median.append(lower_median(self.y, w, members))
            mean.append(float(np.sum(self.y*w)))
            d = {'effective_neighbors': float(1./np.sum(w*w)), 'maximum_weight': float(w.max())}
            if training_months is not None:
                d['training_month_weights'] = {str(m): float(w[np.asarray(training_months)==m].sum())
                                               for m in sorted(set(training_months))}
            diagnostics.append(d)
        return np.array(median), np.array(mean), diagnostics
