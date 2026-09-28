"""V23 train-only histogram target prediction. No packaging or uploads."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .data import FEATURES
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import PeriodicRegressor, digest

@dataclass
class HistogramGrid:
    edges: np.ndarray
    sigma_bins: float = .75

    @classmethod
    def fit(cls, y, bins=64, padding=.05, sigma_bins=.75):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.ptp(y) <= 0:
            raise ValueError('Nonempty finite nonconstant targets required')
        if not isinstance(bins, int) or bins < 2 or padding < 0 or sigma_bins <= 0:
            raise ValueError('Invalid histogram specification')
        spread = np.ptp(y)
        return cls(np.linspace(y.min()-padding*spread, y.max()+padding*spread, bins+1), sigma_bins)

    @property
    def sigma(self):
        return float((self.edges[1]-self.edges[0])*self.sigma_bins)

    def encode(self, y, arm):
        from scipy.special import ndtr
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError('Finite one-dimensional targets required')
        index = np.clip(np.searchsorted(self.edges, y, side='right')-1, 0, len(self.edges)-2)
        if arm == 'hard':
            mass = np.zeros((len(y), len(self.edges)-1), dtype=np.float64)
            mass[np.arange(len(y)), index] = 1.
            return mass
        if arm != 'gaussian':
            raise ValueError('Unknown target distribution')
        if (y < self.edges[0]).any() or (y > self.edges[-1]).any():
            raise ValueError('Gaussian training target lies outside fitted support')
        cdf = ndtr((self.edges[None, :]-y[:, None])/self.sigma)
        mass = np.maximum(np.diff(cdf, axis=1), 0.)
        return mass / mass.sum(axis=1, keepdims=True)

def histogram_median(mass, edges):
    mass = np.asarray(mass, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if (mass.ndim != 2 or edges.shape != (mass.shape[1]+1,)
            or not np.isfinite(mass).all() or (mass < 0).any()
            or not np.isfinite(edges).all() or (np.diff(edges) <= 0).any()
            or not np.allclose(mass.sum(1), 1., atol=1e-6, rtol=0)):
        raise ValueError('Valid simplex and increasing histogram support required')
    mass = mass / mass.sum(1, keepdims=True)
    cdf = mass.cumsum(1)
    cdf[:, -1] = 1.
    index = np.argmax(cdf >= .5, axis=1)
    rows = np.arange(len(mass))
    before = np.where(index > 0, cdf[rows, np.maximum(index-1, 0)], 0.)
    fraction = (.5-before)/mass[rows, index]
    return edges[index] + fraction*(edges[index+1]-edges[index])

def mixture_probabilities(logits):
    import torch
    if logits.ndim != 3:
        raise ValueError('Expected rows, heads, bins')
    return torch.softmax(logits, dim=-1).mean(1)

def histogram_loss(logits, target):
    import torch
    if logits.ndim != 3 or target.shape != (logits.shape[0], logits.shape[-1]):
        raise ValueError('Histogram loss shape mismatch')
    return -(target[:, None, :]*torch.log_softmax(logits, -1)).sum(-1).mean()

def make_histogram_network(settings, n_categories):
    from rtdl_num_embeddings import PeriodicEmbeddings
    from tabm import TabM
    embedding = PeriodicEmbeddings(len(FEATURES), d_embedding=settings['embedding_dim'],
        n_frequencies=settings['n_frequencies'], frequency_init_scale=.01, lite=settings['lite'])
    return TabM.make(n_num_features=len(FEATURES), cat_cardinalities=[n_categories],
        d_out=64, k=settings['tabm_k'], n_blocks=settings['blocks'],
        d_block=settings['width'], dropout=settings['dropout'], num_embeddings=embedding)

class HistogramRegressor(PeriodicRegressor):
    def __init__(self, arm, settings):
        if arm not in ('hard', 'gaussian'):
            raise ValueError('Unknown histogram arm')
        super().__init__({'backbone': 'tabm', 'frequency': .01}, settings)
        self.arm = arm

    def _initialize(self, frame, y):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(self.settings['random_seed'])
        self.preprocessor_ = NumericPreprocessor(structure='raw_tabm').fit(frame)
        self.mean_, self.std_ = float(np.mean(y)), float(np.std(y))
        self.grid_ = HistogramGrid.fit(y)
        self.model_ = make_histogram_network(self.settings, self.preprocessor_.n_spout_categories_)
        self.optimizer_ = torch.optim.AdamW(self.model_.parameters(),
            lr=self.settings['learning_rate'], weight_decay=self.settings['weight_decay'])

    def _train(self, frame, y, epochs, validation=None):
        import torch
        x, cat = self._inputs(frame)
        target = torch.as_tensor(self.grid_.encode(y, self.arm), dtype=torch.float32)
        rng = np.random.default_rng(self.settings['random_seed'])
        best, best_epoch, stale = float('inf'), 0, 0
        trace = []
        for epoch in range(1, epochs+1):
            self.model_.train()
            total = 0.
            order = rng.permutation(len(frame))
            for start in range(0, len(order), self.settings['batch_size']):
                idx = order[start:start+self.settings['batch_size']]
                self.optimizer_.zero_grad(set_to_none=True)
                loss = histogram_loss(self.model_(x[idx], cat[idx]), target[idx])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite histogram loss')
                loss.backward()
                self.optimizer_.step()
                total += float(loss.detach())*len(idx)
            row = {'epoch': epoch, 'train_ce': total/len(frame)}
            if validation is not None:
                prediction = self.predict(validation[0])
                value = float(np.abs(prediction-validation[1]).mean()/self.std_)
                row['inner_standardized_median_mae'] = value
                if value < best-self.settings['min_delta']:
                    best, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
            trace.append(row)
            if validation is not None and stale >= self.settings['patience']:
                break
        return (best_epoch if validation is not None else epochs), trace

    def fit(self, frame, y):
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(frame),):
            raise ValueError('Training target row mismatch')
        HistogramGrid.fit(y)
        mask = group_safe_inner_folds(frame, seed=self.settings['inner_seed'])['fold'] != 0
        inner = frame.loc[mask].reset_index(drop=True)
        self._initialize(inner, y[mask])
        inner_support, inner_sigma = self.grid_.edges[[0, -1]].tolist(), self.grid_.sigma
        inner_means = self.preprocessor_.means_.tolist()
        selected, inner_trace = self._train(inner, y[mask], self.settings['max_epochs'],
            (frame.loc[~mask], y[~mask]))
        if selected < 1:
            raise ValueError('No finite checkpoint')
        self._initialize(frame, y)
        _, outer_trace = self._train(frame, y, selected)
        self.model_.eval()
        mass = self.predict_mass(frame)
        midpoint = (self.grid_.edges[:-1]+self.grid_.edges[1:])/2
        self.metadata_ = {
            'arm': self.arm, 'selected_epoch': selected, 'selection_executed_epochs': len(inner_trace),
            'epoch_cap_hit': len(inner_trace) == self.settings['max_epochs'],
            'fit_rows': len(frame), 'inner_fit_rows': int(mask.sum()), 'optimizer_runs': 2,
            'fit_ids_digest': digest(frame.sample_id.tolist()),
            'inner_ids_digest': digest(frame.loc[mask, 'sample_id'].tolist()),
            'inner_feature_means': inner_means, 'outer_feature_means': self.preprocessor_.means_.tolist(),
            'inner_support': inner_support, 'outer_support': self.grid_.edges[[0, -1]].tolist(),
            'inner_sigma': inner_sigma, 'outer_sigma': self.grid_.sigma, 'bins': 64,
            'inner_heldout_outside_support': int(((y[~mask] < inner_support[0]) | (y[~mask] > inner_support[1])).sum()),
            'inner_trace': inner_trace, 'outer_trace': outer_trace,
            'train_entropy': float(-(mass*np.log(np.maximum(mass, 1e-300))).sum(1).mean()),
            'diagnostic_train_mean_mae': float(np.abs(mass@midpoint-y).mean()),
            'diagnostic_train_median_mae': float(np.abs(histogram_median(mass, self.grid_.edges)-y).mean()),
            'parameter_count': sum(p.numel() for p in self.model_.parameters()),
        }
        return self

    def predict_mass(self, frame):
        import torch
        x, cat = self._inputs(frame)
        self.model_.eval()
        with torch.no_grad():
            return mixture_probabilities(self.model_(x, cat)).numpy().astype(np.float64)

    def predict(self, frame):
        return histogram_median(self.predict_mass(frame), self.grid_.edges)

def select_finalist(records):
    expected = {(t, a) for t in ('tap_iron', 'tap_time_len') for a in ('hard', 'gaussian')}
    if len(records) != 4 or {(r['target'], r['arm']) for r in records} != expected:
        raise ValueError('Complete frozen candidate pool required')
    if any(set(r['seed_gains']) != {'42', '3407'} for r in records):
        raise ValueError('Complete development seeds required')
    controls = {r['target']: r for r in records if r['arm'] == 'hard'}
    eligible = []
    for row in records:
        if row['arm'] != 'gaussian':
            continue
        gain = np.asarray(list(row['seed_gains'].values()))
        control = np.asarray(list(controls[row['target']]['seed_gains'].values()))
        if (gain > 0).all() and gain.mean() >= .01 and (gain-control).mean() > 0:
            eligible.append((float(gain.mean()), row['target']))
    return min(eligible, key=lambda x: (-x[0], ('tap_iron', 'tap_time_len').index(x[1])))[1] if eligible else None
