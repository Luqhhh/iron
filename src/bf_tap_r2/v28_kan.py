"""Frozen V28 train-only preprocessing and regression orchestration."""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from .data import FEATURES


def __getattr__(name):
    if name in ('EdgeKAN', 'SplineLinear'):
        from . import v28_kan_core
        return getattr(v28_kan_core, name)
    raise AttributeError(name)


class KANPreprocessor:
    def _numeric(self, frame):
        value = frame.loc[:, FEATURES].to_numpy(dtype=float)
        if not np.isfinite(value).all():
            raise ValueError('Numeric features must be finite')
        return value

    def fit(self, frame):
        numeric = self._numeric(frame)
        categories = frame.spout_no.to_numpy(dtype=float)
        if len(frame) < 2 or not np.isfinite(categories).all():
            raise ValueError('Finite categories and at least two training rows required')
        self.vocabulary_ = np.sort(np.unique(categories))
        self.quantiles_ = QuantileTransformer(n_quantiles=min(1000, len(frame)),
            output_distribution='uniform', subsample=len(frame), random_state=42).fit(numeric)
        return self

    def transform(self, frame):
        numeric = 2 * self.quantiles_.transform(self._numeric(frame)) - 1
        categories = frame.spout_no.to_numpy(dtype=float)
        if not np.isfinite(categories).all():
            raise ValueError('Categories must be finite')
        onehot = categories[:, None] == self.vocabulary_[None, :]
        return np.concatenate([numeric, onehot], axis=1).astype(np.float32)

class TrainingSession:
    """One optimizer, with a mandatory durable start callback before construction."""
    def __init__(self, frame, y, arm, event, stage):
        import torch
        from .v28_kan_core import EdgeKAN
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all():
            raise ValueError('Invalid training target')
        torch.set_num_threads(1)
        self.preprocessor = KANPreprocessor().fit(frame)
        self.mean, self.std = float(y.mean()), max(float(y.std()), 1e-8)
        self.x = torch.from_numpy(self.preprocessor.transform(frame))
        self.y = torch.as_tensor((y - self.mean) / self.std, dtype=torch.float32)
        self.model = EdgeKAN(self.x.shape[1], arm)
        event({'event': 'optimizer_started', 'stage': stage, 'arm': arm})
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001,
            weight_decay=.0001, betas=(.9, .999), eps=1e-8)
        self.rng = np.random.default_rng(42)
        self.max_gradient_norm = 0.

    def step(self, indices):
        import torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = ((self.model(self.x[indices]).flatten() - self.y[indices]) ** 2).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        grads = [p.grad for p in self.model.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise ValueError('Nonfinite or missing gradient')
        norm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
        self.max_gradient_norm = max(self.max_gradient_norm, norm)
        self.optimizer.step()
        return float(loss.detach())

    def epoch(self):
        order = self.rng.permutation(len(self.x))
        return float(np.mean([self.step(order[i:i+256]) for i in range(0, len(order), 256)]))

    def predict(self, frame, chunk_rows=256):
        import torch
        if chunk_rows < 1:
            raise ValueError('Positive prediction chunk required')
        x = torch.from_numpy(self.preprocessor.transform(frame))
        self.model.eval()
        with torch.no_grad():
            values = []
            # Always evaluate 256 rows: batch shape can change float32 GEMM rounding.
            # Zero padding is safe because every KAN operation is row independent.
            stride = min(chunk_rows, 256)
            for i in range(0, len(x), stride):
                part = x[i:i+stride]
                padded = torch.zeros((256, x.shape[1]), dtype=x.dtype)
                padded[:len(part)] = part
                values.append(self.model(padded)[:len(part)].flatten().numpy().astype(float))
        result = (np.concatenate(values) if values else np.empty(0)) * self.std + self.mean
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite prediction')
        return result

    def snapshot(self):
        import torch
        from .v7_periodic import digest
        self.model.eval()
        with torch.no_grad():
            x = self.x
            outside = []
            for layer in self.model.layers:
                outside.append(float(((x < -1) | (x > 1)).float().mean()))
                x = layer(x)
        return {'target_mean': self.mean, 'target_std': self.std,
            'vocabulary': self.preprocessor.vocabulary_.tolist(),
            'quantiles': self.preprocessor.quantiles_.quantiles_.tolist(),
            'quantile_references': self.preprocessor.quantiles_.references_.tolist(),
            'transform_digest': digest(self.preprocessor.quantiles_.quantiles_.tolist()),
            'parameter_count': sum(p.numel() for p in self.model.parameters()),
            'max_gradient_norm': self.max_gradient_norm,
            'spline_norms': [float(l.spline_weight.detach().norm()) for l in self.model.layers],
            'hidden_outside_grid_fraction': outside,
            'tree_routing_diagnostics': 'not_applicable_to_KAN'}


class KANRegressor:
    def __init__(self, arm, settings=None):
        self.arm, self.settings = arm, dict(settings or {})

    def fit(self, frame, y, event):
        import resource
        import time
        from .v3_4_bags import group_safe_inner_folds
        from .v7_periodic import digest
        started = time.monotonic()
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all():
            raise ValueError('Invalid training targets')
        split = group_safe_inner_folds(frame, seed=42)
        mask = split['fold'] != 0
        inner = frame.loc[mask].reset_index(drop=True)
        session = TrainingSession(inner, y[mask], self.arm, event, 'inner')
        best, selected, stale = float('inf'), 0, 0
        trace = []
        for epoch in range(1, self.settings.get('max_epochs', 240) + 1):
            loss = session.epoch()
            value = float(np.mean(np.abs(session.predict(frame.loc[~mask]) - y[~mask])) / session.std)
            trace.append({'epoch': epoch, 'loss': loss, 'validation_scaled_MAE': value})
            if value < best - 1e-5:
                best, selected, stale = value, epoch, 0
            else:
                stale += 1
            if stale >= 25:
                break
        if selected < 1:
            raise ValueError('No finite selected epoch')
        inner_snapshot = session.snapshot()
        session = TrainingSession(frame, y, self.arm, event, 'outer')
        outer_trace = [{'epoch': epoch, 'loss': session.epoch()} for epoch in range(1, selected + 1)]
        self.session_ = session
        self.mean_, self.std_ = session.mean, session.std
        outer_snapshot = session.snapshot()
        self.metadata_ = {**outer_snapshot, 'selected_epoch': selected,
            'fit_rows': len(frame), 'inner_fit_rows': int(mask.sum()), 'optimizer_runs': 2,
            'inner': inner_snapshot, 'inner_trace': trace, 'outer_trace': outer_trace,
            'fit_ids_digest': digest(frame.sample_id.tolist()),
            'inner_ids_digest': digest(inner.sample_id.tolist()),
            'validation_ids_digest': digest(frame.loc[~mask].sample_id.tolist()),
            'group_hash': split['group_hash'], 'inner_fold_hash': split['inner_fold_hash'],
            'runtime_seconds': time.monotonic() - started,
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        return self

    def predict(self, frame, chunk_rows=256):
        return self.session_.predict(frame, chunk_rows)