"""Fixed local ridge slopes and a KNN control using identical query neighbors."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .data import FEATURES, TARGETS

K = 64
RIDGE = .1
MAX_CONDITION = 1e6
MAX_COEFFICIENT = 1e4


def features(frame):
    if (set(frame.columns) != {'sample_id', 'spout_no', *FEATURES}
            or len(frame.columns) != len(set(frame.columns)) or not len(frame)
            or frame.sample_id.isna().any() or frame.sample_id.astype(str).duplicated().any()
            or (frame.sample_id.astype(str).str.len() == 0).any()):
        raise ValueError('Unique IDs and exact unlabeled feature schema required')
    x = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    cat = frame.spout_no.to_numpy(dtype=np.float64)
    if (not np.isfinite(x).all() or not np.isfinite(cat).all()
            or not np.equal(cat, np.floor(cat)).all() or (np.abs(cat) > 2**53).any()):
        raise ValueError('Finite features and exact integer spouts required')
    return x, cat.astype(np.int64)


def local_solution(x, y, query):
    """Mean squared loss plus RIDGE*||beta||²; the intercept is unpenalized."""
    xm, ym = x.mean(0), y.mean(0)
    dx, dy = x-xm, y-ym
    gram = dx.T @ dx / len(x) + RIDGE*np.eye(x.shape[1])
    condition = float(np.linalg.cond(gram))
    factor = np.linalg.cholesky(gram)
    beta = np.linalg.solve(factor.T, np.linalg.solve(factor, dx.T @ dy / len(x)))
    prediction = ym + (query-xm) @ beta
    if (not np.isfinite(beta).all() or not np.isfinite(prediction).all()
            or not np.isfinite(condition) or condition > MAX_CONDITION
            or np.max(np.abs(beta)) > MAX_COEFFICIENT):
        raise ValueError('Frozen numerical gate failed; no fallback')
    return prediction, ym, beta, condition


class LocalRidgeNeighborhood:
    """Train-only z scores, one-hot spouts and standardized two-target slopes."""
    def fit(self, frame, y):
        x, cat = features(frame)
        y = np.asarray(y, dtype=np.float64)
        if len(x) < K or y.shape != (len(x), len(TARGETS)) or not np.isfinite(y).all():
            raise ValueError('At least64 training rows and finite two-target labels required')
        self.ids = frame.sample_id.to_numpy(str)
        self.mean, self.scale = x.mean(0), x.std(0)
        self.scale[self.scale == 0] = 1.
        self.categories = np.unique(cat)
        self.ymean, self.yscale = y.mean(0), y.std(0)
        self.yscale[self.yscale == 0] = 1.
        self.x = self.encode(frame)
        self.y = (y-self.ymean)/self.yscale
        return self

    def encode(self, frame):
        x, cat = features(frame)
        codes = {int(v): i+1 for i, v in enumerate(self.categories)}
        onehot = np.zeros((len(x), len(codes)+1))
        onehot[np.arange(len(x)), [codes.get(int(v), 0) for v in cat]] = 1.
        value = np.concatenate([(x-self.mean)/self.scale, onehot], axis=1)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite encoded features')
        return value

    def predict(self, frame):
        if set(frame.sample_id.astype(str)) & set(self.ids):
            raise ValueError('Query IDs overlap fitted rows')
        queries = self.encode(frame)
        linear, knn, neighbors, beta, conditions = [], [], [], [], []
        for q in queries:
            diff = self.x-q
            distance = np.sum(diff*diff, axis=1)
            if not np.isfinite(distance).all():
                raise ValueError('Nonfinite neighbor distances')
            indices = np.lexsort((self.ids, distance))[:K]
            a, b, slope, condition = local_solution(self.x[indices], self.y[indices], q)
            linear.append(a*self.yscale+self.ymean)
            knn.append(b*self.yscale+self.ymean)
            neighbors.append(indices); beta.append(slope); conditions.append(condition)
        result = dict(linear=np.array(linear), knn=np.array(knn), neighbors=np.array(neighbors),
                      beta=np.array(beta), condition=np.array(conditions))
        if not all(np.isfinite(a).all() for a in result.values()):
            raise ValueError('Nonfinite output')
        return result

    def save(self, path):
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, ids=self.ids, mean=self.mean, scale=self.scale,
                categories=self.categories, ymean=self.ymean, yscale=self.yscale,
                x=self.x, y=self.y, k=np.array(K), ridge=np.array(RIDGE))

    @classmethod
    def load(cls, path):
        obj = cls()
        names = {'ids', 'mean', 'scale', 'categories', 'ymean', 'yscale', 'x', 'y', 'k', 'ridge'}
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != names or saved['k'].item() != K or saved['ridge'].item() != RIDGE:
                raise ValueError('Saved recipe differs')
            for name in names-{'k', 'ridge'}:
                setattr(obj, name, saved[name].copy())
        n, d = len(obj.ids), len(FEATURES)+len(obj.categories)+1
        if (n < K or obj.ids.shape != (n,) or len(set(obj.ids)) != n
                or obj.mean.shape != (len(FEATURES),) or obj.scale.shape != obj.mean.shape
                or obj.ymean.shape != (2,) or obj.yscale.shape != (2,)
                or obj.x.shape != (n, d) or obj.y.shape != (n, 2)
                or obj.categories.ndim != 1 or not len(obj.categories)
                or not np.array_equal(obj.categories, np.unique(obj.categories))
                or not np.equal(obj.categories, np.floor(obj.categories)).all()
                or any(not np.isfinite(getattr(obj, key)).all() for key in names-{'ids', 'k', 'ridge'})
                or (obj.scale <= 0).any() or (obj.yscale <= 0).any()):
            raise ValueError('Invalid saved state')
        return obj


def independent_predict(model, frame):
    """Separate distance reduction and augmented least-squares solution, no fit."""
    queries = model.encode(frame)
    linear, knn, neighbors, beta = [], [], [], []
    for q in queries:
        delta = model.x-q
        distance = np.einsum('ij,ij->i', delta, delta)
        indices = np.lexsort((model.ids, distance))[:K]
        x, y = model.x[indices], model.y[indices]
        xm, ym = x.mean(0), y.mean(0)
        # Solve a rectangular augmented system using SVD, independently of Cholesky.
        design = np.vstack([(x-xm)/np.sqrt(K), np.sqrt(RIDGE)*np.eye(x.shape[1])])
        response = np.vstack([(y-ym)/np.sqrt(K), np.zeros((x.shape[1], 2))])
        slope = np.linalg.lstsq(design, response, rcond=None)[0]
        linear.append((ym+np.einsum('i,ij->j', q-xm, slope))*model.yscale+model.ymean)
        knn.append(ym*model.yscale+model.ymean)
        neighbors.append(indices); beta.append(slope)
    return dict(linear=np.array(linear), knn=np.array(knn), neighbors=np.array(neighbors), beta=np.array(beta))
