"""Matched bagged trees with train-only, full-rank block PCA rotations."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import resource

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .data import FEATURES, TARGETS
from .v7_periodic import digest
from .v30_deep_kernel import select_weight
from .v42_splines import SplinePreprocessor


def stream(seed, tree_index, number):
    return np.random.default_rng(np.random.SeedSequence([seed, tree_index, number]))


def tree_design(n_rows, settings, tree_index):
    seed = settings['seed']
    bag = stream(seed, tree_index, 0).integers(n_rows, size=n_rows)
    order = stream(seed, tree_index, 1).permutation(len(FEATURES))
    size = settings['rotation_group_size']
    groups = [order[i:i+size] for i in range(0, len(order), size)]
    rng = stream(seed, tree_index, 2)
    samples = [bag[rng.integers(n_rows, size=int(np.ceil(settings['pca_sample_fraction']*n_rows)))]
               for _ in groups]
    tree_seed = int(stream(seed, tree_index, 3).integers(2**31-1))
    return bag, groups, samples, tree_seed


def pca_block(x, indices, sample):
    values = x[np.ix_(sample, indices)]
    center = values.mean(axis=0)
    _, singular, vt = np.linalg.svd(values-center, full_matrices=False)
    axes = vt.T.copy()
    for j in range(axes.shape[1]):
        pivot = int(np.argmax(np.abs(axes[:, j])))
        if axes[pivot, j] < 0:
            axes[:, j] *= -1
    return {'indices': indices.tolist(), 'center': center.tolist(),
            'axes': axes.tolist(), 'singular': singular.tolist(),
            'sample_digest': digest(sample.tolist())}


def project(x, blocks):
    """Fixed arithmetic per row avoids BLAS batch-size dependent splits."""
    result = x.copy()
    for block in blocks:
        indices = block['indices']
        centered = x[:, indices]-np.asarray(block['center'])
        axes = np.asarray(block['axes'])
        for j, output in enumerate(indices):
            column = np.zeros(len(x))
            for k in range(len(indices)):
                column += centered[:, k]*axes[k, j]
            result[:, output] = column
    return result.astype(np.float32)


def export_tree(estimator):
    tree = estimator.tree_
    return {'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(),
            'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(),
            'value': tree.value[:, 0, 0].tolist(), 'count': tree.n_node_samples.tolist()}


def leaf_indices(x, tree):
    nodes = np.zeros(len(x), dtype=np.int64)
    left, right = np.asarray(tree['left']), np.asarray(tree['right'])
    feature, threshold = np.asarray(tree['feature']), np.asarray(tree['threshold'])
    for _ in range(len(left)):
        rows = np.flatnonzero(left[nodes] >= 0)
        if not len(rows):
            return nodes
        old = nodes[rows]
        nodes[rows] = np.where(x[rows, feature[old]] <= threshold[old], left[old], right[old])
    raise ValueError('Cyclic or invalid tree')


def predict_tree(x, tree):
    return np.asarray(tree['value'])[leaf_indices(x, tree)]


class RotationRegressor:
    def __init__(self, recipe, settings):
        if recipe not in ('AXIS', 'ROTATE'):
            raise ValueError('Unknown forest arm')
        self.recipe, self.settings = recipe, deepcopy(settings)

    def fit(self, frame, y):
        y = np.asarray(y, float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or len(frame) < 10:
            raise ValueError('Invalid training labels or insufficient rows')
        if any(t in frame for t in TARGETS):
            frame = frame.drop(columns=[t for t in TARGETS if t in frame])
        self.preprocessor_ = SplinePreprocessor().fit(frame)
        self.fit_ids_ = frame.sample_id.astype(str).tolist()
        x = self.preprocessor_.transform(frame)
        self.members_, self.maximum_export_difference_ = [], 0.
        cfg = self.settings
        for i in range(cfg['trees']):
            bag, groups, samples, seed = tree_design(len(x), cfg, i)
            blocks = ([pca_block(x, g, s) for g, s in zip(groups, samples)]
                      if self.recipe == 'ROTATE' else [])
            transformed = project(x, blocks)
            estimator = DecisionTreeRegressor(
                criterion=cfg['criterion'], splitter=cfg['splitter'],
                min_samples_leaf=cfg['min_samples_leaf'], min_samples_split=cfg['min_samples_split'],
                max_depth=cfg['max_depth'], max_features=cfg['max_features'],
                ccp_alpha=cfg['ccp_alpha'], random_state=seed)
            estimator.fit(transformed[bag], y[bag])
            tree = export_tree(estimator)
            diff = float(np.max(np.abs(predict_tree(transformed, tree)-estimator.predict(transformed))))
            if diff != 0:
                raise ValueError('Exported tree does not exactly match sklearn')
            self.maximum_export_difference_ = max(self.maximum_export_difference_, diff)
            self.members_.append({'index': i, 'tree_seed': seed, 'bag_digest': digest(bag.tolist()),
                                  'blocks': blocks, 'tree': tree})
        return self

    def predict(self, frame):
        if any(t in frame for t in TARGETS):
            raise ValueError('Query targets must be removed')
        x = self.preprocessor_.transform(frame)
        result = np.zeros(len(x))
        for member in self.members_:
            result += predict_tree(project(x, member['blocks']), member['tree'])
        return result/len(self.members_)

    def metadata(self):
        return {'recipe': self.recipe, 'fit_ids_digest': digest(self.fit_ids_), 'fit_rows': len(self.fit_ids_),
                'preprocessing': self.preprocessor_.metadata(), 'trees': len(self.members_),
                'maximum_export_difference': self.maximum_export_difference_,
                'total_nodes': sum(len(m['tree']['left']) for m in self.members_),
                'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}

    def save(self, path):
        with Path(path).open('x') as f:
            json.dump({'recipe': self.recipe, 'settings': self.settings,
                       'preprocessing': self.preprocessor_.metadata(), 'fit_ids': self.fit_ids_,
                       'maximum_export_difference': self.maximum_export_difference_,
                       'members': self.members_}, f, allow_nan=False)

    @classmethod
    def load(cls, path):
        saved = json.loads(Path(path).read_text())
        obj = cls(saved['recipe'], saved['settings'])
        obj.preprocessor_ = SplinePreprocessor.restore(saved['preprocessing'])
        obj.members_, obj.fit_ids_ = saved['members'], saved['fit_ids']
        obj.maximum_export_difference_ = saved['maximum_export_difference']
        return obj


def fit_partition(fitting, calibration, outer_training, query, target, recipe, settings, calibration_base, grid):
    if (set(fitting.sample_id)&set(calibration.sample_id)
            or set(outer_training.sample_id)!=set(fitting.sample_id)|set(calibration.sample_id)
            or set(outer_training.sample_id)&set(query.sample_id)):
        raise ValueError('Invalid nested partitions')
    if any(t in query for t in TARGETS):
        raise ValueError('Query labels must be removed')
    selector = RotationRegressor(recipe, settings).fit(fitting, fitting[target].to_numpy())
    cp = selector.predict(calibration.drop(columns=list(TARGETS)))
    weight, losses = select_weight(calibration[target], calibration_base, cp, grid)
    final = RotationRegressor(recipe, settings).fit(outer_training, outer_training[target].to_numpy())
    final.calibration_model_ = selector
    return final, final.predict(query), {'weight': weight, 'calibration_mae_by_weight': losses,
                                        'calibration': selector.metadata(), 'refit': final.metadata()}, cp
