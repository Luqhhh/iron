"""Saved-forest audit without fitting CART or reconstructing query labels."""
import numpy as np

from .data import FEATURES
from .v7_periodic import digest
from .v45_rotation import project, leaf_indices, tree_design


def verify_forest(model, frame, y):
    cfg = model.settings
    if len(model.members_) != cfg['trees'] or model.fit_ids_ != frame.sample_id.astype(str).tolist():
        raise ValueError('Forest count or fit identity mismatch')
    x = model.preprocessor_.transform(frame)
    for i, member in enumerate(model.members_):
        bag, groups, samples, seed = tree_design(len(x), cfg, i)
        if (member['index'] != i or member['tree_seed'] != seed
                or member['bag_digest'] != digest(bag.tolist())):
            raise ValueError('Bootstrap or tree seed mismatch')
        blocks = member['blocks']
        if len(blocks) != (len(groups) if model.recipe == 'ROTATE' else 0):
            raise ValueError('Wrong rotation block count')
        for block, indices, sample in zip(blocks, groups, samples):
            if block['indices'] != indices.tolist() or block['sample_digest'] != digest(sample.tolist()):
                raise ValueError('PCA fit subset/group mismatch')
            values = x[np.ix_(sample, indices)]
            center = np.asarray(block['center'])
            if not np.array_equal(center, values.mean(axis=0)):
                raise ValueError('PCA center not training-only')
            axes, singular = np.asarray(block['axes']), np.asarray(block['singular'])
            if axes.shape != (len(indices), len(indices)) or not np.isfinite(axes).all():
                raise ValueError('Incomplete/nonfinite PCA basis')
            if not np.allclose(axes.T@axes, np.eye(len(indices)), atol=1e-12, rtol=0):
                raise ValueError('PCA basis is not orthogonal')
            gram = (values-center).T@(values-center)
            if not np.allclose(axes.T@gram@axes, np.diag(singular**2), atol=1e-8, rtol=1e-10):
                raise ValueError('Saved PCA axes do not diagonalize training covariance')
            if np.any(np.diff(singular)>1e-10) or np.any(singular<0):
                raise ValueError('PCA components not in singular-value order')
            for j in range(len(indices)):
                if axes[np.argmax(np.abs(axes[:, j])), j] < 0:
                    raise ValueError('PCA component sign differs')
        transformed = project(x, blocks)
        tree = member['tree']
        left, right = np.asarray(tree['left']), np.asarray(tree['right'])
        feature, threshold = np.asarray(tree['feature']), np.asarray(tree['threshold'])
        value, count = np.asarray(tree['value']), np.asarray(tree['count'])
        n = len(left)
        if any(len(v)!=n for v in (right, feature, threshold, value, count)):
            raise ValueError('Tree array shapes differ')
        if not np.isfinite(value).all() or not np.isfinite(threshold).all():
            raise ValueError('Nonfinite saved tree')
        pending, seen, traversal = [0], set(), []
        while pending:
            node = pending.pop()
            if node in seen or not 0<=node<n:
                raise ValueError('Tree cycle, duplicate child or out of range')
            seen.add(node);traversal.append(node)
            if left[node] == -1:
                if right[node] != -1 or feature[node] != -2:
                    raise ValueError('Invalid leaf state')
            else:
                if not 0<=feature[node]<x.shape[1] or left[node]<0 or right[node]<0:
                    raise ValueError('Invalid split state')
                pending.extend([int(left[node]), int(right[node])])
        if len(seen)!=n:
            raise ValueError('Orphan tree node')
        leaves = leaf_indices(transformed[bag], tree)
        actual_count = np.bincount(leaves, minlength=n)
        sums = np.bincount(leaves, weights=y[bag], minlength=n)
        leaf_mask = left == -1
        if np.any(actual_count[leaf_mask]<cfg['min_samples_leaf']):
            raise ValueError('Leaf support below frozen minimum')
        for node in reversed(traversal):
            if left[node] != -1:
                actual_count[node] = actual_count[left[node]]+actual_count[right[node]]
                sums[node] = sums[left[node]]+sums[right[node]]
        if not np.array_equal(actual_count, count) or count[0]!=len(bag):
            raise ValueError('Saved node counts do not match bootstrap training rows')
        if not np.allclose(sums/count, value, rtol=1e-12, atol=1e-9):
            raise ValueError('Saved means do not match training target means')
    return {'trees_verified': len(model.members_), 'refits': 0}
