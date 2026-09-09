"""Identity-aligned fixed convex combinations, with no fit or calibration."""
import numpy as np
from ..exceptions import ContractError


def blend(left, right, weight=.5):
    parts = []
    for frame in (left, right):
        frame = frame.copy()
        if frame.sample_id.isna().any():
            raise ContractError('null ensemble ID')
        frame['sample_id'] = frame.sample_id.astype(str)
        if frame.sample_id.duplicated().any():
            raise ContractError('duplicate ensemble ID')
        parts.append(frame.set_index('sample_id').sort_index())
    a, b = parts
    if not a.index.equals(b.index) or list(a.columns) != list(b.columns):
        raise ContractError('ensemble identity mismatch')
    if weight not in (.5, .8):
        raise ContractError('unregistered weight')
    if not np.isfinite(a.to_numpy()).all() or not np.isfinite(b.to_numpy()).all():
        raise ContractError('nonfinite ensemble prediction')
    return (weight*a + (1-weight)*b).reset_index()
