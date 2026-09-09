"""Frozen OPT-20 cross-target correction. Inference has no fitting or label access."""
import numpy as np
import pandas as pd
from ..exceptions import ContractError

PRED = ['pred_tap_iron', 'pred_tap_time_len']
INPUT = ['sample_id', *PRED, 'pred_rate']


def directions(predictions, rate_floor=1e-6):
    if list(predictions) != INPUT or predictions.empty or predictions.sample_id.isna().any() or predictions.sample_id.duplicated().any():
        raise ContractError('unique prediction-only structural inputs required')
    values = predictions[PRED+['pred_rate']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError('finite nonnegative structural inputs required')
    iron, time, rate = values.T
    usable = rate > rate_floor
    delta = np.zeros((len(values), 2))
    delta[usable, 0] = rate[usable]*time[usable]-iron[usable]
    delta[usable, 1] = iron[usable]/rate[usable]-time[usable]
    if not np.isfinite(delta).all():
        raise ContractError('nonfinite structural direction')
    return delta, usable


def lad_coefficient(actual, base, direction):
    """Exact constrained scalar LAD, choosing the smallest minimizer on ties."""
    actual, base, direction = [np.asarray(x, dtype=float) for x in (actual, base, direction)]
    if actual.ndim != 1 or actual.shape != base.shape or base.shape != direction.shape or not len(actual):
        raise ContractError('LAD requires aligned nonempty vectors')
    if not all(np.isfinite(x).all() for x in (actual, base, direction)):
        raise ContractError('nonfinite LAD input')
    nonzero = direction != 0
    if not nonzero.any(): return 0.0
    ratios = (actual[nonzero]-base[nonzero])/direction[nonzero]
    weights = np.abs(direction[nonzero])
    if not np.isfinite(ratios).all(): raise ContractError('nonfinite LAD ratio')
    order = np.argsort(ratios, kind='mergesort')
    position = np.searchsorted(np.cumsum(weights[order]), .5*weights.sum(), side='left')
    return float(np.clip(ratios[order[position]], 0., 1.))


def select_oof(oof, cutoff, minimum=100):
    required = set(INPUT+['reference_time','label_available_at','fold_cutoff',
        'train_reference_max','train_available_max','history_available_max','tap_iron','tap_time_len'])
    if required-set(oof) or oof.sample_id.duplicated().any():
        raise ContractError('incomplete or duplicate temporal OOF evidence')
    if any(oof[c].isna().any() for c in required): raise ContractError('null OOF evidence')
    if ((oof.train_reference_max >= oof.fold_cutoff) |
        (oof.train_available_max > oof.fold_cutoff) |
        (oof.history_available_max > oof.fold_cutoff) |
        (oof.fold_cutoff > oof.reference_time) |
        (oof.label_available_at < oof.reference_time)).any():
        raise ContractError('OOF temporal leakage')
    selected = oof.loc[(oof.reference_time < cutoff) & (oof.fold_cutoff < cutoff) &
                       (oof.label_available_at <= cutoff)].copy()
    if len(selected) < minimum: raise ContractError('insufficient causal OOF rows')
    return selected.sort_values(['reference_time','sample_id'],kind='mergesort')


def fit_correction(oof, cutoff, minimum=100, rate_floor=1e-6):
    selected = select_oof(oof, cutoff, minimum)
    delta, _ = directions(selected[INPUT], rate_floor)
    alpha = [lad_coefficient(selected[t], selected[p], delta[:,i])
             for i,(t,p) in enumerate(zip(('tap_iron','tap_time_len'),PRED))]
    return alpha, selected


def apply_correction(predictions, alpha, rate_floor=1e-6):
    coefficients = np.asarray(alpha,dtype=float)
    if coefficients.shape != (2,) or not np.isfinite(coefficients).all() or ((coefficients < 0)|(coefficients > 1)).any():
        raise ContractError('two fixed structural coefficients in [0,1] required')
    delta, _ = directions(predictions,rate_floor)
    values = np.maximum(0.,predictions[PRED].to_numpy()+delta*coefficients)
    if not np.isfinite(values).all(): raise ContractError('nonfinite corrected prediction')
    out = pd.DataFrame(values,columns=PRED,index=predictions.index)
    out.insert(0,'sample_id',predictions.sample_id.to_numpy())
    return out
