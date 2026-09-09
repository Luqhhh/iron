"""OPT-19 inference policy. Accepts metadata, immutable base history and models only."""
import numpy as np
import pandas as pd
from ..artifacts import stable_digest
from ..exceptions import ContractError
from ..features.history import build_history_features
from ..schema import validate_history
from .component_export import META, PRED
from .history_stable import stable_features
from .snapshot_ensemble import blend

ROW = ['sample_id', 'tap_no', 'spout_no', 'reference_time', 'tap_iron',
       'tap_time_len', 'tap_end_time', 'available_at', 'depth']


def rollout(samples, entries, loaded, builder):
    if set(samples) != set(META + ['tap_no']) or samples.empty:
        raise ContractError('policy requires metadata only')
    if samples.isna().any().any() or samples.sample_id.duplicated().any() or samples.tap_no.duplicated().any():
        raise ContractError('incomplete or duplicate metadata')
    if set(entries) != {'OR', 'HR'} or set(loaded) != set(entries):
        raise ContractError('exact R2 components required')
    samples = samples.sort_values(['reference_time', 'sample_id'], kind='mergesort').reset_index(drop=True)
    frozen, parts = {}, {}
    for role, (model, history) in loaded.items():
        validate_history(history)
        cutoff = pd.Timestamp(entries[role]['cutoff'])
        if (history.available_at > cutoff).any() or (history.reference_time >= cutoff).any() or (samples.reference_time < cutoff).any():
            raise ContractError('invalid base boundary')
        if set(history.sample_id) & set(samples.sample_id) or set(history.tap_no) & set(samples.tap_no):
            raise ContractError('base/evaluation overlap')
        frozen[role] = builder.X(samples[META], entries[role], history)
        p = model.predict_raw(frozen[role]).clip(lower=0)
        p.insert(0, 'sample_id', samples.sample_id.to_numpy()); parts[role] = p
    u0 = blend(parts['OR'], parts['HR'], .8).set_index('sample_id')
    pending, audit, outputs = [], [], []
    for reference, group in samples.groupby('reference_time', sort=True):
        visible_rows = [r for r in pending if r['available_at'] <= reference]
        if any(r['reference_time'] >= reference for r in visible_rows):
            raise ContractError('pseudo origin must be strictly earlier')
        visible = pd.DataFrame(visible_rows, columns=ROW)
        digest = stable_digest(visible.astype(str).to_dict('records'))
        depth = 1 + max((r['depth'] for r in visible_rows), default=0)
        current = {}
        for role, (model, base) in loaded.items():
            X = frozen[role].loc[group.index].copy()
            if len(visible):
                history = pd.concat([base, visible.drop(columns='depth')], ignore_index=True)
                dynamic, _ = build_history_features(group[META], history, fit_cutoff=reference,
                    last_k=tuple(builder.a['features']['history']['last_k_mean']))
                for c in dynamic:
                    if c in X: X[c] = dynamic[c]
                X = stable_features(X, group[META], history, reference, 'R2')[list(X)]
            p = model.predict_raw(X).clip(lower=0)
            p.insert(0, 'sample_id', group.sample_id.to_numpy()); current[role] = p
        pred = blend(current['OR'], current['HR'], .8).set_index('sample_id')
        if not pending and not np.array_equal(pred[PRED].to_numpy(), u0.loc[pred.index, PRED].to_numpy()):
            raise ContractError('first timestamp must equal frozen R2 exactly')
        additions = []
        for s in group.itertuples():
            values = pred.loc[s.sample_id, PRED].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ContractError('invalid recursive output')
            completion = reference + pd.to_timedelta(values[1], unit='min')
            if pd.isna(completion) or completion < reference: raise ContractError('invalid completion')
            row = dict(sample_id=s.sample_id, tap_no=s.tap_no, spout_no=s.spout_no,
                reference_time=reference, tap_iron=values[0], tap_time_len=values[1],
                tap_end_time=completion, available_at=completion, depth=depth)
            additions.append(row)
            audit.append(dict(sample_id=s.sample_id, sample_reference_time=str(reference),
                base_bundle_sha256={r:e['bundle_sha256'] for r,e in entries.items()},
                base_history_sha256={r:e['history_snapshot_sha256'] for r,e in entries.items()},
                pseudo_history_rows_visible=len(visible), pseudo_history_sha256=digest,
                pseudo_available_at_max=str(visible.available_at.max()) if len(visible) else None,
                pseudo_history_depth=depth-1, predicted_completion_overlap=sum(r['available_at']>reference for r in pending),
                same_spout_completion_overlap=sum(r['available_at']>reference and str(r['spout_no'])==str(s.spout_no) for r in pending),
                prediction_before_update=values.tolist(), prediction_after_history_state=values.tolist(),
                prediction_frozen_U0=u0.loc[s.sample_id, PRED].tolist(),
                prediction_delta=(values-u0.loc[s.sample_id, PRED].to_numpy()).tolist()))
        # Equal timestamp samples are all predicted before any completion is registered.
        pending.extend(additions); outputs.append(pred.reset_index())
    return u0.reset_index(), pd.concat(outputs, ignore_index=True), audit, pd.DataFrame(pending)
