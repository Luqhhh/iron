"""Fixed OPT-12 transformations; historical rows, never feature-cell perturbations."""
import numpy as np
import pandas as pd
from ..exceptions import ContractError


def stable_features(frame, samples, history, cutoff, variant):
    if variant == 'raw':
        return frame
    if variant not in ('R1', 'R2'):
        raise ContractError('unknown fixed history strategy')
    target_cols = [c for c in frame if c.startswith('history__') and
                   ('__tap_iron__' in c or '__tap_time_len__' in c)]
    if variant == 'R1':
        return frame.drop(columns=[c for c in target_cols if c.endswith('__latest') or c.endswith('__last3_mean')])
    result = frame.drop(columns=target_cols).copy()
    origin = history.loc[history.available_at <= cutoff].sort_values(
        ['available_at', 'reference_time', 'sample_id'], kind='mergesort')
    rows = []
    for s in samples.itertuples():
        visible = origin.loc[(origin.available_at <= s.reference_time) &
                             (origin.sample_id.astype(str) != str(s.sample_id))]
        row = {}
        for name, group in [('all', visible), ('spout', visible.loc[visible.spout_no.astype(str) == str(s.spout_no)])]:
            for target in ('tap_iron', 'tap_time_len'):
                for k in (30, 100):
                    values = group[target].tail(k).dropna()
                    prefix = f'history__{name}__{target}__last{k}'
                    row[prefix + '_median'] = float(values.median()) if len(values) else np.nan
                    row[prefix + '_count'] = float(len(values))
        rows.append(row)
    return pd.concat([result, pd.DataFrame(rows, index=samples.index)], axis=1)
