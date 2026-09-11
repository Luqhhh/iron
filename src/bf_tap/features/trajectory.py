"""OPT-24's fixed, causal operation trajectory profile; no old builder changes."""
import numpy as np
import pandas as pd

from ..exceptions import ContractError
from ..schema import validate_event_source

VALUES = ('air_volume', 'total_press_diff', 'hot_air_press', 'oxygen')
WINDOWS = (6, 24)
STATS = ('slope', 'mean_abs_step_rate', 'valid_pair_count')
COLUMNS = [f'trajectory__{v}__{w}h__{s}' for v in VALUES for w in WINDOWS for s in STATS]


def build_trajectory_features(samples, events, *, contract_value_columns):
    """Preserve missing rows when forming adjacent pairs; audit without targets."""
    relevant = ['event_time', 'available_at', *contract_value_columns]
    validate_event_source(events, event_time='event_time', available_at='available_at',
                          value_columns=list(contract_value_columns))
    if not set(VALUES).issubset(contract_value_columns):
        raise ContractError('trajectory variables missing from operation contract')
    duplicate = events.duplicated('event_time', keep=False)
    if duplicate.any():
        if any(len(g.drop_duplicates()) > 1 for _, g in events.loc[duplicate, relevant].groupby('event_time', dropna=False)):
            raise ContractError('conflicting rows share the same event_time')
        events = events.drop_duplicates(relevant)
    events = events.sort_values(['event_time', 'available_at'], kind='mergesort')
    rows, audits = [], []
    for sample in samples.itertuples(index=False):
        ref = sample.reference_time
        visible = events.loc[(events.event_time <= ref) & (events.available_at <= ref)]
        row, audit = {}, {'sample_id': sample.sample_id}
        for window in WINDOWS:
            part = visible.loc[visible.event_time > ref - pd.Timedelta(hours=window)]
            times = (part.event_time - ref).dt.total_seconds().to_numpy() / 3600.
            dt = np.diff(times)
            audit[f'{window}h_observations'] = len(part)
            audit[f'{window}h_span_hours'] = float(times[-1]-times[0]) if len(times) else 0.
            for value in VALUES:
                x = part[value].to_numpy(dtype=float)
                finite = np.isfinite(x)
                t, y = times[finite], x[finite]
                slope = np.nan
                if len(np.unique(t)) >= 3:
                    centered = t-t.mean()
                    denom = np.dot(centered, centered)
                    if denom > 0:
                        slope = float(np.dot(centered, y-y.mean()) / denom)
                pairs = finite[:-1] & finite[1:] & (dt > 0) & (dt <= 1.5)
                count = int(pairs.sum())
                rate = float(np.mean(np.abs(np.diff(x)[pairs])/dt[pairs])) if count >= 2 else np.nan
                stem = f'trajectory__{value}__{window}h'
                row.update({stem+'__slope': slope, stem+'__mean_abs_step_rate': rate,
                            stem+'__valid_pair_count': float(count)})
        rows.append(row); audits.append(audit)
    result = pd.DataFrame(rows, index=samples.index, columns=COLUMNS, dtype=float)
    if np.isinf(result.to_numpy()).any():
        raise ContractError('trajectory overflow')
    return result, pd.DataFrame(audits, index=samples.index)


def append_trajectory(base, extra):
    if list(extra) != COLUMNS or not extra.index.equals(base.index) or set(base) & set(COLUMNS):
        raise ContractError('trajectory schema/index collision')
    result = pd.concat([base, extra], axis=1)
    if not result[list(base)].equals(base):
        raise ContractError('old features changed')
    return result
