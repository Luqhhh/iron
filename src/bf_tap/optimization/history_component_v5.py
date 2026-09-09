"""Versioned T1/T2 increments applied after the unchanged R2 feature builder."""
from __future__ import annotations
import pandas as pd
from ..exceptions import ContractError
from ..features.history import build_history_features

MAIN = 'E09_PROCESS_CHANGE_E02'
AUX = 'E04'
AGES = ['history__all__latest_available_age_minutes',
        'history__spout__latest_available_age_minutes']


def transform(frame, samples, history, cutoff, variant, component):
    if variant == 'R2':
        return frame
    if variant == 'T2':
        if component != AUX or not set(AGES).issubset(frame):
            raise ContractError('T2 requires the R2 auxiliary and both age columns')
        return frame.drop(columns=AGES)
    if variant != 'T1' or component != MAIN or any(c in frame for c in AGES):
        raise ContractError('T1 requires the age-free R2 main component')
    raw, _ = build_history_features(samples,history,fit_cutoff=cutoff)
    result=frame.copy()
    for group in ('all','spout'):
        for target in ('tap_iron','tap_time_len'):
            stem=f'history__{group}__{target}'
            m30,m100=f'{stem}__last30_median',f'{stem}__last100_median'
            if not {m30,m100}.issubset(frame):
                raise ContractError('T1 requires the R2 medians')
            result[f'{stem}__mean10_minus_median30']=raw[f'{stem}__last10_mean']-frame[m30]
            result[f'{stem}__median30_minus_median100']=frame[m30]-frame[m100]
    return result
