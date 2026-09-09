"""Require temporal certificates for BOTH the reused and newly trained predictors."""
from ..exceptions import ContractError
from .dual_ratio import fit_time


def fit_certified_time(oof,cutoff,minimum=100):
    fields=['inverse_fit_cutoff','inverse_train_reference_max','inverse_train_available_max','inverse_history_available_max']
    if set(fields)-set(oof) or oof[fields].isna().any().any():raise ContractError('missing inverse OOF certificate')
    if ((oof.inverse_fit_cutoff!=oof.fold_cutoff) |
        (oof.inverse_fit_cutoff>oof.reference_time) |
        (oof.inverse_train_reference_max>=oof.inverse_fit_cutoff) |
        (oof.inverse_train_available_max>oof.inverse_fit_cutoff) |
        (oof.inverse_history_available_max>oof.inverse_fit_cutoff)).any():raise ContractError('inverse OOF temporal leakage')
    return fit_time(oof,cutoff,minimum)
