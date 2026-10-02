"""Fresh-process saved-state and fixed-replacement audit; no new fits."""
from pathlib import Path
import math

import numpy as np
import torch

from .sam_ema import SAMEMARegressor
from .component_regularization_run import RECIPE, outputs
from .data import FEATURES, TARGETS
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest


def verify_sam_ema_saved(path, training, y, arm, settings, mechanisms, validation=None, expected_epoch=None):
    model=SAMEMARegressor.load(path);saved=model.saved;trace=saved["trace"]
    if saved["arm"]!=arm or saved["settings"]!=settings or saved["mechanisms"]!=mechanisms or saved["recipe"]!=RECIPE:
        raise ValueError("Saved recipe identity mismatch")
    if trace["fit_ids_digest"]!=digest(training.sample_id.tolist()) or trace["fit_rows"]!=len(training):
        raise ValueError("Saved fit row identity mismatch")
    x=training[list(FEATURES)].to_numpy(float)
    np.testing.assert_array_equal(saved["preprocessing"]["means"],x.mean(axis=0))
    np.testing.assert_array_equal(saved["preprocessing"]["stds"],x.std(axis=0))
    if saved["preprocessing"]!=NumericPreprocessor(structure="raw_tabm").fit(training).metadata():
        raise ValueError("Train-only preprocessing mismatch")
    np.testing.assert_array_equal(saved["mean"],np.asarray(y,float).mean(axis=0))
    np.testing.assert_array_equal(saved["std"],np.asarray(y,float).std(axis=0))
    if any(v.is_floating_point() and (v.dtype!=torch.float32 or not torch.isfinite(v).all()) for v in saved["state"].values()):
        raise ValueError("Invalid saved dtype/value")
    if arm != 'SAM_EMA': raise ValueError('Unexpected composed mechanism')
    history=trace["history"];batches=math.ceil(len(training)/settings["batch_size"])
    if [r["epoch"] for r in history]!=list(range(1,trace["stopped_epoch"]+1)):
        raise ValueError("Incomplete epoch trace")
    if not 1<=trace["selected_epoch"]<=trace["stopped_epoch"]<=settings["max_epochs"]:
        raise ValueError("Invalid epoch range")
    for row in history:
        if row["updates"]!=batches or row["gradient_evaluations"]!=batches*2:
            raise ValueError("Update/gradient count mismatch")
        if row['ema_updates'] != batches: raise ValueError('EMA was not updated once per restored AdamW step')
        if any(not np.isfinite(v) or v<0 for v in row.values()):raise ValueError("Invalid training trace")
    if trace["updates"]!=batches*len(history) or trace["gradient_evaluations"]!=sum(r["gradient_evaluations"] for r in history):
        raise ValueError("Total update mismatch")
    if trace['ema_updates'] != trace['updates']: raise ValueError('Total EMA update count differs')
    if validation is not None:
        best,selected,stale=float("inf"),0,0
        for row in history:
            if stale>=settings["patience"]:raise ValueError("Training beyond patience")
            if row["validation_mae"]<best-settings["min_delta"]:
                best,selected,stale=row["validation_mae"],row["epoch"],0
            else:stale+=1
        if selected!=trace["selected_epoch"]:raise ValueError("Incorrect selected epoch")
        if trace["stopped_epoch"]<settings["max_epochs"] and stale!=settings["patience"]:
            raise ValueError("Premature stopping")
        query=validation.drop(columns=[t for t in TARGETS if t in validation])
        # Raw-space recomputation differs slightly from the original float32
        # standardized arithmetic; use the actual standardized tensor instead.
        vx,vc=model._inputs(query)
        vy=torch.as_tensor((validation[outputs("tap_iron" if len(saved["mean"])==2 else "tap_time_len")].to_numpy()-model.mean_)/model.std_,dtype=torch.float32)
        with torch.no_grad():actual=float((model.model_(vx,vc).mean(1)-vy).abs().mean())
        if actual!=best:raise ValueError("Selected checkpoint does not reproduce validation metric")
    else:
        if trace["selected_epoch"]!=trace["stopped_epoch"] or (expected_epoch is not None and trace["selected_epoch"]!=expected_epoch):
            raise ValueError("Refit epoch differs from selection")
    return model

