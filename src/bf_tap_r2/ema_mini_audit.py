"""Independent partition/trajectory verification for saved native mini EMA states."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import math

from .ema_mini_model import MiniEMARegressor
from .component_regularization import ComponentRegressor
from .component_regularization_run import RECIPE, outputs
from .data import FEATURES, TARGETS
from .v3_4_bags import group_safe_inner_folds
from .v3_6_networks import NumericPreprocessor
from .v7_periodic import digest
from .ema_nested_residual import memory, forbid_training

def verify_saved(path, training, y, arm, settings, mechanisms, validation=None, expected_epoch=None):
    model=MiniEMARegressor.load(path);saved=model.saved;trace=saved["trace"]
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
    history=trace["history"];batches=math.ceil(len(training)/settings["batch_size"])
    if [r["epoch"] for r in history]!=list(range(1,trace["stopped_epoch"]+1)):
        raise ValueError("Incomplete epoch trace")
    if not 1<=trace["selected_epoch"]<=trace["stopped_epoch"]<=settings["max_epochs"]:
        raise ValueError("Invalid epoch range")
    for row in history:
        if row["updates"]!=batches or row["gradient_evaluations"]!=batches*(2 if arm=="SAM" else 1):
            raise ValueError("Update/gradient count mismatch")
        if any(not np.isfinite(v) or v<0 for v in row.values()):raise ValueError("Invalid training trace")
    if trace["updates"]!=batches*len(history) or trace["gradient_evaluations"]!=sum(r["gradient_evaluations"] for r in history):
        raise ValueError("Total update mismatch")
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


def audit_models(source,fitting,query,expected,settings,mechanisms):
    inner = group_safe_inner_folds(fitting,seed=settings['inner_seed'])['fold']
    f = fitting.loc[inner != 0].reset_index(drop=True);v = fitting.loc[inner == 0]
    with forbid_training(), patch.object(MiniEMARegressor, "_initialize", side_effect=RuntimeError("Cold inference cannot initialize training")):
        selector = verify_saved(source/'selection.pt',f,f[['tap_time_len']].to_numpy(),'EMA',settings,mechanisms,v)
        model = verify_saved(source/'refit.pt',fitting,fitting[['tap_time_len']].to_numpy(),'EMA',settings,mechanisms,
                             expected_epoch=selector.saved['trace']['selected_epoch'])
        maximum=0.
        for role,predictor in [('selection',selector),('refit',model)]:
            if role not in expected:
                continue
            p = expected[role]
            np.testing.assert_array_equal(predictor.predict(query)[:,0],p)
            for result in [predictor.predict(query.iloc[::-1])[::-1,0],
                    np.concatenate([predictor.predict(query.iloc[i:i+37])[:,0] for i in range(0,len(query),37)])]:
                maximum=max(maximum,float(np.max(np.abs(result-p))))
        if maximum > .0005:
            raise ValueError('Cold chunk/order gate failed')
    return dict(status='passed',maximum_difference=maximum,states=2,
                selected_epoch=selector.saved['trace']['selected_epoch'],peak_rss_mib=memory())

