"""Fit one temporal fold using a single frozen as-of feature builder."""
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ...artifacts import atomic_write_json, stable_digest
from ...availability import freeze_history_origin
from ...exceptions import ContractError
from ...features import build_features
from ...metrics import score_predictions
from ...splits import select_split
from ...train import fit_baseline
from ..baseline import DualTargetBaseline
from ..sanity import MedianControls
from .m2_recency.model import RecencyModel
from .m3_residual.model import ResidualModel


def prediction_frame(samples, values):
    return pd.DataFrame({"sample_id": samples["sample_id"].astype("string").to_numpy(),
                         **{c: values[c].to_numpy() for c in ("pred_tap_iron", "pred_tap_time_len")}})


def run_fold(inputs, fold, directory, metadata, half_lives, *, determinism=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    train, evaluation, split_audit = select_split(inputs.labels, fold)
    shared = dict(operation=inputs.operation, burden=inputs.burden,
                  history=inputs.history, fit_cutoff=fold.fit_cutoff, config=inputs.features)
    print(f"{fold.id}: features train={len(train)} eval={len(evaluation)}", flush=True)
    train_features = build_features(train, **shared)
    eval_features = build_features(evaluation, **shared)
    if list(train_features.X) != list(eval_features.X):
        raise ContractError("train/evaluation feature mismatch")
    for role, result in (("train", train_features), ("eval", eval_features)):
        for source, audit in result.audit.items():
            audit.to_csv(directory / f"{role}_{source}_asof_audit.csv", index=False)
    snapshot = freeze_history_origin(inputs.history, fold.fit_cutoff)
    snapshot.to_csv(directory / "history_snapshot.csv", index=False)
    train[["sample_id", "reference_time", "label_available_at"]].to_csv(
        directory / "training_membership.csv", index=False)
    evaluation[["sample_id", "reference_time", "label_available_at"]].to_csv(
        directory / "evaluation_membership.csv", index=False)
    base_metadata = {
        **metadata, "baseline_config": inputs.baseline, "feature_config": inputs.features,
        "semantic_contract": inputs.semantic, "contract_digests": inputs.contract_digests,
        "training": {
            "mode": "local-optimization-development", "fold_id": fold.id,
            "fit_cutoff": str(fold.fit_cutoff), "train_start": str(fold.train_start),
            "eligible_rows": len(train), "history_rows": len(snapshot),
            "eligible_sample_id_sha256": stable_digest(train["sample_id"].astype(str).tolist()),
            "history_origin_sha256": stable_digest(snapshot[["sample_id", "available_at"]].astype(str).to_dict("records")),
        },
    }
    controls = MedianControls(
        min_group_count=inputs.baseline["controls"]["per_spout"]["min_group_count"]).fit(train)
    control_state = {"global": controls.global_, "by_spout": controls.by_spout_}
    atomic_write_json(directory / "controls.json", {**control_state, "sha256": stable_digest(control_state)})
    predictions = {name: controls.predict(evaluation, name) for name in ("B0", "B1")}
    audit = {"split": split_audit, "models": {}, "feature_columns": list(train_features.X)}
    model_ids = ["CatBoost", *[f"M2_H{h}" for h in half_lives], "M3_RESIDUAL"]
    for name in model_ids:
        model_start = perf_counter()
        print(f"{fold.id}: fitting {name}", flush=True)
        bundle = directory / name / "bundle"
        if name == "CatBoost":
            model = fit_baseline(train, train_features.X, inputs.baseline["parameters"],
                                 categorical=tuple(inputs.baseline["categorical_features"]))
            raw = model.predict_raw(eval_features.X)
            model.save(bundle, metadata=base_metadata, history_snapshot=snapshot)
            restored = DualTargetBaseline.load(bundle)
            replay = restored.predict_raw(eval_features.X)
            info = {"fit_rows": len(train), "warmup_rows": 0}
            if determinism:
                again = fit_baseline(train, train_features.X, inputs.baseline["parameters"],
                                     categorical=tuple(inputs.baseline["categorical_features"]))
                info["determinism_max_abs_diff"] = float(np.max(np.abs(
                    raw.to_numpy() - again.predict_raw(eval_features.X).to_numpy())))
        else:
            model = (RecencyModel(int(name.split("H")[1])) if name.startswith("M2_H")
                     else ResidualModel())
            model.fit(train, train_features.X, fold.fit_cutoff)
            raw = model.predict_raw(evaluation, eval_features.X)
            model.save(bundle, metadata=base_metadata, history_snapshot=snapshot)
            restored = type(model).load(bundle)
            replay = restored.predict_raw(evaluation, eval_features.X)
            info = dict(model.state_)
            if determinism and name in {"M2_H30", "M3_RESIDUAL"}:
                again = type(model)(**model.options).fit(train, train_features.X, fold.fit_cutoff)
                info["determinism_max_abs_diff"] = float(np.max(np.abs(
                    raw.to_numpy() - again.predict_raw(evaluation, eval_features.X).to_numpy())))
            if name == "M3_RESIDUAL":
                model.anchor_audit_.to_csv(directory / name / "anchor_audit.csv", index=False)
        difference = float(np.max(np.abs(raw.to_numpy() - replay.to_numpy())))
        if difference > 1e-10 or info.get("determinism_max_abs_diff", 0.) > 1e-10:
            raise ContractError(f"{name}: deterministic replay failed")
        info.update(roundtrip_max_abs_diff=difference, fit_seconds=perf_counter() - model_start)
        audit["models"][name] = info
        prediction_frame(evaluation, raw).to_csv(directory / name / "predictions_raw.csv", index=False)
        predictions[name] = prediction_frame(evaluation, raw.clip(lower=0.0))
    audit["elapsed_seconds"] = perf_counter() - start
    metrics = {}
    for name, frame in predictions.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
        metrics[name] = score_predictions(evaluation, frame)
    atomic_write_json(directory / "metrics.json", metrics)
    atomic_write_json(directory / "audit.json", audit)
    actual = evaluation[["sample_id", "tap_iron", "tap_time_len", "reference_time", "spout_no"]].copy()
    actual.insert(0, "fold_id", fold.id)
    actual.to_csv(directory / "actual.csv", index=False)
    for frame in predictions.values():
        frame.insert(0, "fold_id", fold.id)
    return actual, predictions, audit
