"""Replay a frozen fold without access to train_samples or tap_history_train."""
import json
from pathlib import Path

import pandas as pd

from ...artifacts import (file_identities, stable_digest, validate_inference_source_contract,
                         verify_file_identities)
from ...data import normalize_event_source
from ...exceptions import ContractError
from ...features import build_features
from ...io import read_csv
from ..baseline import DualTargetBaseline
from .derive import replay_selection
from .folds import prediction_frame
from .m2_recency.model import RecencyModel
from .m3_residual.model import ResidualModel


def predict_fold(fold_directory, samples, paths, selection_path):
    root = Path(fold_directory)
    frozen = json.loads(Path(selection_path).read_text())
    if stable_digest({k: v for k, v in frozen.items() if k != "selection_sha256"}) != frozen["selection_sha256"]:
        raise ContractError("frozen selection digest mismatch")
    baseline_root = root / "CatBoost/bundle"
    baseline = DualTargetBaseline.load(baseline_root)
    metadata = json.loads((baseline_root / "bundle.json").read_text())
    cfg, semantic = metadata["feature_config"], metadata["semantic_contract"]
    identities = file_identities({name: paths[name] for name in ("operation_hourly", "burden_change")})
    validate_inference_source_contract(
        metadata["inference_source_contract"], identities,
        semantic_contract_sha256=metadata["contract_digests"]["semantic_contract_sha256"])
    cutoff = pd.Timestamp(metadata["training"]["fit_cutoff"])
    if samples["reference_time"].isna().any() or (samples["reference_time"] < cutoff).any():
        raise ContractError("offline samples must follow the fold cutoff")
    history = read_csv(baseline_root / "history_snapshot.csv",
                       time_columns=["reference_time", "tap_end_time", "available_at"])
    sources = {}
    for name, filekey in (("operation", "operation_hourly"), ("burden", "burden_change")):
        mapping = semantic["sources"][name]
        sources[name] = normalize_event_source(
            read_csv(paths[filekey]), event_time_column=mapping["event_time_column"],
            available_at_column=mapping["available_at_column"],
            value_columns=cfg[name]["value_columns"], missing_markers=mapping["missing_markers"])
    X = build_features(samples, history=history, fit_cutoff=cutoff, config=cfg, **sources).X
    predictions = {"CatBoost": prediction_frame(samples, baseline.predict(X))}
    controls = json.loads((root / "controls.json").read_text())
    if stable_digest({k: v for k, v in controls.items() if k != "sha256"}) != controls["sha256"]:
        raise ContractError("controls digest mismatch")
    for name in ("B0", "B1"):
        values = {}
        for target in ("tap_iron", "tap_time_len"):
            values["pred_" + target] = [
                (controls["by_spout"].get(str(spout), controls["global"]) if name == "B1"
                 else controls["global"])[target] for spout in samples["spout_no"]]
        predictions[name] = prediction_frame(samples, pd.DataFrame(values))
    for name, kind in ((f"M2_H{frozen['m2_half_life_days']}", RecencyModel),
                       ("M3_RESIDUAL", ResidualModel)):
        model = kind.load(root / name / "bundle")
        if (model.metadata_["feature_config"] != cfg
                or model.metadata_["semantic_contract"] != semantic
                or model.metadata_["training"] != metadata["training"]):
            raise ContractError("candidate and baseline fold contracts differ")
        predictions[name] = prediction_frame(samples, model.predict(samples, X))
    predictions.update(replay_selection(predictions, frozen))
    verify_file_identities(identities)
    return predictions
