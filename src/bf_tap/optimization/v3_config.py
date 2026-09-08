"""The v1 protocol is fixed; descriptive YAML must not silently alter execution."""
from pathlib import Path

import yaml

from ..exceptions import ContractError


def validate_registration(directory: str | Path = "configs/optimization_v0_3") -> dict:
    root = Path(directory)
    calibration = yaml.safe_load((root / "calibration.yaml").read_text())
    expected = {
        "schema_version": "calibration-protocol-v1",
        "selection_train_cutoff_days_before_outer": 56,
        "calibration_train_cutoff_days_before_outer": 28,
        "history_policy": "frozen_at_each_fit_cutoff",
        "selection_label_available_by": "calibration_train_cutoff",
        "calibration_label_available": "strictly_before_outer_cutoff",
        "minimum_samples": 100, "fallback": "zero", "residual": "median_prediction_minus_actual",
        "targets": ["tap_time_len"], "early_stopping_patience": 150,
        "frozen_component_iterations": 800,
    }
    if calibration != expected:
        raise ContractError("calibration YAML differs from executable protocol v1; register a new protocol")
    features = yaml.safe_load((root / "features.yaml").read_text())
    if (features.get("schema_version") != "optimization-features-v1"
            or features.get("base_candidate") != "E09"
            or features.get("base_selection") != "configs/optimization_v0_2/features.yaml"
            or set(features.get("experiments", {})) != {"F-A", "F-B", "F-C", "F-D"}
            or features["experiments"]["F-A"] != {"kind": "remove_burden", "status": "registered"}
            or features["experiments"]["F-B"] != {"kind": "segmented_operation", "windows_hours": [[6, 12], [12, 24], [24, 48]], "statistics": ["mean", "count"], "status": "registered"}
            or any(features["experiments"][f].get("status") != "BLOCKED" for f in ("F-C", "F-D"))):
        raise ContractError("feature YAML differs from executable v1 feature registration")
    experiment = yaml.safe_load((root / "experiment.yaml").read_text())
    if (experiment.get("schema_version") != "optimization-v0.3-plan-v1"
            or experiment.get("optimization_id") != "optimization-v0.3"
            or experiment.get("references") != ["E12-raw", "E12-CVcal"]
            or experiment.get("reference_selection") != "global_minimum_development_J_tie_raw"
            or experiment.get("search_feature_candidate") != "E09"
            or experiment.get("screening_metric") != "mean_two_fold_target_wmape"
            or experiment.get("promotion") != "one_configuration_per_model_type_per_target"
            or experiment.get("max_promoted_target_configurations") != 4
            or experiment.get("max_feature_experiments") != 4
            or experiment.get("max_feature_model_crosses") != 2
            or experiment.get("max_target_combinations") != 2
            or experiment.get("protected_access_authorized") is not False
            or experiment.get("platform_upload_authorized") is not False):
        raise ContractError("experiment YAML differs from executable development v1 registration")
    return {"experiment": experiment, "calibration": calibration, "features": features}
