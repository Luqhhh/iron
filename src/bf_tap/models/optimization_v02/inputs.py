"""Protected-development inputs shared by every local optimization candidate."""
from dataclasses import dataclass

import pandas as pd

from ...config import (load_yaml, validate_baseline_config, validate_data_paths,
                       validate_feature_config, validate_frozen_contracts,
                       validate_semantic_contract)
from ...data import normalize_event_source
from ...exceptions import ContractError
from ...io import read_csv, read_development_history, read_development_labels
from ...protection import load_protection_policy
from ...schema import (validate_samples, validate_history, validate_cross_table_metadata,
                       validate_cross_table_consistency)
from .config import validate_optimization_common_config


@dataclass
class Inputs:
    labels: pd.DataFrame
    history: pd.DataFrame
    operation: pd.DataFrame
    burden: pd.DataFrame
    baseline: dict
    features: dict
    semantic: dict
    common: dict
    data: dict
    protection: object
    contract_digests: dict
    consistency: dict


def load_inputs(data_config, config_root, requested_end):
    data = load_yaml(data_config)
    validate_data_paths(data, command="development")
    baseline = load_yaml(config_root / "baseline.yaml")
    features = load_yaml(config_root / "features.yaml")
    semantic = load_yaml(config_root / "data_contract.yaml")
    common = load_yaml(config_root / "optimization/v0.2/common.yaml")
    protection = load_protection_policy(config_root / "protection.yaml")
    validate_baseline_config(baseline)
    validate_feature_config(features)
    validate_semantic_contract(semantic)
    digests = validate_frozen_contracts(baseline, features, semantic)
    validate_optimization_common_config(common)
    if pd.Timestamp(common["development_label_end_exclusive"]) != protection.development_label_end_exclusive:
        raise ContractError("local optimization and protected label boundaries differ")
    protection.validate_development_read(requested_end)
    paths = data["paths"]
    mapping = semantic["sources"]["history"]
    end_column = semantic["targets"]["available_at_column"]
    metadata_columns = ["sample_id", "tap_no", "spout_no", "reference_time"]
    train_meta = read_csv(paths["train_samples"], usecols=metadata_columns,
                          time_columns=["reference_time"])
    history_meta = read_csv(paths["tap_history_train"],
                            usecols=metadata_columns + [end_column],
                            time_columns=["reference_time", end_column])
    history_meta["available_at"] = history_meta[end_column]
    consistency = {"metadata": validate_cross_table_metadata(train_meta, history_meta)}
    labels = read_development_labels(paths["train_samples"], requested_end=requested_end,
                                     protection_policy=protection)
    history = read_development_history(
        paths["tap_history_train"], requested_end=requested_end,
        protection_policy=protection, available_at_column=mapping["available_at_column"])
    history["available_at"] = history[mapping["available_at_column"]]
    validate_samples(labels, labeled=True)
    validate_history(history, end_time=mapping["end_time_column"], available_at="available_at")
    consistency["targets"] = validate_cross_table_consistency(labels, history)
    availability = history_meta[["sample_id", end_column]].rename(
        columns={end_column: "label_available_at"})
    labels = labels.merge(availability, on="sample_id", validate="one_to_one", how="left")
    if labels["label_available_at"].isna().any():
        raise ContractError("missing label availability")
    sources = {}
    for name, file_key in (("operation", "operation_hourly"), ("burden", "burden_change")):
        mapping = semantic["sources"][name]
        sources[name] = normalize_event_source(
            read_csv(paths[file_key]), event_time_column=mapping["event_time_column"],
            available_at_column=mapping["available_at_column"],
            value_columns=features[name]["value_columns"],
            missing_markers=mapping["missing_markers"])
    return Inputs(labels, history, sources["operation"], sources["burden"], baseline,
                  features, semantic, common, data, protection, digests, consistency)
