"""V30 matching frozen B0 refits; no query labels and no weight selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .data import TARGETS
from .v4_1_reference import V36FixedRecipeFactory
from .v5_replicate import load_candidate
from .v5_spec import load_v5_spec
from .v7_periodic import PeriodicRegressor, digest
from .v12_joint import JointRegressor


def fit_b0(root, training, query, spec):
    """Refit all 32 frozen pipelines; neural pipelines may fit internally twice."""
    if any(target in query for target in TARGETS):
        raise ValueError("B0 query contains labels")
    if set(training.sample_id) & set(query.sample_id):
        raise ValueError("B0 training/query overlap")
    root = Path(root)
    factory = V36FixedRecipeFactory(root, workers=spec["budget"]["reference_workers"])
    base_count = (len(factory.a_factory._recovery.BASE_NAMES)
                  + sum(map(len, factory.a_factory._expert_ids.values()))
                  + sum(map(len, factory.selected_experts.values())) + 3)
    if base_count != spec["budget"]["b0_pipeline_fit_calls_per_factory"]:
        raise ValueError("Frozen B0 component count changed")
    bundle = factory.fit_predict(training, query)
    n_fit, n_meta = load_candidate(root, load_v5_spec(root), "v36", spec["reference"]["n_trial"])
    n_prediction = n_fit(training, query, "tap_time_len")
    joint_spec = yaml.safe_load((root / spec["reference"]["joint_spec"]).read_text())
    joint = JointRegressor(joint_spec["recipes"][spec["reference"]["joint_recipe"]], joint_spec["training"])
    joint.fit(training, training[list(TARGETS)].to_numpy())
    joint_prediction = joint.predict(query)[:, 0]
    periodic_spec = yaml.safe_load((root / spec["reference"]["periodic_spec"]).read_text())
    periodic = PeriodicRegressor(periodic_spec["recipes"][spec["reference"]["periodic_recipe"]],
                                 periodic_spec["training"])
    periodic.fit(training, training.tap_time_len.to_numpy())
    periodic_prediction = periodic.predict(query)
    parts = {"v36_iron": bundle["b36"]["tap_iron"], "v36_time": bundle["b36"]["tap_time_len"],
             "v12_iron": joint_prediction, "n_time": n_prediction, "v7_time": periodic_prediction}
    iron = spec["reference"]["iron_weights"]
    time = spec["reference"]["time_weights"]
    result = {"tap_iron": iron["v36"] * parts["v36_iron"] + iron["v12_joint"] * parts["v12_iron"],
              "tap_time_len": time["v36"] * parts["v36_time"] + time["n0048"] * parts["n_time"]
              + time["v7_periodic"] * parts["v7_time"]}
    if any(v.shape != (len(query),) or not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Invalid B0 refit output")
    return {**result, **parts}, {"factory_calls": 1, "component_pipeline_fits": base_count,
        "training_rows": len(training), "query_rows": len(query),
        "fit_ids_digest": digest(training.sample_id.tolist()),
        "query_ids_digest": digest(query.sample_id.tolist()),
        "query_labels_received": False, "v36": bundle["meta"], "n": n_meta,
        "joint": joint.metadata_, "periodic": periodic.metadata_}
