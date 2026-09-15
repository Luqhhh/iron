"""Protected, deterministic reconstruction of the scored V10 test-A package.

The candidate is frozen before labels are opened.  It combines only the
registered V6I iron branch and V8 QRF-median time branch; test metadata is used
for prediction, never for fitting or selection.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ..artifacts import (
    atomic_write_json,
    build_inference_source_contract,
    file_identities,
    file_sha256,
    runtime_environment,
    stable_digest,
    verify_file_identities,
)
from ..config import load_yaml
from ..exceptions import ContractError, ProtectedLabelError
from ..io import parse_local_time
from ..models.baseline import DualTargetBaseline, FROZEN_PARAMETERS
from ..offline import _load_process_sources
from ..protection import load_protection_policy
from ..schema import validate_cross_table_consistency, validate_history, validate_samples
from ..submission import pack_submission, validate_submission, write_submission
from .component_export import ComponentFeatures, META
from .final_lifecycle import (
    _read_eligible_rows,
    algorithm as base_algorithm,
    eligibility_identity,
    eligible,
    label_metadata,
    sample_metadata,
)
from .lifecycle import PURPOSES, authorize_protected_operation
from .qrf_time_run import pack as pack_qrf_handoff
from .rate_model import RateModel, schema
from .recency_model import HALF_LIFE_DAYS, RecencyModel, weights
from .refresh_factorial import Context, stamp
from .structural import INPUT, lad_coefficient, select_oof


REPO = Path(__file__).resolve().parents[3]
IDENTITY = "optimization-v0.17/V10_V6I_IRON_V8_TIME/reconstruction-v1"
COMPONENTS = ("E09_PROCESS_CHANGE_E02", "E04")
TARGETS = ("tap_iron", "tap_time_len")
PRED = ("pred_tap_iron", "pred_tap_time_len")
FIT_MONTHS = tuple(range(4, 12))
EXPECTED_TRAIN_ROWS = 2754
EXPECTED_TEST_ROWS = 335
EXPECTED_RESULT_SHA256 = "d430da7399c3125db1768b1d7d236ec6eb95e13c633e5389b138dfe39c97b3c6"
ORIGINAL_ZIP_SHA256 = "be83f1f623c12f4ef03479f2fcacdfb756920d41c4e09d1c4f6b55a2279761ac"
QRF_PARAMETERS = {
    "n_estimators": 256,
    "criterion": "squared_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 10,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 0.7,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "max_samples": None,
    "oob_score": False,
    "random_state": 2026,
    "n_jobs": 8,
    "verbose": 0,
    "warm_start": False,
    "ccp_alpha": 0.0,
    "monotonic_cst": None,
}


class FitBudget:
    """Small explicit counter used by the registered recency model wrapper."""

    def __init__(self, limit: int):
        self.limit = limit
        self.allowed = 0

    def allow(self, _model) -> None:
        self.allowed += 1
        if self.allowed > self.limit:
            raise ContractError("registered recency fit budget exceeded")


def _json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recipe() -> dict:
    registration = load_yaml(REPO / "configs/optimization_v0_17/experiment.yaml")
    value = {
        "candidate": "V10_V6I_IRON_V8_TIME",
        "stage": "test_a",
        "base_algorithm": base_algorithm(),
        "registration": registration,
        "fit_months": list(FIT_MONTHS),
        "component_weights": {COMPONENTS[0]: 0.8, COMPONENTS[1]: 0.2},
        "recency": {
            "target": "tap_iron",
            "half_life_days": HALF_LIFE_DAYS,
            "parameters": FROZEN_PARAMETERS,
        },
        "iron_coefficient": {
            "source": "causal monthly OOF April-November",
            "objective": "constrained scalar LAD",
            "tie_break": "smallest minimizer",
            "range": [0.0, 1.0],
        },
        "qrf": {
            "protocol": "QRF_FULLTRAIN_LEAF_v1",
            "parameters": QRF_PARAMETERS,
            "statistic": "lower weighted median",
        },
        "known_result_csv_sha256": EXPECTED_RESULT_SHA256,
        "known_original_zip_sha256": ORIGINAL_ZIP_SHA256,
        "test_targets": "forbidden",
    }
    # Frozen manifests are JSON.  Normalize dataclass tuples now so an
    # immediate load/validate cycle compares the same representation.
    return json.loads(json.dumps(value, ensure_ascii=False))


def source_files() -> dict[str, Path]:
    relative = [
        "src/bf_tap/optimization/v10_rebuild.py",
        "scripts/optimization_v17_final_submission.py",
        "src/bf_tap/optimization/final_lifecycle.py",
        "src/bf_tap/optimization/lifecycle.py",
        "src/bf_tap/optimization/component_export.py",
        "src/bf_tap/optimization/refresh_factorial.py",
        "src/bf_tap/optimization/history_stable.py",
        "src/bf_tap/optimization/recency_model.py",
        "src/bf_tap/optimization/rate_model.py",
        "src/bf_tap/optimization/structural.py",
        "src/bf_tap/optimization/qrf_time_run.py",
        "src/bf_tap/models/baseline.py",
        "src/bf_tap/submission.py",
        "workers/qrf_v015/final_rebuild.py",
        "workers/qrf_v015/worker.py",
        "workers/qrf_v015/qrf_model.py",
        "workers/qrf_v015/preprocessing.py",
        "workers/qrf_v015/pyproject.toml",
        "workers/qrf_v015/uv.lock",
        "configs/optimization_v0_17/experiment.yaml",
        "configs/optimization_v0_17/access_scope.yaml",
        "configs/optimization_v0_8/experiment.yaml",
        "configs/optimization_v0_12/experiment.yaml",
        "configs/baseline.yaml",
        "configs/features.yaml",
        "configs/data_contract.yaml",
        "configs/protection.yaml",
        "configs/optimization_v0_2/features.yaml",
        "configs/optimization_v0_2/experiment.yaml",
        "uv.lock",
        "pyproject.toml",
    ]
    return {name: REPO / name for name in relative}


def _input_paths(data_config: str | Path) -> dict[str, str]:
    paths = load_yaml(data_config)["paths"]
    names = (
        "train_samples",
        "tap_history_train",
        "operation_hourly",
        "burden_change",
        "data_dictionary",
        "test_a_samples",
        "test_b_samples",
        "test_c_samples",
    )
    if any(not paths.get(name) for name in names):
        raise ContractError("complete official data paths are required")
    return {name: paths[name] for name in names}


def freeze(data_config: str | Path, output: str | Path) -> Path:
    """Freeze all identities and the exact V10 recipe without reading targets."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    policy = load_protection_policy(REPO / "configs/protection.yaml")
    paths = _input_paths(data_config)
    inputs = file_identities(paths)
    candidate = recipe()
    metadata = label_metadata(paths, candidate["base_algorithm"]["semantic"])
    test = sample_metadata(paths["test_a_samples"])
    cutoff = test.reference_time.min()
    selected = eligible(metadata, cutoff)
    if len(metadata) != EXPECTED_TRAIN_ROWS or len(selected) != EXPECTED_TRAIN_ROWS:
        raise ContractError("official final training row count differs")
    if len(test) != EXPECTED_TEST_ROWS or test.sample_id.duplicated().any():
        raise ContractError("official test-A identity differs")
    if set(test.sample_id.astype(str)) & set(metadata.sample_id.astype(str)):
        raise ContractError("training and test-A sample IDs overlap")
    if cutoff != pd.Timestamp(candidate["registration"]["fit_cutoff"]):
        raise ContractError("test-A cutoff differs from frozen V10 recipe")
    manifest = {
        "schema_version": "frozen-algorithm-v3",
        "algorithm_identity": IDENTITY,
        "policy_digest": policy.digest,
        "algorithm": candidate,
        "algorithm_sha256": stable_digest(candidate),
        "inputs": inputs,
        "source_files": file_identities(source_files()),
        "final_fit_cutoff": str(cutoff),
        "final_eligibility": eligibility_identity(selected),
        "stages": {
            "test_a": {
                "rows": len(test),
                "sample_ids_sha256": stable_digest(test.sample_id.astype(str).tolist()),
                "reference_min": str(test.reference_time.min()),
                "reference_max": str(test.reference_time.max()),
            }
        },
        "protected_interval": [str(policy.protected_start), str(policy.protected_end)],
        "selection": "fixed_from_existing_user_reported_test_a_evidence",
        "test_distribution_used_for_selection": False,
        "protected_labels_read": False,
    }
    atomic_write_json(output / "frozen_manifest.json", manifest)
    atomic_write_json(
        output / "freeze_receipt.json",
        {
            "status": "PASS_FROZEN_BEFORE_PROTECTED_ACCESS",
            "manifest_sha256": file_sha256(output / "frozen_manifest.json"),
            "rows": {"train": len(metadata), "test_a": len(test)},
            "protected_labels_read": False,
            "test_targets_read": False,
        },
    )
    return output / "frozen_manifest.json"


def validate_manifest(path: str | Path) -> tuple[dict, object]:
    manifest = _json(path)
    policy = load_protection_policy(REPO / "configs/protection.yaml")
    expected = recipe()
    if (
        manifest.get("schema_version") != "frozen-algorithm-v3"
        or manifest.get("algorithm_identity") != IDENTITY
        or manifest.get("policy_digest") != policy.digest
        or manifest.get("algorithm") != expected
        or manifest.get("algorithm_sha256") != stable_digest(expected)
        or manifest.get("protected_interval")
        != [str(policy.protected_start), str(policy.protected_end)]
    ):
        raise ProtectedLabelError("frozen V10 manifest identity differs")
    verify_file_identities(manifest["inputs"])
    verify_file_identities(manifest["source_files"])
    paths = {name: value["path"] for name, value in manifest["inputs"].items()}
    test = sample_metadata(paths["test_a_samples"])
    metadata = label_metadata(paths, expected["base_algorithm"]["semantic"])
    cutoff = test.reference_time.min()
    if str(cutoff) != manifest["final_fit_cutoff"]:
        raise ProtectedLabelError("frozen cutoff differs from current test-A metadata")
    if eligibility_identity(eligible(metadata, cutoff)) != manifest["final_eligibility"]:
        raise ProtectedLabelError("frozen final eligibility differs")
    return manifest, policy


def _protected_final_inputs(manifest_path, authorization, ledger):
    manifest, policy = validate_manifest(manifest_path)
    access = authorize_protected_operation(
        policy=policy,
        lifecycle="final_training",
        purpose=PURPOSES["final_training"],
        frozen_manifest=manifest_path,
        authorization=authorization,
        ledger=ledger,
        algorithm_identity=IDENTITY,
    )
    paths = {name: value["path"] for name, value in manifest["inputs"].items()}
    semantic = manifest["algorithm"]["base_algorithm"]["semantic"]
    metadata = label_metadata(paths, semantic)
    cutoff = pd.Timestamp(manifest["final_fit_cutoff"])
    selected = eligible(metadata, cutoff)
    selected = selected.loc[selected.reference_time < policy.protected_end]
    if eligibility_identity(selected) != manifest["final_eligibility"]:
        raise ProtectedLabelError("authorized final rows differ from frozen identity")
    labels = _read_eligible_rows(paths["train_samples"], selected.sample_id).merge(
        selected[["sample_id", "label_available_at"]], on="sample_id", validate="one_to_one"
    )
    history = _read_eligible_rows(paths["tap_history_train"], selected.sample_id)
    available = semantic["targets"]["available_at_column"]
    for column in {"reference_time", semantic["sources"]["history"]["end_time_column"], available}:
        history[column] = parse_local_time(history[column], column)
    history["available_at"] = history[available]
    labels = labels.sort_values(["reference_time", "sample_id"], kind="mergesort").reset_index(drop=True)
    history = history.sort_values(["reference_time", "sample_id"], kind="mergesort").reset_index(drop=True)
    validate_samples(labels, labeled=True)
    validate_history(history)
    validate_cross_table_consistency(labels, history)
    if not labels.sample_id.astype(str).equals(history.sample_id.astype(str)):
        raise ContractError("final label/history row identities differ")
    if not np.array_equal(labels[list(TARGETS)].to_numpy(), history[list(TARGETS)].to_numpy()):
        raise ContractError("final label/history target values differ")
    return manifest, paths, labels, history, access


def _training_rows(labels: pd.DataFrame, history: pd.DataFrame, cutoff: pd.Timestamp):
    train = eligible(labels, cutoff).reset_index(drop=True)
    h = history.loc[
        (history.reference_time < cutoff) & (history.available_at <= cutoff)
    ].copy()
    h = h.sort_values(["reference_time", "sample_id"], kind="mergesort").reset_index(drop=True)
    if not train.sample_id.astype(str).equals(h.sample_id.astype(str)):
        raise ContractError("fold label/history row identities differ")
    if not np.array_equal(train[list(TARGETS)].to_numpy(), h[list(TARGETS)].to_numpy()):
        raise ContractError("fold label/history targets differ")
    return train, h


def _fit_fold(
    *,
    root: Path,
    name: str,
    cutoff: pd.Timestamp,
    labels: pd.DataFrame,
    history: pd.DataFrame,
    builder: ComponentFeatures,
    manifest: dict,
    contract: dict,
    recency_budget: FitBudget,
):
    train, h = _training_rows(labels, history, cutoff)
    a = manifest["algorithm"]["base_algorithm"]
    folder = root / "models" / name
    folder.mkdir(parents=True, exist_ok=False)
    training = {
        "fit_cutoff": str(cutoff),
        "history_cutoff": str(cutoff),
        "label_available_cutoff": str(cutoff),
        "variant": "R2",
        "rows": len(train),
        "sample_ids_sha256": stable_digest(train.sample_id.astype(str).tolist()),
        "reference_max": str(train.reference_time.max()),
        "label_available_max": str(train.label_available_at.max()),
        "history_available_max": str(h.available_at.max()),
        "training_history_policy": "original_per_sample_asof",
    }
    base_models = {}
    matrices = {}
    for component in COMPONENTS:
        intent = folder / f"{component}_intent.json"
        atomic_write_json(intent, {**training, "component": component, "target_fits": 2})
        x = Context.X(builder, train[META], cutoff, component, "R2", h)
        model = DualTargetBaseline(a["baseline"]["parameters"], ("spout_no",))
        model.fit(x, train[list(TARGETS)])
        destination = folder / "base" / component
        model.save(
            destination,
            history_snapshot=h,
            metadata={
                "baseline_config": a["baseline"],
                "feature_config": a["features"],
                "semantic_contract": a["semantic"],
                "contract_digests": a["contract_digests"],
                "inference_source_contract": contract,
                "training": {**training, "component": component},
                "code_identity": manifest["source_files"],
                "environment": runtime_environment(),
                "lockfile_sha256": manifest["source_files"]["uv.lock"]["sha256"],
            },
        )
        restored = DualTargetBaseline.load(destination)
        if not np.array_equal(model.predict_raw(x).to_numpy(), restored.predict_raw(x).to_numpy()):
            raise ContractError("saved base model prediction differs")
        base_models[component] = restored
        matrices[component] = x

    atomic_write_json(folder / "rate_intent.json", {**training, "target_fits": 1})
    rate = RateModel().fit(
        matrices[COMPONENTS[0]], train.tap_iron, train.tap_time_len, a["baseline"]["parameters"]
    )
    rate.save(
        folder / "rate",
        {
            "parameters": a["baseline"]["parameters"],
            "training": training,
            "inference_source_contract": contract,
            "manifest_sha256": file_sha256(root / "frozen_manifest.json"),
            "rate_spec": load_yaml(REPO / "configs/optimization_v0_8/experiment.yaml")["rate"],
            "algorithm_contract": a["contract_digests"],
        },
        h,
    )
    rate = RateModel.load(folder / "rate")

    x_recency = matrices[COMPONENTS[0]].copy()
    ids = pd.Index(train.sample_id.astype(str), name="sample_id")
    x_recency.index = ids
    metadata = train[["sample_id", "spout_no", "reference_time", "label_available_at"]].copy()
    metadata = metadata.rename(columns={"label_available_at": "available_at"})
    weight, detail = weights(metadata, cutoff)
    detail.to_csv(folder / "recency_weights.csv", index=False)
    recency_training = {
        **training,
        "training_ids_sha256": stable_digest(ids.tolist()),
        "cutoff": str(cutoff),
        "available_max": str(metadata.available_at.max()),
        "feature_schema": schema(x_recency),
        "weight_sha256": file_sha256(folder / "recency_weights.csv"),
    }
    atomic_write_json(folder / "recency_iron_intent.json", recency_training)
    target = pd.Series(train.tap_iron.to_numpy(), index=ids, name="tap_iron")
    recency = RecencyModel().fit(
        x_recency,
        target,
        weight,
        metadata,
        cutoff,
        "tap_iron",
        schema(x_recency),
        recency_budget,
    )
    recency.save(folder / "recency_iron", recency_training)
    recency = RecencyModel.load(folder / "recency_iron")
    record = {
        **training,
        "base_target_fits": 4,
        "rate_fits": 1,
        "recency_iron_fits": 1,
        "bundles": {
            component: file_sha256(folder / "base" / component / "bundle.json")
            for component in COMPONENTS
        },
        "rate_bundle": file_sha256(folder / "rate" / "bundle.json"),
        "recency_bundle": file_sha256(folder / "recency_iron" / "bundle.json"),
    }
    atomic_write_json(folder / "fit_record.json", record)
    return {
        "base": base_models,
        "rate": rate,
        "recency": recency,
        "history": h,
        "training": training,
    }


def _predict_parts(models, samples: pd.DataFrame, cutoff: pd.Timestamp, builder: ComponentFeatures):
    samples = samples[META].copy()
    history = models["history"]
    entry09 = {"cutoff": str(cutoff), "component": COMPONENTS[0], "variant": "R2"}
    entry04 = {"cutoff": str(cutoff), "component": COMPONENTS[1], "variant": "R2"}
    x09 = builder.X(samples, entry09, history)
    x04 = builder.X(samples, entry04, history)
    p09 = models["base"][COMPONENTS[0]].predict_raw(x09).clip(lower=0.0)
    p04 = models["base"][COMPONENTS[1]].predict_raw(x04).clip(lower=0.0)
    ids = samples.sample_id.astype(str).tolist()
    old_iron = 0.8 * p09[PRED[0]].to_numpy() + 0.2 * p04[PRED[0]].to_numpy()
    old_time = 0.8 * p09[PRED[1]].to_numpy() + 0.2 * p04[PRED[1]].to_numpy()
    rate = models["rate"].predict(x09)
    recency_x = x09.copy()
    recency_x.index = pd.Index(ids, name="sample_id")
    direct_iron = models["recency"].predict(recency_x)
    base_iron = 0.8 * direct_iron + 0.2 * p04[PRED[0]].to_numpy()
    direction = np.zeros(len(samples), dtype=float)
    usable = rate > 1e-6
    direction[usable] = (rate * old_time - base_iron)[usable]
    values = np.column_stack([old_iron, old_time, rate, direct_iron, base_iron, direction])
    if not np.isfinite(values).all() or (values[:, :5] < 0).any():
        raise ContractError("nonfinite or negative V10 iron inputs")
    return pd.DataFrame(
        {
            "sample_id": ids,
            PRED[0]: old_iron,
            PRED[1]: old_time,
            "pred_rate": rate,
            "direct_iron": direct_iron,
            "base_iron": base_iron,
            "direction_iron": direction,
        },
        index=samples.index,
    )


def _fit_beta(oof: pd.DataFrame, cutoff: pd.Timestamp):
    selected = select_oof(oof, cutoff, minimum=100)
    beta = lad_coefficient(selected.tap_iron, selected.base_iron, selected.direction_iron)
    residual = selected.base_iron.to_numpy() + beta * selected.direction_iron.to_numpy() - selected.tap_iron.to_numpy()
    zero = np.isclose(residual, 0.0, atol=1e-10, rtol=0.0)
    fixed = float(np.sum(np.sign(residual[~zero]) * selected.direction_iron.to_numpy()[~zero]))
    kink = float(np.abs(selected.direction_iron.to_numpy()[zero]).sum())
    if not 0.0 <= beta <= 1.0 or (beta > 0 and fixed - kink > 1e-8) or (beta < 1 and fixed + kink < -1e-8):
        raise ContractError("final iron LAD optimality certificate failed")
    return beta, selected, {"fixed_subgradient": fixed, "kink_mass": kink}


def _load_final_models(folder: Path):
    return {
        "base": {
            component: DualTargetBaseline.load(folder / "base" / component)
            for component in COMPONENTS
        },
        "rate": RateModel.load(folder / "rate"),
        "recency": RecencyModel.load(folder / "recency_iron"),
        "history": DualTargetBaseline.load(folder / "base" / COMPONENTS[0]).load_history_snapshot(),
    }


def _cold_root_checks(models, samples, op, burden, a, beta, expected):
    groups = {
        "reverse": samples.iloc[::-1],
        "subset": samples.iloc[:: max(1, len(samples) // 7)],
        "single": samples.iloc[[len(samples) // 2]],
    }
    checks = {}
    expected_indexed = expected.set_index("sample_id")
    for name, group in groups.items():
        builder = ComponentFeatures(a, op, burden)
        parts = _predict_parts(models, group[META], pd.Timestamp(models["training"]["fit_cutoff"]), builder)
        iron = np.maximum(0.0, parts.base_iron + beta * parts.direction_iron)
        wanted = expected_indexed.loc[group.sample_id.astype(str), "pred_tap_iron"].to_numpy()
        if not np.array_equal(iron.to_numpy(), wanted):
            raise ContractError(f"root iron prediction changes for {name}")
        checks[name] = True
    chunks = []
    for start in range(0, len(samples), 127):
        group = samples.iloc[start : start + 127]
        builder = ComponentFeatures(a, op, burden)
        parts = _predict_parts(models, group[META], pd.Timestamp(models["training"]["fit_cutoff"]), builder)
        part = pd.DataFrame(
            {
                "sample_id": group.sample_id.astype(str).to_numpy(),
                "pred_tap_iron": np.maximum(0.0, parts.base_iron + beta * parts.direction_iron),
            }
        )
        chunks.append(part)
    chunked = pd.concat(chunks, ignore_index=True)
    if not np.array_equal(chunked.pred_tap_iron.to_numpy(), expected.pred_tap_iron.to_numpy()):
        raise ContractError("root iron prediction changes with chunking")
    checks["chunks"] = True
    return checks


def build(manifest_path, authorization, ledger, worker_python, output_root):
    manifest_path = Path(manifest_path).resolve()
    root = Path(output_root).resolve()
    if manifest_path != root / "frozen_manifest.json":
        raise ContractError("build must use the frozen manifest inside its unique run directory")
    if (root / "build_started.json").exists():
        raise ContractError("this final build was already attempted; use a new run directory")
    atomic_write_json(root / "build_started.json", {"status": "STARTED", "manifest_sha256": file_sha256(manifest_path)})
    manifest, paths, labels, history, access = _protected_final_inputs(manifest_path, authorization, ledger)
    a = manifest["algorithm"]["base_algorithm"]
    op, burden, _ = _load_process_sources(paths, a["semantic"], a["features"])
    contract = build_inference_source_contract(
        manifest["inputs"], semantic_contract_sha256=a["contract_digests"]["semantic_contract_sha256"]
    )
    metadata = label_metadata(paths, a["semantic"])
    builder = ComponentFeatures(a, op, burden)
    recency_budget = FitBudget(len(FIT_MONTHS) + 1)
    oof_rows = []
    fit_counts = {"base_target": 0, "rate": 0, "recency_iron": 0, "qrf": 0, "qrf_preprocessor": 0}
    for month in FIT_MONTHS:
        cutoff = stamp(month)
        models = _fit_fold(
            root=root,
            name=f"oof_{month:02d}",
            cutoff=cutoff,
            labels=labels,
            history=history,
            builder=builder,
            manifest=manifest,
            contract=contract,
            recency_budget=recency_budget,
        )
        fit_counts["base_target"] += 4
        fit_counts["rate"] += 1
        fit_counts["recency_iron"] += 1
        evaluation = metadata.loc[
            (metadata.reference_time >= cutoff)
            & (metadata.reference_time < cutoff + pd.DateOffset(months=1)),
            [*META, "label_available_at"],
        ].copy()
        parts = _predict_parts(models, evaluation[META], cutoff, builder)
        part = evaluation.merge(parts, on="sample_id", validate="one_to_one")
        truth = labels[["sample_id", *TARGETS]]
        part = part.merge(truth, on="sample_id", validate="one_to_one")
        for column, value in (
            ("fold_cutoff", cutoff),
            ("train_reference_max", models["training"]["reference_max"]),
            ("train_available_max", models["training"]["label_available_max"]),
            ("history_available_max", models["training"]["history_available_max"]),
        ):
            part[column] = pd.Timestamp(value)
        part.to_csv(root / "models" / f"oof_{month:02d}" / "oof.csv", index=False)
        oof_rows.append(part)
        builder.cache.clear()
        print(f"V10 reconstruction OOF {month:02d}: train={models['training']['rows']} eval={len(part)}", flush=True)

    oof = pd.concat(oof_rows, ignore_index=True)
    if oof.sample_id.duplicated().any():
        raise ContractError("causal OOF sample IDs overlap")
    cutoff = pd.Timestamp(manifest["final_fit_cutoff"])
    beta, selected, certificate = _fit_beta(oof, cutoff)
    selected.to_csv(root / "final_iron_coefficient_oof.csv", index=False)
    coefficient = {
        "target": "tap_iron",
        "beta": beta,
        "cutoff": str(cutoff),
        "rows": len(selected),
        "OOF_ids_sha256": stable_digest(selected.sample_id.astype(str).tolist()),
        "OOF_sha256": file_sha256(root / "final_iron_coefficient_oof.csv"),
        "reference_max": str(selected.reference_time.max()),
        "available_max": str(selected.label_available_at.max()),
        "certificate": {**certificate, "smallest_minimizer_verified": True},
    }
    atomic_write_json(root / "final_iron_coefficient.json", coefficient)

    final_models = _fit_fold(
        root=root,
        name="final",
        cutoff=cutoff,
        labels=labels,
        history=history,
        builder=builder,
        manifest=manifest,
        contract=contract,
        recency_budget=recency_budget,
    )
    fit_counts["base_target"] += 4
    fit_counts["rate"] += 1
    fit_counts["recency_iron"] += 1
    if final_models["training"]["rows"] != EXPECTED_TRAIN_ROWS:
        raise ContractError("final model training row count differs")

    test = sample_metadata(paths["test_a_samples"])[META]
    final_parts = _predict_parts(final_models, test, cutoff, builder)
    iron = np.maximum(0.0, final_parts.base_iron + beta * final_parts.direction_iron)
    root_predictions = pd.DataFrame(
        {"sample_id": test.sample_id.astype(str).to_numpy(), "pred_tap_iron": iron.to_numpy()}
    )
    root_predictions.to_csv(root / "V6I_iron_full_precision.csv", index=False)

    train, h = _training_rows(labels, history, cutoff)
    x_train = Context.X(builder, train[META], cutoff, COMPONENTS[0], "R2", h)
    x_test = Context.X(builder, test[META], cutoff, COMPONENTS[0], "R2", h)
    if schema(x_train) != schema(x_test):
        raise ContractError("final QRF train/test raw schema differs")
    qrf_root = root / "qrf"
    train_handoff = pack_qrf_handoff(qrf_root / "features" / "train.npz", x_train, train[META], cutoff, h)
    evaluation_handoff = pack_qrf_handoff(qrf_root / "features" / "evaluation.npz", x_test, test[META], cutoff)
    # Keep the virtualenv launcher path itself.  Resolving its `python` symlink
    # would escape the worker environment and silently drop scikit-learn/joblib.
    worker_python = Path(worker_python)
    if not worker_python.is_absolute():
        worker_python = Path.cwd() / worker_python
    worker_python = worker_python.absolute()
    worker_script = REPO / "workers/qrf_v015/final_rebuild.py"
    if not worker_python.is_file():
        raise FileNotFoundError(worker_python)
    subprocess.run(
        [
            str(worker_python),
            str(worker_script),
            "--train",
            str(qrf_root / "features" / "train.npz"),
            "--evaluation",
            str(qrf_root / "features" / "evaluation.npz"),
            "--output",
            str(qrf_root / "model"),
        ],
        cwd=worker_script.parent,
        check=True,
    )
    worker_bundle = _json(qrf_root / "model" / "bundle.json")
    if worker_bundle["parameters"] != QRF_PARAMETERS or worker_bundle["protocol"] != "QRF_FULLTRAIN_LEAF_v1":
        raise ContractError("worker QRF recipe differs from frozen manifest")
    if worker_bundle["train_input"] != train_handoff or worker_bundle["evaluation_input"] != evaluation_handoff:
        raise ContractError("worker QRF handoff identity differs")
    fit_counts["qrf"] = worker_bundle["fit_counts"]["forest"]
    fit_counts["qrf_preprocessor"] = worker_bundle["fit_counts"]["preprocessor"]
    with np.load(qrf_root / "model" / "predictions.npz", allow_pickle=False) as arrays:
        ids = arrays["ids"].astype(str).tolist()
        qrf_time = arrays["median"].astype(float)
    if ids != test.sample_id.astype(str).tolist():
        raise ContractError("QRF prediction IDs or order differ from test-A")
    result = pd.DataFrame(
        {
            "sample_id": ids,
            "pred_tap_iron": iron.to_numpy(),
            "pred_tap_time_len": qrf_time,
        }
    )
    validate_submission(result, test.sample_id)

    restored = _load_final_models(root / "models" / "final")
    restored["training"] = final_models["training"]
    cold_parts = _predict_parts(restored, test, cutoff, ComponentFeatures(a, op, burden))
    cold_iron = np.maximum(0.0, cold_parts.base_iron + beta * cold_parts.direction_iron)
    if not np.array_equal(cold_iron.to_numpy(), iron.to_numpy()):
        raise ContractError("restored full iron prediction differs")
    cold_frame = pd.DataFrame({"sample_id": ids, "pred_tap_iron": cold_iron.to_numpy()})
    cold_checks = _cold_root_checks(restored, test, op, burden, a, beta, cold_frame)
    atomic_write_json(
        root / "cold_validation.json",
        {
            "status": "PASS",
            "root_iron_checks": cold_checks,
            "qrf_checks": worker_bundle["checks"],
            "same_bundle_max_abs_difference": 0.0,
            "inference_fits": 0,
        },
    )

    submission = root / "submission"
    write_submission(result, test.sample_id, submission / "result.csv")
    result_sha = file_sha256(submission / "result.csv")
    exact = result_sha == EXPECTED_RESULT_SHA256
    if not exact:
        atomic_write_json(
            root / "known_result_mismatch.json",
            {
                "status": "BLOCKED_NOT_BYTE_IDENTICAL_TO_SCORED_V10",
                "expected": EXPECTED_RESULT_SHA256,
                "actual": result_sha,
                "package_created": False,
            },
        )
        raise ContractError("reconstructed result.csv does not match the scored V10 identity")
    archive = pack_submission(
        submission / "result.csv",
        stage="test_a",
        team_name="Luqhhh",
        output_dir=submission,
        expected_ids=test.sample_id,
    )
    with ZipFile(archive) as handle:
        if handle.namelist() != ["result.csv"] or file_sha256(submission / "result.csv") != EXPECTED_RESULT_SHA256:
            raise ContractError("final archive content differs after packaging")
    verify_file_identities(manifest["inputs"])
    verify_file_identities(manifest["source_files"])
    release = {
        "status": "PASS_READY_TO_SUBMIT",
        "candidate": "V10_V6I_IRON_V8_TIME",
        "stage": "test_a",
        "rows": len(result),
        "fit_cutoff": str(cutoff),
        "fit_counts": fit_counts,
        "iron_beta": beta,
        "protected_access": access,
        "protected_labels_read": True,
        "test_targets_read": False,
        "test_distribution_used_for_selection": False,
        "result_sha256": result_sha,
        "matches_scored_V10_result_exactly": exact,
        "archive_sha256": file_sha256(archive),
        "original_archive_sha256": ORIGINAL_ZIP_SHA256,
        "archive_note": "new ZIP container; inner result.csv is byte-identical to scored V10",
        "platform_uploads": 0,
        "user_reported_reference_score": 83.1951,
        "platform_evidence_status": "USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED",
    }
    atomic_write_json(root / "release_manifest.json", release)
    (root / "README.txt").write_text(
        "Submit submission/Luqhhh_bf_tap_predict_prelim.zip for the current 335-row test_a stage.\n"
        "The ZIP contains only result.csv. The CSV is byte-identical to the previously scored V10 result.\n"
        "No platform upload was performed automatically. Do not use this package for test_b or test_c.\n",
        encoding="utf-8",
    )
    atomic_write_json(
        root / "receipt.json",
        {
            "status": "PASS",
            "manifest_sha256": file_sha256(manifest_path),
            "release_manifest_sha256": file_sha256(root / "release_manifest.json"),
            "ledger": str(Path(ledger).resolve()),
            "artifacts": file_identities(
                {
                    "zip": archive,
                    "result_csv": submission / "result.csv",
                    "cold_validation": root / "cold_validation.json",
                    "iron_coefficient": root / "final_iron_coefficient.json",
                    "qrf_bundle": qrf_root / "model" / "bundle.json",
                }
            ),
        },
    )
    return archive, release


def verify_package(path: str | Path, data_config: str | Path) -> dict:
    path = Path(path).resolve()
    paths = _input_paths(data_config)
    expected = sample_metadata(paths["test_a_samples"])
    with ZipFile(path) as archive:
        if archive.namelist() != ["result.csv"]:
            raise ContractError("submission ZIP must contain only result.csv")
        payload = archive.read("result.csv")
    temporary = path.parent / f".{path.stem}.verify-result.csv"
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_bytes(payload)
        frame = pd.read_csv(temporary, dtype={"sample_id": "string"})
        validate_submission(frame, expected.sample_id)
        result_sha = file_sha256(temporary)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "PASS" if result_sha == EXPECTED_RESULT_SHA256 else "VALID_FORMAT_UNKNOWN_PREDICTION_IDENTITY",
        "rows": len(frame),
        "members": ["result.csv"],
        "result_sha256": result_sha,
        "matches_scored_V10_result_exactly": result_sha == EXPECTED_RESULT_SHA256,
        "zip_sha256": file_sha256(path),
    }
