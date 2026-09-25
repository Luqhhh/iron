"""Round2 V4.2-r2 follow-up tests: repair RNG, stage gates and cache identity.

Everything here uses synthetic frames.  Nothing reads the round-two snapshot,
fits a submission package, or uploads anything.

The three properties this file exists to protect:

1. a network's initialisation is controlled by an explicit seed that is applied
   *before* the first ``torch.randn`` — reproducible in-process, after unrelated
   tasks, and in a brand-new interpreter;
2. the coarse gate and the full-coverage gate are different states, and an
   incomplete coverage is ``NOT_EVALUATED_INCOMPLETE_COVERAGE`` rather than a
   failure or a pass;
3. the repaired candidate's cache key includes the full identity, so a seed,
   source or data change can never hit the pre-repair cache.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v4_2_followup import (
    FOLLOWUP_REPAIR_VERSION,
    REPAIR_TRIAL_ID_SUFFIX,
    _ledger_key,
    b_fit_identity,
    candidate_identity,
    candidate_identity_digest,
    evaluate_stages,
    fusion_protocol_plan,
    replication_protocol_plan,
    select_units,
)
from bf_tap_r2.v4_2_rng import (
    BATCH_ORDER_SEED_OFFSET,
    V42RandomnessPlan,
    initialisation_hash,
    source_digest_payload,
    torch_seed_context,
)
from bf_tap_r2.v4_2_screen import aggregate_screen
from bf_tap_r2.v4_2_spec import load_search_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = {"max_epochs": 3, "early_stopping_patience": 2, "batch_size": 64, "seed": 42}
FAST_CONFIG = {"max_epochs": 2, "early_stopping_patience": 2, "batch_size": 64, "seed": 42}


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------

def synthetic_frame(n: int = 140, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(FEATURES)))
    frame = pd.DataFrame(x, columns=list(FEATURES))
    frame.insert(0, "sample_id", [f"SYN_{i:05d}" for i in range(n)])
    frame["spout_no"] = rng.integers(1, 4, size=n)
    frame["tap_iron"] = (
        60.0 + 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3]
        + rng.normal(scale=1.0, size=n)
    )
    frame["tap_time_len"] = 20.0 + 1.5 * x[:, 4] + rng.normal(scale=0.5, size=n)
    return frame


# ---------------------------------------------------------------------------
# 1. randomness plan and initialisation control
# ---------------------------------------------------------------------------

def test_randomness_plan_is_explicit_and_uses_arithmetic_offsets() -> None:
    plan = V42RandomnessPlan.from_training_seed(42, inner_split_seed=20260925)
    payload = plan.as_dict()
    assert payload["training_seed"] == 42
    assert payload["init_seed"] == 42
    assert plan.stage_init_seed("stage1") == plan.stage_init_seed("stage2") == 42, (
        "both stages must use the one registered initialisation seed"
    )
    assert plan.batch_order_seed_stage1 == 42 + BATCH_ORDER_SEED_OFFSET
    assert plan.batch_order_seed_stage2 == 42 + BATCH_ORDER_SEED_OFFSET + 1
    assert plan.batch_order_seed_stage1 != plan.init_seed
    assert payload["seed_derivation"] == "arithmetic_offset_never_process_local"
    encoded = json.dumps(payload)
    for forbidden in ("worker_id", "os.getpid", "getpid", "hash(", "task_index"):
        assert forbidden not in encoded


def test_initialisation_seed_reaches_the_first_randn_and_is_isolated() -> None:
    import torch

    from bf_tap_r2.v4_2_n_node import NodeSpec, _make_network

    spec = NodeSpec("N2")

    def build(seed: int) -> str:
        with torch_seed_context(seed):
            return initialisation_hash(_make_network(spec, len(FEATURES), 3))

    first = build(42)
    # An unrelated task that consumes the global torch RNG must not move it.
    torch.manual_seed(1234)
    torch.randn(4096)
    second = build(42)
    assert first == second

    other_seed = build(43)
    assert other_seed != first, "a different seed must change the initialisation"

    # And the ambient RNG state must be left exactly as it was found.
    before = torch.random.get_rng_state().clone()
    with torch_seed_context(42):
        _make_network(spec, len(FEATURES), 3)
    assert torch.equal(before, torch.random.get_rng_state())


def test_initialisation_hash_is_order_sensitive_and_repeatable() -> None:
    import torch

    from bf_tap_r2.v4_2_n_node import NodeSpec, _make_network

    spec = NodeSpec("N0")
    with torch_seed_context(7):
        a = _make_network(spec, len(FEATURES), 3)
    with torch_seed_context(7):
        b = _make_network(spec, len(FEATURES), 3)
    assert initialisation_hash(a) == initialisation_hash(b)
    with torch.no_grad():
        first = next(iter(a.parameters()))
        first.add_(0.01)
    assert initialisation_hash(a) != initialisation_hash(b)


def test_full_fit_is_reproducible_in_process() -> None:
    from bf_tap_r2.v4_2_n_node import NodeEnsembleRegressor

    frame = synthetic_frame(140, seed=3)

    def fit_once() -> tuple[str, str, int, np.ndarray]:
        model = NodeEnsembleRegressor("N2", n_inner_splits=3, inner_seed=5)
        model.fit(frame, "tap_iron", train_config=FAST_CONFIG)
        random_meta = model.fit_outcome_.randomness
        return (
            random_meta["initialisation_hash_stage1"],
            random_meta["final_model_state_hash"],
            model.fit_outcome_.best_epoch,
            model.predict(frame),
        )

    first = fit_once()
    # Insert unrelated randomness between the two fits.
    np.random.default_rng(999).normal(size=10_000)
    second = fit_once()
    assert first[0] == second[0], "stage-1 initialisation hash"
    assert first[1] == second[1], "final model state hash"
    assert first[2] == second[2], "selected epoch"
    assert np.array_equal(first[3], second[3]), "predictions"
    assert first[2] >= 1


def test_full_fit_is_reproducible_in_a_new_process(tmp_path: Path) -> None:
    frame = synthetic_frame(110, seed=6)
    frame_path = tmp_path / "frame.pkl"
    frame.to_pickle(frame_path)

    script = f"""
import json, pickle
import numpy as np
from bf_tap_r2.v4_2_n_node import NodeEnsembleRegressor

frame = pickle.load(open({str(frame_path)!r}, "rb"))
model = NodeEnsembleRegressor("N2", n_inner_splits=3, inner_seed=5)
model.fit(frame, "tap_iron", train_config={json.dumps(FAST_CONFIG)})
prediction = model.predict(frame)
print(json.dumps({{
    "init1": model.fit_outcome_.randomness["initialisation_hash_stage1"],
    "init2": model.fit_outcome_.randomness["initialisation_hash_stage2"],
    "final": model.fit_outcome_.randomness["final_model_state_hash"],
    "best_epoch": int(model.fit_outcome_.best_epoch),
    "prediction": prediction.tolist(),
}}))
"""
    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True, timeout=900,
        )
        outputs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert outputs[0]["init1"] == outputs[1]["init1"]
    assert outputs[0]["init2"] == outputs[1]["init2"]
    assert outputs[0]["final"] == outputs[1]["final"]
    assert outputs[0]["best_epoch"] == outputs[1]["best_epoch"]
    assert np.allclose(outputs[0]["prediction"], outputs[1]["prediction"], atol=0.0)

    # The same protocol repeated inside this process must match the children too.
    from bf_tap_r2.v4_2_n_node import NodeEnsembleRegressor

    model = NodeEnsembleRegressor("N2", n_inner_splits=3, inner_seed=5)
    model.fit(frame, "tap_iron", train_config=FAST_CONFIG)
    assert model.fit_outcome_.randomness["initialisation_hash_stage1"] == outputs[0]["init1"]
    assert np.allclose(model.predict(frame), outputs[0]["prediction"], atol=0.0)


def test_model_source_digest_is_content_addressed() -> None:
    payload = source_digest_payload()
    assert set(payload) == {"files", "sha256"}
    assert len(payload["sha256"]) == 64
    assert "v4_2_n_node.py" in payload["files"]


def test_inference_numeric_scope_is_documented_not_overclaimed() -> None:
    """The repair makes a fixed batch shape deterministic, not bitwise across shapes."""
    from bf_tap_r2.v4_2_n_node import NodeEnsembleRegressor
    from bf_tap_r2.v4_2_train import NEURAL_INFERENCE_ATOL, NEURAL_INFERENCE_RTOL

    frame = synthetic_frame(140, seed=5)
    model = NodeEnsembleRegressor("N2", n_inner_splits=3, inner_seed=5)
    model.fit(frame, "tap_iron", train_config=FAST_CONFIG)
    query = frame.iloc[:40].reset_index(drop=True)
    full = model.predict(query)
    assert np.array_equal(model.predict(query), full), "fixed batch shape is deterministic"
    chunked = model.predict_chunked(query, chunk_size=7)
    assert np.allclose(full, chunked, atol=NEURAL_INFERENCE_ATOL, rtol=NEURAL_INFERENCE_RTOL)
    scope = model.fit_outcome_.randomness["reproducibility_scope"]
    assert "bitwise" in scope and "thread" in scope


# ---------------------------------------------------------------------------
# 2. stage gates
# ---------------------------------------------------------------------------

def _write_synthetic_run(
    output: Path,
    train: pd.DataFrame,
    fold_vectors: dict[int, np.ndarray],
    seeds: tuple[int, ...],
    folds: tuple[int, ...],
    *,
    deviation: dict[int, float],
    baseline_factor: float = 0.97,
) -> None:
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    actual = {target: train[target].to_numpy(float) for target in TARGETS}
    cache: dict[str, np.ndarray] = {}
    for seed in seeds:
        for fold in folds:
            vector = np.full((len(train), len(TARGETS)), np.nan)
            rows = fold_vectors[seed] == fold
            for column, target in enumerate(TARGETS):
                vector[rows, column] = actual[target][rows] * baseline_factor
            cache[f"s{seed}-f{fold}-pred"] = vector
            cache[f"s{seed}-f{fold}-id"] = np.asarray("{}")
            full = np.full(len(train), np.nan)
            for fold_inner in folds:
                rows_inner = fold_vectors[seed] == fold_inner
                full[rows_inner] = actual["tap_iron"][rows_inner] * float(deviation[seed])
            np.save(
                output / "predictions" / f"N-N2-tap_iron-s{seed}-f{fold}.npy", full
            )
    np.savez_compressed(output / "baselines.npz", **cache)


def _run_aggregate(tmp_path: Path, *, deviation: dict[int, float], folds: tuple[int, ...]):
    spec = load_search_spec(root=REPO_ROOT)
    train = synthetic_frame(200, seed=21)
    seeds = (42, 3407)
    rng = np.random.default_rng(5)
    fold_vectors = {seed: rng.integers(0, 5, size=len(train)) for seed in seeds}
    output = tmp_path / f"run-{'-'.join(str(f) for f in folds)}"
    output.mkdir(parents=True, exist_ok=True)
    _write_synthetic_run(output, train, fold_vectors, seeds, folds, deviation=deviation)
    aggregate_screen(
        root=REPO_ROOT, output=output, spec=spec, train=train,
        fold_vectors=fold_vectors, units=[("N", "N2", "tap_iron")],
        seeds=seeds, folds=folds, trial_id_suffix=REPAIR_TRIAL_ID_SUFFIX,
    )
    row = json.loads((output / "coarse_summary.json").read_text(encoding="utf-8"))[0]
    return output, row


def test_incomplete_coverage_is_neither_a_pass_nor_a_failure(tmp_path: Path) -> None:
    _, row = _run_aggregate(tmp_path, deviation={42: 0.99, 3407: 0.98}, folds=(0, 1))
    coverage = row["coverage_gate"]
    assert coverage["status"] == "NOT_EVALUATED_INCOMPLETE_COVERAGE"
    assert coverage["evaluated"] is False
    assert coverage["passes"] is None, "partial coverage must not report a boolean pass/fail"
    assert coverage["covered_folds"] == 4 and coverage["expected_folds"] == 10
    assert all(entry["passes"] is None for entry in coverage["per_path"].values())
    assert row["trial_id"].endswith(REPAIR_TRIAL_ID_SUFFIX)
    # The *coarse* gate is still judged, and it is a separate state.
    assert row["gate"]["passes"] is True


def test_complete_coverage_gate_passes_and_fails_explicitly(tmp_path: Path) -> None:
    _, passing = _run_aggregate(tmp_path, deviation={42: 0.99, 3407: 0.98}, folds=(0, 1, 2, 3, 4))
    coverage = passing["coverage_gate"]
    assert coverage["evaluated"] is True
    assert coverage["status"] == "PASSED"
    assert coverage["passes"] is True
    assert coverage["path"] in {"direct", "fixed_quarter"}
    assert coverage["per_path"][coverage["path"]]["both_seeds_positive"] is True

    # A candidate far enough from the truth that even the quarter blend is worse
    # than the baseline: 0.75 * 0.97 + 0.25 * 1.4 sits on the wrong side.
    _, failing = _run_aggregate(tmp_path, deviation={42: 1.4, 3407: 1.5}, folds=(0, 1, 2, 3, 4))
    failed = failing["coverage_gate"]
    assert failed["evaluated"] is True
    assert failed["status"] == "FAILED"
    assert failed["passes"] is False


def test_stage_machine_reports_unrun_stages_as_not_evaluated() -> None:
    incomplete = _row(status="NOT_EVALUATED_INCOMPLETE_COVERAGE", evaluated=False,
                      covered=4, expected=10, paths={"fixed_quarter": 0.009})
    stages = evaluate_stages(
        coarse_row=_coarse_row(passes=True, mean=0.009),
        full_row=incomplete,
    )
    assert stages["stages"]["coarse"]["status"] == "PASSED"
    assert stages["stages"]["full_coverage"]["status"] == "NOT_EVALUATED_INCOMPLETE_COVERAGE"
    assert stages["stages"]["fusion"]["status"] == "NOT_EVALUATED_INCOMPLETE_COVERAGE"
    assert stages["next_action"] == "RUN_FULL_COVERAGE"

    failed = _row(status="FAILED", evaluated=True, covered=10, expected=10,
                  paths={"fixed_quarter": 0.014})
    stopped = evaluate_stages(coarse_row=_coarse_row(passes=True, mean=0.009), full_row=failed)
    assert stopped["stages"]["full_coverage"]["status"] == "FAILED"
    assert stopped["stages"]["fusion"]["status"] == "NOT_EVALUATED"
    assert stopped["next_action"] == "STOP_FULL_GATE_FAILED"
    # Unrun downstream stages are never called failures.
    assert stopped["stages"]["replication"]["status"] == "NOT_EVALUATED"
    assert stopped["stages"]["submission"]["status"] == "NOT_EVALUATED"

    coarse_failed = evaluate_stages(
        coarse_row=_coarse_row(passes=False, mean=-0.001), full_row=None,
    )
    assert coarse_failed["next_action"] == "STOP_COARSE_GATE_FAILED"
    assert coarse_failed["stages"]["fusion"]["status"] == "NOT_EVALUATED"


def _coarse_row(*, passes: bool, mean: float) -> dict:
    return {
        "gate": {
            "passes": bool(passes),
            "path": "fixed_quarter" if passes else None,
            "per_path": {
                "direct": {"mean_gain": -0.1, "both_seeds_positive": False},
                "fixed_quarter": {"mean_gain": float(mean), "both_seeds_positive": bool(passes)},
            },
        },
        "per_seed": {"42": {"fold_level": {"0": {}, "1": {}}},
                     "3407": {"fold_level": {"0": {}, "1": {}}}},
    }


def _row(*, status: str, evaluated: bool, covered: int, expected: int, paths: dict) -> dict:
    return {
        "coverage_gate": {
            "status": status,
            "evaluated": bool(evaluated),
            "covered_folds": int(covered),
            "expected_folds": int(expected),
            "mean_gain_min": 0.02,
            "positive_folds_min": 8,
            "path": "fixed_quarter" if status == "PASSED" else None,
            "per_path": {name: {"mean_gain": value} for name, value in paths.items()},
        },
        "positive_folds": {"fixed_quarter": {"positive": 8, "total": covered}},
    }


def test_fusion_and_replication_protocols_are_recorded_but_disabled() -> None:
    fusion = fusion_protocol_plan()
    assert fusion["enabled"] is False
    assert fusion["alpha_grid"] == [0.0, 0.1, 0.25, 0.5, 1.0]
    assert fusion["tie_break"] == "smaller_alpha"
    assert any("outer validation" in item for item in fusion["forbidden"])
    assert fusion["both_endpoints_refit_in_every_inner_U_H"] is True
    replication = replication_protocol_plan()
    assert replication["enabled"] is False
    assert replication["rerun_training_seed"] == 3407
    assert replication["structure_reselection_on_second_seed"] is False


def test_followup_declarations_keep_path_b_disabled(tmp_path: Path) -> None:
    from bf_tap_r2.v4_2_followup import load_followup_declarations

    declarations = load_followup_declarations(root=REPO_ROOT)
    assert declarations["followup"]["repair_version"] == FOLLOWUP_REPAIR_VERSION
    assert declarations["followup"]["numeric_encoding"] == "raw"
    assert declarations["followup"]["path"] == "fixed_quarter"
    path_b = declarations["path_b"]
    assert path_b["enabled"] is False
    assert path_b["proposed_change"]["feature_logits_shape"] == ["trees", "depth", "features"]
    assert path_b["required_control"].strip().startswith("The repaired shared-selector")
    assert path_b["budget"]["coarse_slots"] == 4
    assert "learnable_temperature" in path_b["proposed_change"]["forbidden_additions"]
    assert "PLE" not in path_b["proposed_change"]["keeps"]

    # Turning the design on must be rejected, not silently honoured.  Replace
    # only the indented YAML key, not the prose mention in the header comment.
    original = (REPO_ROOT / "configs/round2_v4_2/FOLLOWUP_SPEC.yaml").read_text(encoding="utf-8")
    assert "\n  enabled: false\n" in original
    tampered = tmp_path / "followup_spec.yaml"
    tampered.write_text(
        original.replace("\n  enabled: false\n", "\n  enabled: true\n"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="design-only"):
        load_followup_declarations(tampered, root=REPO_ROOT)


# ---------------------------------------------------------------------------
# 3. identity, recipe filtering and cache isolation
# ---------------------------------------------------------------------------

def _identity_inputs(*, training_seed: int = 42):
    spec = load_search_spec(root=REPO_ROOT)
    train = synthetic_frame(160, seed=8)
    rng = np.random.default_rng(2)
    fold_vectors = {seed: rng.integers(0, 5, size=len(train)) for seed in (42, 3407)}
    return spec, train, fold_vectors, training_seed


def test_recipe_filtering_selects_only_the_requested_recipe() -> None:
    spec = load_search_spec(root=REPO_ROOT)
    selected = select_units(spec, lines=["N"], recipes=["N2"], targets=["tap_iron"])
    assert selected == [("N", "N2", "tap_iron")]
    assert ("N", "N2", "tap_time_len") not in selected
    assert len(select_units(spec, recipes=["N2"])) == 2
    assert select_units(spec, recipes=["N2"]) != select_units(spec, lines=["N"], recipes=["N2"], targets=["tap_iron"])
    with pytest.raises(ValueError):
        select_units(spec, recipes=["N9"])


def test_candidate_identity_changes_with_seed_data_and_source() -> None:
    spec, train, fold_vectors, _ = _identity_inputs()

    def build(**overrides):
        kwargs = dict(
            spec=spec, line="N", recipe="N2", target="tap_iron",
            seeds=(42, 3407), folds=(0, 1, 2, 3, 4), fold_vectors=fold_vectors,
            training_seed=42, inner_seed=20260925, torch_threads=4,
        )
        kwargs.update(overrides)
        return candidate_identity(train, **kwargs)

    baseline = build()
    assert baseline["candidate_trial_id"] == f"N-N2-tap_iron{REPAIR_TRIAL_ID_SUFFIX}"
    assert baseline["numeric_encoding"] == "raw", "N2 is the raw-encoded recipe, not PLE"
    assert baseline["seeds"]["batch_order_seed_offset"] == 1_000_003

    changed_seed = build(training_seed=43)
    assert candidate_identity_digest(baseline) != candidate_identity_digest(changed_seed)

    changed_folds = build(folds=(0, 1))
    assert candidate_identity_digest(baseline) != candidate_identity_digest(changed_folds)

    changed_recipe = build(recipe="N3")
    assert candidate_identity_digest(baseline) != candidate_identity_digest(changed_recipe)

    perturbed = train.copy(deep=True)
    perturbed.loc[3, FEATURES[0]] = float(perturbed.loc[3, FEATURES[0]]) + 1e-6
    changed_data = candidate_identity(
        perturbed, spec=spec, line="N", recipe="N2", target="tap_iron",
        seeds=(42, 3407), folds=(0, 1, 2, 3, 4), fold_vectors=fold_vectors,
        training_seed=42, inner_seed=20260925, torch_threads=4,
    )
    assert candidate_identity_digest(baseline) != candidate_identity_digest(changed_data)


def test_ledger_key_separates_pre_and_post_repair_entries() -> None:
    task = {"line": "N", "recipe": "N2", "target": "tap_iron", "seed": 42, "fold": 0}
    key = _ledger_key(task, "digest-a")
    assert FOLLOWUP_REPAIR_VERSION in key
    assert key != _ledger_key(task, "digest-b")
    assert key.endswith("N|N2|tap_iron|42|0")


def test_b_fit_identity_is_independent_of_the_repair() -> None:
    train = synthetic_frame(120, seed=9)
    dependency = {"python": "3.12.12", "packages": {"torch": "2.14.0+cpu"}}
    first = b_fit_identity(
        train, seed=42, fold=0, source_hash="summary:x|ledger:y",
        fold_vector_sha256="a" * 64, dependency=dependency,
    )
    second = b_fit_identity(
        train, seed=42, fold=0, source_hash="summary:x|ledger:y",
        fold_vector_sha256="a" * 64, dependency=dependency,
    )
    assert first == second
    assert first["kind"] == "V42_FOLLOWUP_B_FIT_IDENTITY_V2"
    assert first["model_recipe"] == "V36_fixed_recipe_a_frozen_deployment_weights_v1"
    assert len(first["reference_source_digest"]) == 64
    # The repair source digest is *not* part of the B_fit key: repairing N must
    # not invalidate every frozen base model.
    assert "source_digest" not in first
    assert first["reference_source_digest"] != source_digest_payload()["sha256"]

    other_fold = b_fit_identity(
        train, seed=42, fold=1, source_hash="summary:x|ledger:y",
        fold_vector_sha256="a" * 64, dependency=dependency,
    )
    assert other_fold != first
    other_dep = b_fit_identity(
        train, seed=42, fold=0, source_hash="summary:x|ledger:y",
        fold_vector_sha256="a" * 64, dependency={"python": "3.12.12", "packages": {"torch": "other"}},
    )
    assert other_dep != first
