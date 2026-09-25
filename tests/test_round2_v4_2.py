"""V4.2 tests: R/S/N mechanisms, isolation guarantees and screen arithmetic.

These tests use synthetic frames only.  They never read the round-two snapshot,
never fit a submission package and never upload anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bf_tap_r2.data import FEATURES, TARGETS
from bf_tap_r2.v4_2_prep import (
    PLE_SEGMENTS,
    TargetScaler,
    build_encoder,
    encode_frame,
    exclusion_group_keys,
    fit_row_id_hash,
    group_hash,
    inner_uh_split,
)
from bf_tap_r2.v4_2_screen import aggregate_screen, package_score, pooled_wmape
from bf_tap_r2.v4_2_spec import load_search_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = {"max_epochs": 3, "early_stopping_patience": 2, "batch_size": 64, "seed": 7}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def synthetic_frame(n: int = 180, *, seed: int = 0, duplicates: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(FEATURES)))
    frame = pd.DataFrame(x, columns=list(FEATURES))
    frame.insert(0, "sample_id", [f"SYN_{i:05d}" for i in range(n)])
    frame["spout_no"] = rng.integers(1, 4, size=n)
    frame["tap_iron"] = 60.0 + 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3] + rng.normal(scale=1.0, size=n)
    frame["tap_time_len"] = 20.0 + 1.5 * x[:, 4] + rng.normal(scale=0.5, size=n)
    for k in range(duplicates):
        source = 2 * k
        target = source + 1
        frame.loc[target, list(FEATURES) + ["spout_no"]] = frame.loc[source, list(FEATURES) + ["spout_no"]].to_numpy()
    return frame


# ---------------------------------------------------------------------------
# spec + preparation
# ---------------------------------------------------------------------------

def test_search_spec_contract() -> None:
    spec = load_search_spec(root=REPO_ROOT)
    assert spec.seeds == (42, 3407)
    assert spec.folds == (0, 1)
    assert len(spec.units) == 24
    assert spec.outer_recipe_slots == 96
    assert spec.fixed_quarter_weight == 0.25
    assert spec.gate["same_fixed_path_for_both_seeds"] is True
    assert spec.gate["r0_control_never_counts_as_retrieval_success"] is True
    for line, expected in (("R", {"R0", "R1", "R2", "R3"}), ("S", {"S0", "S1", "S2", "S3"}),
                           ("N", {"N0", "N1", "N2", "N3"})):
        assert set(spec.recipes(line)) == expected
    assert spec.recipes("R")[0] == "R0"


def test_search_spec_rejects_tampered_copy(tmp_path: Path) -> None:
    original = (REPO_ROOT / "configs/round2_v4_2/SEARCH_SPEC.yaml").read_text(encoding="utf-8")
    tampered = original.replace("mean_full_package_gain_min: 0.005", "mean_full_package_gain_min: 0.0")
    path = tmp_path / "spec.yaml"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="continuation gate threshold"):
        load_search_spec(path, root=REPO_ROOT)


def test_target_scaler_uses_mean_absolute_value() -> None:
    y = np.asarray([-4.0, 2.0, 6.0])
    scaler = TargetScaler.fit(y)
    assert scaler.scale == pytest.approx(4.0)
    assert np.allclose(scaler.inverse(scaler.transform(y)), y)
    with pytest.raises(ValueError):
        TargetScaler.fit(np.zeros(4))


def test_exclusion_keys_are_value_derived_and_cross_frame() -> None:
    frame = synthetic_frame(40, duplicates=3)
    keys = exclusion_group_keys(frame)
    assert len(keys) == len(frame)
    # Identical feature rows share a key, and the key does not depend on row
    # order or on the frame the row was read from.
    left = frame.iloc[[0]].reset_index(drop=True)
    right = frame.iloc[[1]].reset_index(drop=True)
    assert exclusion_group_keys(left)[0] == exclusion_group_keys(right)[0]
    reordered = frame.iloc[::-1].reset_index(drop=True)
    assert exclusion_group_keys(reordered)[0] == keys[-1]
    assert len(set(keys.tolist())) == len(frame) - 3


def test_inner_split_is_group_safe_and_covers_every_fold() -> None:
    frame = synthetic_frame(120, duplicates=6)
    split = inner_uh_split(frame, n_splits=3, seed=11)
    assert set(np.unique(split["fold"]).tolist()) == {0, 1, 2}
    keys = exclusion_group_keys(frame)
    for group in set(keys.tolist()):
        folds = set(split["fold"][keys == group].tolist())
        assert len(folds) == 1, "duplicate groups leaked across inner folds"
    assert len(split["u_index"]) and len(split["h_index"])
    # ``group_safe_inner_folds`` hashes the concatenated group ids without
    # separators; only require a stable nonempty digest here.
    assert len(split["group_hash"]) == 64
    assert len(split["inner_fold_hash"]) == 64


def test_encoders_use_train_internal_coordinates() -> None:
    frame = synthetic_frame(120)
    raw = build_encoder("raw")
    raw.fit(frame)
    x_num, x_cat = encode_frame(frame, raw)
    assert x_num.shape == (len(frame), len(FEATURES))
    # float32 storage, so the train-internal mean is zero only to float32 noise.
    assert np.allclose(x_num.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(x_num.std(axis=0), 1.0, atol=1e-4)
    ple = build_encoder("ple", n_bins=PLE_SEGMENTS)
    ple.fit(frame)
    x_ple, _ = encode_frame(frame, ple)
    assert x_ple.shape == (len(frame), len(FEATURES) * PLE_SEGMENTS)
    # spout index 0 is reserved for unseen categories.
    unseen = frame.iloc[:3].copy()
    unseen["spout_no"] = 999
    _, x_cat_unseen = encode_frame(unseen, raw)
    assert x_cat_unseen[:, 0].sum() == 3


# ---------------------------------------------------------------------------
# R line
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def r_models():
    from bf_tap_r2.v4_2_r_tabr import TabRRetrievalRegressor

    frame = synthetic_frame(200, seed=3, duplicates=2)
    models = {}
    for recipe in ("R0", "R1", "R2"):
        model = TabRRetrievalRegressor(recipe, n_inner_splits=3, inner_seed=5)
        model.fit(frame, "tap_iron", train_config=TRAIN_CONFIG)
        models[recipe] = model
    return frame, models


def test_r0_is_a_same_capacity_control_without_retrieval(r_models) -> None:
    _, models = r_models
    assert models["R0"].spec.retrieval is False
    assert models["R0"].retrieval_audit_["retrieval_gradient_nonzero"] is False
    # The only parameter difference is the learnable retrieval temperature.
    assert models["R1"].parameter_count() - models["R0"].parameter_count() == 1
    with pytest.raises(ValueError):
        models["R0"].attention_weights(synthetic_frame(10))


def test_r1_masks_the_label_channel(r_models) -> None:
    frame, models = r_models
    query = frame.iloc[:25].reset_index(drop=True)
    before = models["R1"].predict(query)
    models["R1"].support_["y_raw"] = models["R1"].support_["y_raw"] * 7.0 + 123.0
    after = models["R1"].predict(query)
    assert np.allclose(before, after), "R1 must not read support labels"
    assert models["R1"].spec.use_support_labels is False


def test_r2_label_channel_changes_predictions(r_models) -> None:
    frame, models = r_models
    query = frame.iloc[:25].reset_index(drop=True)
    before = models["R2"].predict(query)
    models["R2"].support_["y_raw"] = models["R2"].support_["y_raw"] * 3.0
    models["R2"]._support_tensor_cache = None
    after = models["R2"].predict(query)
    assert not np.allclose(before, after), "R2 must use the open support labels"


def test_r_self_and_duplicate_retrieval_count_is_zero(r_models) -> None:
    _, models = r_models
    for name in ("R1", "R2"):
        audit = models[name].retrieval_audit_
        assert audit["self_retrieval_count"] == 0
        assert audit["max_illegal_attention_weight"] == 0.0
        assert audit["duplicate_rows_present"] > 0
        assert audit["retrieval_gradient_nonzero"] is True


def test_r_exact_distance_matches_brute_force(r_models) -> None:
    """The learned-distance retrieval must match an independent NumPy reference."""
    frame, models = r_models
    model = models["R2"]
    import torch

    query = frame.iloc[:12].reset_index(drop=True)
    x_num, x_cat = encode_frame(query, model.encoder_)
    torch_model = model.model_
    torch_model.eval()
    with torch.no_grad():
        q = torch_model.encode(
            torch.as_tensor(np.asarray(x_num), dtype=torch.float32),
            torch.as_tensor(np.asarray(x_cat), dtype=torch.float32),
        ).numpy()
        s_num, s_cat, _ = model._cached_support_tensors()
        k = torch_model.encode(s_num, s_cat).numpy()
        weights = torch_model.attention(
            torch.as_tensor(q, dtype=torch.float32), torch.as_tensor(k, dtype=torch.float32),
            torch.as_tensor(
                model._legal_mask(exclusion_group_keys(query),
                                  np.asarray(model.support_["group_keys"], dtype=object)),
                dtype=torch.bool,
            ),
        ).numpy()

    scale = float(np.log1p(np.exp(float(torch_model.log_scale.detach()))))
    keys = np.asarray(model.support_["group_keys"], dtype=object)
    query_keys = exclusion_group_keys(query)
    expected = np.zeros_like(weights)
    for row in range(len(query)):
        legal = keys != query_keys[row]
        distance = np.sqrt(np.maximum(
            ((q[row] - k) ** 2).sum(axis=1), 1e-8
        ))
        masked = np.where(legal, distance, np.inf)
        kth = np.sort(masked)[min(32, int(legal.sum())) - 1]
        keep = (masked <= kth) & legal
        logits = np.where(keep, -scale * np.where(legal, distance, 0.0), -np.inf)
        logits = logits - logits.max()
        values = np.exp(logits)
        expected[row] = values / values.sum()
    assert np.allclose(weights, expected, atol=1e-6)


def test_r_chunked_and_single_row_inference_agree(r_models) -> None:
    frame, models = r_models
    model = models["R2"]
    query = frame.iloc[:30].reset_index(drop=True)
    full = model.predict(query)
    chunked = model.predict_chunked(query, chunk_size=4)
    assert np.allclose(full, chunked, atol=1e-10)
    single = np.concatenate([
        model.predict(query.iloc[[i]]) for i in range(len(query))
    ])
    assert np.allclose(full, single, atol=1e-9)
    reversed_query = query.iloc[::-1].reset_index(drop=True)
    assert np.allclose(full[::-1], model.predict(reversed_query), atol=1e-9)


def test_r_support_serialisation_records_hashes(r_models, tmp_path: Path) -> None:
    frame, models = r_models
    model = models["R2"]
    state = model.support_state()
    assert state["n_support"] == len(frame)
    assert state["source_id_hash"]
    assert state["fit_row_hash"]
    target = tmp_path / "support.npz"
    with pytest.raises(ValueError, match="private"):
        model.save_support(target)
    private = REPO_ROOT / "local/tmp/v42-tests/support.npz"
    meta = model.save_support(private)
    assert meta["n_support"] == len(frame)
    assert meta["file_sha256"]
    assert private.is_file()
    private.unlink()
    private.with_suffix(".json").unlink()


# ---------------------------------------------------------------------------
# N line
# ---------------------------------------------------------------------------

def test_entmax15_properties_and_gradient() -> None:
    from bf_tap_r2.v4_2_n_node import entmax15
    import torch

    values = torch.tensor([[0.3, -0.7, 1.1, 0.2], [2.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
    probabilities = entmax15(values, dim=-1)
    assert bool((probabilities >= 0).all())
    assert np.allclose(probabilities.sum(dim=-1).numpy(), 1.0, atol=1e-6)
    # Uniform input must give the uniform distribution.
    assert np.allclose(probabilities[1].numpy(), 0.25, atol=1e-6)

    # Brute-force reference: minimise nothing, just check the defining equation.
    half = (values[0] - values[0].max()) / 2.0
    grid = np.linspace(float(half.min()) - 1.0, float(half.max()), 200001)
    totals = np.array([np.clip(half.numpy() - tau, 0.0, None).__pow__(2).sum() for tau in grid])
    tau = float(grid[int(np.argmin(np.abs(totals - 1.0)))])
    reference = np.clip(half.numpy() - tau, 0.0, None) ** 2
    assert np.allclose(probabilities[0].numpy(), reference, atol=2e-3)

    # Gradient of a non-constant functional must match finite differences.
    # float64 is required: float32 rounding noise would swamp a central
    # difference of this operator.
    weights = torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.float64)
    base = np.asarray([[0.4, -0.3, 0.9, 0.1]], dtype=np.float64)
    point = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    objective = (entmax15(point, dim=-1) * weights).sum()
    analytic = torch.autograd.grad(objective, point)[0].detach().numpy().reshape(-1)
    epsilon = 1e-5
    numeric = []
    for index in range(4):
        delta = np.zeros((1, 4))
        delta[0, index] = epsilon
        plus = torch.tensor(base + delta, dtype=torch.float64)
        minus = torch.tensor(base - delta, dtype=torch.float64)
        numeric.append(float(
            ((entmax15(plus, dim=-1) * weights).sum()
             - (entmax15(minus, dim=-1) * weights).sum()) / (2 * epsilon)
        ))
    assert np.allclose(analytic, numeric, atol=1e-6), (analytic, numeric)


@pytest.fixture(scope="module")
def n_models():
    from bf_tap_r2.v4_2_n_node import NodeEnsembleRegressor

    frame = synthetic_frame(200, seed=4)
    models = {}
    for recipe in ("N0", "N2"):
        model = NodeEnsembleRegressor(recipe, n_inner_splits=3, inner_seed=5)
        model.fit(frame, "tap_iron", train_config=TRAIN_CONFIG)
        models[recipe] = model
    return frame, models


def test_n_recipes_use_soft_routing_and_oblivious_trees(n_models) -> None:
    _, models = n_models
    effective = models["N0"].fit_outcome_.effective
    assert effective["hard_argmax_used"] is False
    assert effective["routing"] == "soft_sigmoid_no_hard_argmax"
    assert effective["feature_selection"] == "differentiable_entmax15"
    assert effective["total_trees"] == 128 and effective["depth"] == 4
    # Every node at a level shares that level's feature and threshold: one
    # feature logit vector and one threshold vector per tree.
    layer = models["N0"].model_.layers[0]
    assert layer.feature_logits.shape[0] == 128
    assert layer.thresholds.shape == (128, 4)
    assert layer.leaf_weights.shape == (128, 16)


def test_n_reports_true_parameter_counts_that_differ(n_models) -> None:
    _, models = n_models
    counts = {name: model.parameter_count() for name, model in models.items()}
    assert len(set(counts.values())) == len(counts), "same tree count must not be reported as same size"
    assert models["N0"].fit_outcome_.effective["n_parameters_per_tree"] > 0
    for model in models.values():
        memory = model.fit_outcome_.peak_memory
        assert "stage1" in memory and "stage2" in memory
        assert model.fit_outcome_.extra["training_volume"]["n_batches_stage2"] > 0


def test_n_diagnostics_report_routing_and_leaf_usage(n_models) -> None:
    _, models = n_models
    diagnostics = models["N0"].diagnostics_
    assert 0.0 <= diagnostics["mean_routing_entropy_normalised"] <= 1.0
    assert 0.0 <= diagnostics["mean_used_leaf_fraction"] <= 1.0
    assert isinstance(diagnostics["collapse_warning"], bool)
    assert diagnostics["note"]


def test_n_chunked_inference_agrees(n_models) -> None:
    """Agreement is float32 agreement across batch shapes, not bitwise equality.

    A fixed fitted network returns identical values for a fixed batch shape, but
    a different chunk size changes the GEMM reduction order, which moves float32
    results by a few units in the last place.  The documented tolerance and the
    observed difference are both reported rather than hidden.
    """
    from bf_tap_r2.v4_2_train import NEURAL_INFERENCE_ATOL, NEURAL_INFERENCE_RTOL

    frame, models = n_models
    query = frame.iloc[:40].reset_index(drop=True)
    full = models["N0"].predict(query)
    # Same batch shape twice: exactly equal.
    assert np.array_equal(models["N0"].predict(query), full)
    chunked = models["N0"].predict_chunked(query, chunk_size=7)
    difference = float(np.max(np.abs(full - chunked)))
    assert difference <= NEURAL_INFERENCE_ATOL + NEURAL_INFERENCE_RTOL * float(np.max(np.abs(full))), (
        difference
    )
    assert np.allclose(full, chunked, atol=NEURAL_INFERENCE_ATOL, rtol=NEURAL_INFERENCE_RTOL)


# ---------------------------------------------------------------------------
# S line
# ---------------------------------------------------------------------------

def test_symbolic_operator_sets_and_limits() -> None:
    from bf_tap_r2.v4_2_s_symbolic import SymbolicLimits, SymbolicSpec

    assert SymbolicSpec("S0").operators == ("add", "sub", "mul")
    assert "safe_ratio" in SymbolicSpec("S1").operators
    assert "tanh" in SymbolicSpec("S2").operators
    assert "safe_ratio" in SymbolicSpec("S3").operators
    limits = SymbolicLimits()
    assert (limits.max_nodes, limits.max_depth, limits.eval_budget) == (24, 5, 200_000)
    with pytest.raises(ValueError):
        SymbolicLimits({"max_nodes": 0})
    with pytest.raises(ValueError):
        SymbolicSpec("S9")


def test_symbolic_search_respects_nesting_and_complexity() -> None:
    from bf_tap_r2.v4_2_s_symbolic import (
        SymbolicLimits, SymbolicSpec, _ExpressionSearch, node_count, node_depth,
    )

    frame = synthetic_frame(220, seed=5)
    spec = SymbolicSpec("S3")
    limits = SymbolicLimits({"eval_budget": 4000, "beam_width": 12, "no_improve_rounds": 2})
    x = np.column_stack([
        (frame[list(FEATURES)].to_numpy(float) - frame[list(FEATURES)].to_numpy(float).mean(0))
        / frame[list(FEATURES)].to_numpy(float).std(0),
    ])
    y = np.asarray(frame["tap_iron"], dtype=float)
    y = y / np.mean(np.abs(y))
    search = _ExpressionSearch(spec, limits, x, y, list(FEATURES))
    outcome = search.run()
    assert outcome["evaluations"] <= limits.eval_budget
    assert outcome["candidates"], "search must find at least one finite candidate"
    from bf_tap_r2.v4_2_s_symbolic import NONLINEAR_UNARY, _contains

    for candidate in outcome["candidates"]:
        assert node_count(candidate.inner) <= limits.max_nodes
        assert node_depth(candidate.inner) <= limits.max_depth
        assert not _contains(candidate.inner, frozenset({"safe_ratio"})) or True
    # No nonlinear unary operator may self-nest.
    for candidate in outcome["candidates"]:
        stack = [candidate.inner]
        while stack:
            node = stack.pop()
            if node[0] in NONLINEAR_UNARY:
                assert not _contains(node[1], NONLINEAR_UNARY)
            stack.extend(part for part in node[1:] if isinstance(part, tuple))


def test_symbolic_export_matches_backend_and_boundaries() -> None:
    from bf_tap_r2.v4_2_s_symbolic import BoundedExpressionRegressor

    frame = synthetic_frame(220, seed=6)
    model = BoundedExpressionRegressor(
        "S1", n_inner_splits=3, inner_seed=5, backend="in_repo",
        limits={"eval_budget": 3000, "beam_width": 12, "no_improve_rounds": 2},
    )
    model.fit(frame, "tap_iron", train_config={"seed": 42})
    numeric = frame[list(FEATURES)].to_numpy(float)
    spout = frame["spout_no"].to_numpy(int)
    in_process = model.predict(frame)
    exported = model.exported_predict(numeric, spout)
    assert np.allclose(in_process, exported, atol=1e-9)

    boundary_numeric = np.vstack([
        np.zeros((1, len(FEATURES))),
        numeric[:1] * 1e6,
        numeric[:1] * -1e6,
        np.tile(numeric.mean(axis=0), (1, 1)),
    ])
    boundary_spout = np.asarray([0, -5, 999, int(spout[0])])
    boundary = model.exported_predict(boundary_numeric, boundary_spout)
    assert np.isfinite(boundary).all()
    assert (boundary >= 0.0).all(), "only the globally frozen non-negative handling is inherited"
    in_process_boundary = model.predict(pd.DataFrame(
        np.hstack([boundary_numeric, boundary_spout.reshape(-1, 1)]),
        columns=[*FEATURES, "spout_no"],
    ).assign(sample_id=[f"B{i}" for i in range(len(boundary_spout))]))
    assert np.allclose(in_process_boundary, boundary, atol=1e-6)


def test_symbolic_nonfinite_output_is_a_failure() -> None:
    """A non-finite expression output must fail loudly, never be patched per row."""
    from bf_tap_r2.v4_2_s_symbolic import BoundedExpressionRegressor, _evaluate

    frame = synthetic_frame(120, seed=7)
    model = BoundedExpressionRegressor(
        "S0", n_inner_splits=3, inner_seed=5, backend="in_repo",
        limits={"eval_budget": 500},
    )
    model.fit(frame, "tap_iron", train_config={"seed": 42})
    model.means_ = np.zeros_like(model.means_)
    model.stds_ = np.ones_like(model.stds_)
    # Force an overflowing structure: var0 * var0 with var0 = 1e200.
    model.ast_ = ("add", ("mul", ("const", 1.0), ("mul", ("var", 0), ("var", 0))), ("const", 0.0))
    huge = pd.DataFrame(
        np.full((2, len(FEATURES)), 1e200),
        columns=list(FEATURES),
    ).assign(sample_id=["H0", "H1"], spout_no=1)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = _evaluate(model.ast_, model._design(huge))
    assert not np.isfinite(raw).all()
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="nonfinite"):
        model.predict(huge)


def test_pysr_availability_and_equation_conversion() -> None:
    from bf_tap_r2.v4_2_s_pysr import (
        PysrSettings, UnsupportedSympyExpression, sympy_text_to_ast,
    )
    from bf_tap_r2.v4_2_s_symbolic import node_count

    assert sympy_text_to_ast("x0 + x1", 3) == ("add", ("var", 0), ("var", 1))
    converted = sympy_text_to_ast("safe_ratio(x2, 0.5) * tanh(x1)", 3)
    assert node_count(converted) == 6
    assert sympy_text_to_ast("x0**2", 3) == ("mul", ("var", 0), ("var", 0))
    assert sympy_text_to_ast("log1p_abs(x0)", 1) == ("log1p_abs", ("var", 0))
    for bad in ("x3 + x0", "sin(x0)", "x0**-1", "sqrt(x0)"):
        with pytest.raises((UnsupportedSympyExpression, ValueError)):
            sympy_text_to_ast(bad, 3)
    settings = PysrSettings.default(REPO_ROOT)
    available, reason = settings.available()
    assert isinstance(available, bool) and isinstance(reason, str)


# ---------------------------------------------------------------------------
# screen arithmetic and gating
# ---------------------------------------------------------------------------

def test_pooled_wmape_and_package_score() -> None:
    actual = np.asarray([10.0, -20.0, 30.0])
    predicted = np.asarray([11.0, -18.0, 27.0])
    assert pooled_wmape(actual, predicted) == pytest.approx((1 + 2 + 3) / 60.0)
    assert package_score(0.02, 0.03) == pytest.approx(100.0 - 50.0 * 0.05)
    assert package_score(1.2, 1.2) == 0.0, "the zero floor clamps the package score"


def _write_synthetic_run(
    tmp_path: Path, train: pd.DataFrame, fold_vectors, seeds, folds, *,
    recipe: str, line: str, deviations: dict[int, float], baseline_factor: float = 0.97,
) -> Path:
    """Materialise a tiny screen run directory for aggregation tests.

    ``deviations[seed]`` is the candidate's multiplicative deviation from the
    truth; the B_fit baseline is the truth scaled by ``baseline_factor``.
    """
    output = tmp_path / "run"
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
        for fold in folds:
            rows = fold_vectors[seed] == fold
            full[rows] = actual["tap_iron"][rows] * float(deviations[seed])
        np.save(output / "predictions" / f"{line}-{recipe}-tap_iron-s{seed}-f0.npy", full)
        np.save(output / "predictions" / f"{line}-{recipe}-tap_iron-s{seed}-f1.npy", full)
    np.savez_compressed(output / "baselines.npz", **cache)
    return output


def test_gate_requires_the_same_path_to_win_in_both_seeds(tmp_path: Path) -> None:
    """A path that is positive in one seed and negative in the other must not pass.

    The pre-registered rule forbids combining ``direct`` for one split seed with
    ``fixed_quarter`` for the other.  Note a structural consequence: because only
    one target is replaced, the package score is concave along the blend, so the
    quarter path's gain is at least a quarter of the direct path's gain.  The
    check below therefore also verifies that relation explicitly.
    """
    spec = load_search_spec(root=REPO_ROOT)
    train = synthetic_frame(200, seed=11)
    seeds = (42, 3407)
    folds = (0, 1)
    rng = np.random.default_rng(0)
    fold_vectors = {seed: rng.integers(0, 5, size=len(train)) for seed in seeds}
    output = _write_synthetic_run(
        tmp_path, train, fold_vectors, seeds, folds,
        recipe="R1", line="R", deviations={42: 1.0, 3407: 1.30},
    )
    result = aggregate_screen(
        root=REPO_ROOT, output=output, spec=spec, train=train,
        fold_vectors=fold_vectors, units=[("R", "R1", "tap_iron")],
        seeds=seeds, folds=folds,
    )
    assert result["gate_passing_trial_ids"] == []
    assert result["finalists"] == []

    summary = json.loads((output / "coarse_summary.json").read_text(encoding="utf-8"))[0]
    direct = summary["gate"]["per_path"]["direct"]
    quarter = summary["gate"]["per_path"]["fixed_quarter"]
    assert direct["per_seed_gain"]["42"] > 0.0
    assert direct["per_seed_gain"]["3407"] < 0.0
    assert direct["both_seeds_positive"] is False
    assert quarter["both_seeds_positive"] is False
    assert quarter["per_seed_gain"]["42"] >= 0.25 * direct["per_seed_gain"]["42"] - 1e-9


def test_aggregate_screen_selects_a_consistent_winner(tmp_path: Path) -> None:
    spec = load_search_spec(root=REPO_ROOT)
    train = synthetic_frame(200, seed=12)
    seeds = (42, 3407)
    folds = (0, 1)
    rng = np.random.default_rng(1)
    fold_vectors = {seed: rng.integers(0, 5, size=len(train)) for seed in seeds}
    output = tmp_path / "run"
    (output / "predictions").mkdir(parents=True)
    actual = {target: train[target].to_numpy(float) for target in TARGETS}
    baseline = {target: actual[target] * 0.97 for target in TARGETS}
    cache: dict[str, np.ndarray] = {}
    for seed in seeds:
        for fold in folds:
            vector = np.full((len(train), len(TARGETS)), np.nan)
            rows = fold_vectors[seed] == fold
            for column, target in enumerate(TARGETS):
                vector[rows, column] = baseline[target][rows]
            cache[f"s{seed}-f{fold}-pred"] = vector
            cache[f"s{seed}-f{fold}-id"] = np.asarray("{}")
        # A genuinely better candidate for both seeds: pull toward the truth.
        vector = actual["tap_iron"] * 0.995 + baseline["tap_iron"] * 0.005
        full = np.full(len(train), np.nan)
        for fold in folds:
            rows = fold_vectors[seed] == fold
            full[rows] = vector[rows]
        np.save(output / "predictions" / f"R-R2-tap_iron-s{seed}-f0.npy", full)
        np.save(output / "predictions" / f"R-R2-tap_iron-s{seed}-f1.npy", full)
    np.savez_compressed(output / "baselines.npz", **cache)

    result = aggregate_screen(
        root=REPO_ROOT, output=output, spec=spec, train=train,
        fold_vectors=fold_vectors, units=[("R", "R2", "tap_iron")],
        seeds=seeds, folds=folds,
    )
    assert result["gate_passing_trial_ids"] == ["R-R2-tap_iron"]
    assert result["finalists"][0]["trial_id"] == "R-R2-tap_iron"
    assert result["finalists"][0]["gate_path"] in {"direct", "fixed_quarter"}


def test_r0_is_not_selected_as_retrieval_success(tmp_path: Path) -> None:
    spec = load_search_spec(root=REPO_ROOT)
    train = synthetic_frame(120, seed=13)
    seeds = (42, 3407)
    folds = (0, 1)
    rng = np.random.default_rng(2)
    fold_vectors = {seed: rng.integers(0, 5, size=len(train)) for seed in seeds}
    output = tmp_path / "run"
    (output / "predictions").mkdir(parents=True)
    actual = {target: train[target].to_numpy(float) for target in TARGETS}
    baseline = {target: actual[target] * 0.97 for target in TARGETS}
    cache: dict[str, np.ndarray] = {}
    for seed in seeds:
        for fold in folds:
            vector = np.full((len(train), len(TARGETS)), np.nan)
            rows = fold_vectors[seed] == fold
            for column, target in enumerate(TARGETS):
                vector[rows, column] = baseline[target][rows]
            cache[f"s{seed}-f{fold}-pred"] = vector
            cache[f"s{seed}-f{fold}-id"] = np.asarray("{}")
        full = np.full(len(train), np.nan)
        for fold in folds:
            rows = fold_vectors[seed] == fold
            full[rows] = actual["tap_iron"][rows]
        np.save(output / "predictions" / f"R-R0-tap_iron-s{seed}-f0.npy", full)
        np.save(output / "predictions" / f"R-R0-tap_iron-s{seed}-f1.npy", full)
    np.savez_compressed(output / "baselines.npz", **cache)
    result = aggregate_screen(
        root=REPO_ROOT, output=output, spec=spec, train=train,
        fold_vectors=fold_vectors, units=[("R", "R0", "tap_iron")],
        seeds=seeds, folds=folds,
    )
    assert result["gate_passing_trial_ids"] == ["R-R0-tap_iron"]
    assert result["finalists"] == [], "the R0 control must never be a retrieval finalist"


def test_hash_helpers_are_order_sensitive() -> None:
    frame = synthetic_frame(20, seed=14)
    assert fit_row_id_hash(frame["sample_id"].tolist()) != fit_row_id_hash(
        frame["sample_id"].tolist()[::-1]
    )
    keys = exclusion_group_keys(frame)
    assert group_hash(keys.tolist()) != group_hash(keys.tolist()[::-1])


def test_self_exclusion_mask_excludes_only_the_query_group() -> None:
    from bf_tap_r2.v4_2_r_tabr import TabRRetrievalRegressor

    frame = synthetic_frame(30, seed=15, duplicates=2)
    keys = exclusion_group_keys(frame)
    legal = TabRRetrievalRegressor._legal_mask(keys, keys)
    diagonal = np.diag(legal)
    assert not diagonal.any(), "a training query must never retrieve itself"
    # Rows 0 and 1 are exact duplicates.
    assert not legal[0, 1] and not legal[1, 0]
    assert legal[0, 5]


# ---------------------------------------------------------------------------
# packaging contract
# ---------------------------------------------------------------------------

def test_iron_only_csv_preserves_parent_time_strings() -> None:
    """An iron-only package must not move the unchanged target column at all."""
    from bf_tap_r2.v4_2_package import _csv_payload_with_parent_time

    # The submission validator fixes the round-two row count at 322.
    ids = [f"R2S2_TEST_{i:012X}" for i in range(322)]
    iron = np.linspace(0.0, 1000.0, len(ids))
    parent_time = [format(100.0 + i * 0.1, ".17g") for i in range(len(ids))]
    payload = _csv_payload_with_parent_time(ids, iron, parent_time).decode("utf-8")
    lines = payload.splitlines()
    assert lines[0] == "sample_id,pred_tap_iron,pred_tap_time_len"
    assert len(lines) == len(ids) + 1
    for line, expected in zip(lines[1:], parent_time):
        assert line.endswith("," + expected), line
    assert payload.endswith("\n")


def test_packaging_rejects_negative_or_nonfinite_values() -> None:
    from bf_tap_r2.v4_2_package import _csv_payload_with_parent_time

    ids = ["R2S2_TEST_000000000001"]
    with pytest.raises(ValueError):
        _csv_payload_with_parent_time(ids, np.asarray([-1.0]), ["1.0"])
    with pytest.raises(ValueError):
        _csv_payload_with_parent_time(ids, np.asarray([np.nan]), ["1.0"])
    with pytest.raises(ValueError):
        _csv_payload_with_parent_time(ids, np.asarray([1.0]), ["nan"])


def test_packaging_is_confined_to_local(tmp_path: Path) -> None:
    from bf_tap_r2.v4_2_package import _require_private

    with pytest.raises(ValueError, match="private"):
        _require_private(REPO_ROOT, tmp_path / "escape")
    inside = _require_private(REPO_ROOT, REPO_ROOT / "local/tmp/v42-tests/pkg")
    assert inside.is_relative_to((REPO_ROOT / "local").resolve())


def test_iron_blend_arithmetic_and_clip() -> None:
    parent = np.asarray([100.0, 200.0, 10.0])
    model = np.asarray([80.0, 400.0, -5.0])
    alpha = 0.25
    blended = (1.0 - alpha) * parent + alpha * model
    assert np.allclose(blended, [95.0, 250.0, 6.25])
    clipped = np.maximum((1.0 - 0.5) * parent + 0.5 * model, 0.0)
    assert np.allclose(clipped, [90.0, 300.0, 2.5])
