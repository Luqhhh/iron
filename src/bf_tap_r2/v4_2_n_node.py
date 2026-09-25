"""V4.2 N line: end-to-end differentiable oblivious tree ensemble (NODE-style).

Evidence basis: the NODE paper (https://arxiv.org/abs/1909.06312) and the
authors' implementation (https://github.com/Qwicen/node).

This is an **adapted implementation**, verified numerically and by gradient
checks rather than claimed as a line-by-line reproduction.  What is kept is the
NODE mechanism: an oblivious (per-level feature and threshold) tree ensemble in
which the differentiable feature selection, the soft routing and the leaf
outputs are optimised jointly, with several layers composed by concatenation.

It is explicitly *not* a random forest with leaf medians, and not a weak linear
projection followed by splines.  An oblivious tree is also not treated here as
an arbitrary tree with independent oblique hyperplanes at every node: every node
at one level shares that level's selected feature and threshold.

Four recipes per target with the number of trees fixed at 128 and the depth
fixed at 4:

===== ====================== ====================================
ID    numeric encoding       hierarchy
===== ====================== ====================================
N0    raw standardised       1 layer x 128 trees
N1    eight-segment PLE      1 layer x 128 trees
N2    raw standardised       2 layers x 64 trees each
N3    eight-segment PLE      2 layers x 64 trees each
===== ====================== ====================================

Equal tree counts are *not* equal parameter counts: the true parameter count,
training volume and peak memory are reported per recipe and no strict
parameter-matching claim is made.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .metrics import wmape
from .v4_2_prep import (
    PLE_SEGMENTS,
    TargetScaler,
    build_encoder,
    encode_frame,
    exclusion_group_keys,
    fit_row_id_hash,
    group_hash,
    inner_uh_split,
    transform_hash,
)
from .v4_2_train import NEURAL_LOSS, FitTimer, TrainConfig, TrainOutcome, regression_epochs

__all__ = ["N_RECIPES", "NodeEnsembleRegressor", "NodeSpec", "entmax15", "n_recipe_spec"]

#: The four pre-registered N recipes.
N_RECIPES: dict[str, dict[str, Any]] = {
    "N0": {"recipe_id": "N0", "n_layers": 1, "trees_per_layer": 128, "depth": 4,
           "numeric_encoding": "raw", "hierarchy": "1_layer_x_128_trees"},
    "N1": {"recipe_id": "N1", "n_layers": 1, "trees_per_layer": 128, "depth": 4,
           "numeric_encoding": "ple", "hierarchy": "1_layer_x_128_trees"},
    "N2": {"recipe_id": "N2", "n_layers": 2, "trees_per_layer": 64, "depth": 4,
           "numeric_encoding": "raw", "hierarchy": "2_layers_x_64_trees"},
    "N3": {"recipe_id": "N3", "n_layers": 2, "trees_per_layer": 64, "depth": 4,
           "numeric_encoding": "ple", "hierarchy": "2_layers_x_64_trees"},
}

#: Fixed starting point.  ``selection_sparsity`` is pre-registered as 0.0 so the
#: primary objective stays exactly the shared MAE scheme.
N_START: dict[str, Any] = {
    "total_trees": 128,
    "depth": 4,
    "temperature": 1.0,
    "ple_segments": PLE_SEGMENTS,
    "selection_sparsity": 0.0,
    "init_scale": 0.05,
}


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("The V4.2 N line requires torch") from exc
    return torch


_ENTMAX15_CLASS: Any = None


def _entmax15_class():
    """Build (once) the autograd Function; torch is imported lazily."""
    global _ENTMAX15_CLASS
    if _ENTMAX15_CLASS is not None:
        return _ENTMAX15_CLASS
    torch = _torch()

    class _Entmax15(torch.autograd.Function):
        """Exact alpha=1.5 entmax with an analytic backward pass.

        ``p_i = clamp(x_i / 2 - tau, 0) ** 2`` where ``tau`` solves
        ``sum_i p_i == 1``.  Because ``tau`` depends on the input, treating it as
        a constant would give an approximate gradient; the backward pass here
        uses the implicit-function derivative

            d tau / d x_j = c_j / (2 * sum_i c_i)
            d p_i / d x_j = c_i * delta_ij - c_i * c_j / sum_i c_i

        with ``c_i = clamp(x_i / 2 - tau, 0)``, so the operator is verified by
        finite differences rather than merely asserted to be differentiable.
        """

        @staticmethod
        def forward(ctx, values, dim):
            shift = values.max(dim=dim, keepdim=True).values.detach()
            half = (values - shift) / 2.0
            with torch.no_grad():
                upper = half.max(dim=dim, keepdim=True).values
                lower = upper - 1.0
                for _ in range(60):
                    middle = (lower + upper) / 2.0
                    total = torch.clamp(half - middle, min=0.0).pow(2).sum(dim=dim, keepdim=True)
                    too_large = total > 1.0
                    lower = torch.where(too_large, middle, lower)
                    upper = torch.where(too_large, upper, middle)
                tau = (lower + upper) / 2.0
                c = torch.clamp(half - tau, min=0.0)
                denominator = c.sum(dim=dim, keepdim=True)
                if not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any()):
                    raise ValueError("entmax15 produced an invalid normaliser")
                # The bisection already enforces sum(c ** 2) == 1, so ``c ** 2``
                # *is* the entmax output; dividing by (sum c) ** 2 would change it.
                probabilities = c.pow(2)
            ctx.save_for_backward(c, denominator)
            ctx.dim = dim
            return probabilities

        @staticmethod
        def backward(ctx, grad_output):
            c, denominator = ctx.saved_tensors
            dim = ctx.dim
            inner = (grad_output * c).sum(dim=dim, keepdim=True) / denominator
            return c * (grad_output - inner), None

    _ENTMAX15_CLASS = _Entmax15
    return _ENTMAX15_CLASS


def entmax15(values, dim: int = -1):
    """Sparsemax-family (alpha = 1.5) activation with an exact gradient."""
    torch = _torch()
    if values.ndim < 1:
        raise ValueError("entmax15 requires at least one dimension")
    if dim not in (-1, values.ndim - 1):
        raise ValueError("entmax15 currently supports the last dimension only")
    del torch
    return _entmax15_class().apply(values, int(dim))


def n_recipe_spec(recipe_id: str) -> dict[str, Any]:
    key = str(recipe_id).upper()
    if key not in N_RECIPES:
        raise ValueError(f"Unknown N recipe: {recipe_id!r}")
    return deepcopy(N_RECIPES[key])


class NodeSpec:
    """Validated N recipe specification; nothing is silently ignored."""

    def __init__(self, recipe_id: str, overrides: Mapping[str, Any] | None = None) -> None:
        payload = n_recipe_spec(recipe_id)
        unknown = sorted(set(overrides or {}) - set(N_START))
        if unknown:
            raise ValueError(f"Unknown N starting-point overrides: {unknown}")
        merged = dict(N_START)
        merged.update({str(k): v for k, v in (overrides or {}).items()})
        self.recipe_id = str(payload["recipe_id"])
        self.n_layers = int(payload["n_layers"])
        self.trees_per_layer = int(payload["trees_per_layer"])
        self.depth = int(payload["depth"])
        self.numeric_encoding = str(payload["numeric_encoding"])
        self.hierarchy = str(payload["hierarchy"])
        self.temperature = float(merged["temperature"])
        self.ple_segments = int(merged["ple_segments"])
        self.selection_sparsity = float(merged["selection_sparsity"])
        self.init_scale = float(merged["init_scale"])
        if self.n_layers * self.trees_per_layer != int(merged["total_trees"]):
            raise ValueError(
                "N recipes must keep the total tree count fixed at "
                f"{merged['total_trees']}"
            )
        if self.depth != int(merged["depth"]):
            raise ValueError("N recipes must keep the tree depth fixed")
        if self.temperature <= 0.0 or self.init_scale < 0.0:
            raise ValueError("Invalid N starting point")

    @property
    def total_trees(self) -> int:
        return int(self.n_layers * self.trees_per_layer)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "n_layers": int(self.n_layers),
            "trees_per_layer": int(self.trees_per_layer),
            "total_trees": int(self.total_trees),
            "depth": int(self.depth),
            "numeric_encoding": str(self.numeric_encoding),
            "hierarchy": str(self.hierarchy),
            "temperature": float(self.temperature),
            "ple_segments": int(self.ple_segments),
            "selection_sparsity": float(self.selection_sparsity),
            "init_scale": float(self.init_scale),
            "oblivious": True,
            "note": "oblivious trees share one feature and threshold per level",
        }


def _make_network(spec: NodeSpec, n_num: int, n_cat: int):
    torch = _torch()
    n_leaves = 2 ** int(spec.depth)

    class _ObliviousLayer(torch.nn.Module):
        def __init__(self, n_features: int, n_trees: int) -> None:
            super().__init__()
            self.n_features = int(n_features)
            self.n_trees = int(n_trees)
            self.depth = int(spec.depth)
            self.feature_logits = torch.nn.Parameter(
                spec.init_scale * torch.randn(self.n_trees, self.n_features)
            )
            self.thresholds = torch.nn.Parameter(
                0.05 * torch.randn(self.n_trees, self.depth)
            )
            self.leaf_weights = torch.nn.Parameter(
                0.05 * torch.randn(self.n_trees, n_leaves)
            )
            bits = ((torch.arange(n_leaves)[:, None] >> torch.arange(self.depth)[None, :]) & 1)
            self.register_buffer("leaf_bits", bits.to(torch.float32), persistent=False)

        def selection(self):
            return entmax15(self.feature_logits, dim=1)

        def routing(self, x):
            selected = x @ self.selection().transpose(0, 1)          # (B, T)
            return torch.sigmoid(
                (selected.unsqueeze(-1) - self.thresholds) * float(spec.temperature)
            )                                                        # (B, T, depth)

        def leaf_probabilities(self, response):
            eps = 1e-6
            log_on = torch.log(response.clamp_min(eps))
            log_off = torch.log((1.0 - response).clamp_min(eps))
            bits = self.leaf_bits                                     # (L, depth)
            terms = (
                bits[None, None, :, :] * log_on[:, :, None, :]
                + (1.0 - bits)[None, None, :, :] * log_off[:, :, None, :]
            )
            return torch.exp(terms.sum(dim=-1))                        # (B, T, L)

        def forward(self, x):
            response = self.routing(x)
            probabilities = self.leaf_probabilities(response)
            return torch.einsum("btl,tl->bt", probabilities, self.leaf_weights)

    class _NodeEnsemble(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList()
            width = int(n_num) + int(n_cat)
            for _ in range(int(spec.n_layers)):
                self.layers.append(_ObliviousLayer(width, int(spec.trees_per_layer)))
                width += int(spec.trees_per_layer)
            self.head = torch.nn.Linear(width, 1)

        def forward(self, x_num, x_cat):
            x = torch.cat([x_num, x_cat], dim=1)
            for layer in self.layers:
                x = torch.cat([x, layer(x)], dim=1)
            return self.head(x).reshape(-1)

    return _NodeEnsemble()


class NodeEnsembleRegressor:
    """One NODE-style recipe fitted on a caller-supplied training part."""

    def __init__(
        self,
        recipe_id: str,
        *,
        overrides: Mapping[str, Any] | None = None,
        n_inner_splits: int = 5,
        inner_seed: int = 42,
        torch_threads: int | None = None,
    ) -> None:
        self.spec = NodeSpec(recipe_id, overrides)
        self.n_inner_splits = int(n_inner_splits)
        self.inner_seed = int(inner_seed)
        self.torch_threads = None if torch_threads is None else int(torch_threads)
        self.recipe_id = self.spec.recipe_id

    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        *,
        train_config: Mapping[str, Any] | None = None,
    ) -> "NodeEnsembleRegressor":
        torch = _torch()
        if self.torch_threads is not None:
            torch.set_num_threads(max(1, int(self.torch_threads)))
        config = TrainConfig.from_mapping({**dict(train_config or {}), "loss": NEURAL_LOSS})
        frame = frame.reset_index(drop=True)
        y_raw = np.asarray(frame[target], dtype=np.float64)
        if not np.isfinite(y_raw).all():
            raise ValueError("N line requires finite training targets")
        self.target_ = str(target)

        split = inner_uh_split(frame, n_splits=self.n_inner_splits, seed=self.inner_seed)
        u_frame = frame.iloc[split["u_index"]].reset_index(drop=True)
        h_frame = frame.iloc[split["h_index"]].reset_index(drop=True)
        if len(u_frame) == 0 or len(h_frame) == 0:
            raise ValueError("N line requires a nonempty U/H split")

        u_encoder = build_encoder(self.spec.numeric_encoding, n_bins=self.spec.ple_segments).fit(u_frame)
        u_num, u_cat = encode_frame(u_frame, u_encoder)
        n_num, n_cat = int(np.asarray(u_num).shape[1]), int(np.asarray(u_cat).shape[1])
        u_scaler = TargetScaler.fit(np.asarray(u_frame[target], dtype=np.float64))

        with FitTimer() as timer:
            stage1_net = _make_network(self.spec, n_num, n_cat)
            stage1 = self._run_epochs(
                stage1_net, u_encoder, u_frame, target, u_scaler, config,
                monitor=(h_frame, n_num, n_cat), epochs=int(config.max_epochs),
                early_stopping=True,
            )
        stage1_seconds, stage1_memory = timer.seconds, timer.report()
        best_epoch = max(1, int(stage1["best_epoch"]))

        self.encoder_ = build_encoder(
            self.spec.numeric_encoding, n_bins=self.spec.ple_segments
        ).fit(frame)
        self.scaler_ = TargetScaler.fit(y_raw)
        with FitTimer() as timer:
            self.model_ = _make_network(self.spec, n_num, n_cat)
            final = self._run_epochs(
                self.model_, self.encoder_, frame, target, self.scaler_, config,
                monitor=None, epochs=best_epoch, early_stopping=False,
            )
        stage2_seconds, stage2_memory = timer.seconds, timer.report()

        self.model_.eval()
        self.diagnostics_ = self._diagnostics(frame)
        self.fit_outcome_ = TrainOutcome(
            requested={
                **config.requested(),
                "recipe": self.spec.as_dict(),
                "target": str(target),
                "stage1_epoch_budget": int(config.max_epochs),
                "stage2_epochs": int(best_epoch),
            },
            effective={
                **config.requested(),
                "stage1_effective_epochs": int(stage1["stopped_epoch"]),
                "stage2_effective_epochs": int(final["stopped_epoch"]),
                "n_features_numeric": int(n_num),
                "n_features_categorical": int(n_cat),
                "n_parameters": int(self.parameter_count()),
                "n_parameters_per_tree": round(
                    float(self.parameter_count()) / max(1, self.spec.total_trees), 4
                ),
                "total_trees": int(self.spec.total_trees),
                "depth": int(self.spec.depth),
                "routing": "soft_sigmoid_no_hard_argmax",
                "feature_selection": "differentiable_entmax15",
                "hard_argmax_used": False,
            },
            stop_reason=str(stage1["stop_reason"]),
            best_epoch=int(best_epoch),
            stopped_epoch=int(stage1["stopped_epoch"]),
            best_validation_wmape=float(stage1["best_validation_wmape"]),
            history=list(stage1["history"]),
            seconds=float(stage1_seconds + stage2_seconds),
            peak_memory={"stage1": stage1_memory, "stage2": stage2_memory},
            randomness={
                "training_seed": int(config.seed),
                "inner_split_seed": int(self.inner_seed),
                "inner_splits": int(self.n_inner_splits),
                "batch_order_rng": "numpy_default_rng",
                "initialisation": "torch_manual_seed",
            },
            fit_row_id_hash=fit_row_id_hash(frame["sample_id"].tolist()),
            fit_group_hash=group_hash(exclusion_group_keys(frame).tolist()),
            transform_hash=transform_hash({
                "target_scale": self.scaler_.metadata(),
                "encoder": self.encoder_.metadata(),
                "recipe": self.spec.as_dict(),
            }),
            extra={
                "line": "N",
                "adapter_note": "ADAPTED_NODE_STYLE_IMPLEMENTATION_NOT_PAPER_REPRODUCTION",
                "routing_diagnostics": dict(self.diagnostics_),
                "inner_uh": {
                    "n_u": int(len(u_frame)),
                    "n_h": int(len(h_frame)),
                    "inner_fold_hash": str(split["inner_fold_hash"]),
                },
                "training_volume": {
                    "n_train_rows": int(len(frame)),
                    "n_epochs_stage1": int(stage1["stopped_epoch"]),
                    "n_epochs_stage2": int(final["stopped_epoch"]),
                    "n_batches_stage1": int(stage1["stopped_epoch"]) * int(
                        math.ceil(len(u_frame) / config.batch_size)
                    ),
                    "n_batches_stage2": int(final["stopped_epoch"]) * int(
                        math.ceil(len(frame) / config.batch_size)
                    ),
                },
            },
        )
        return self

    def _encode(self, frame: pd.DataFrame, encoder):
        torch = _torch()
        x_num, x_cat = encode_frame(frame, encoder)
        return (
            torch.as_tensor(np.asarray(x_num), dtype=torch.float32),
            torch.as_tensor(np.asarray(x_cat), dtype=torch.float32),
        )

    def _run_epochs(
        self,
        net,
        encoder,
        fit_frame: pd.DataFrame,
        target: str,
        scaler: TargetScaler,
        config: TrainConfig,
        *,
        monitor,
        epochs: int,
        early_stopping: bool,
    ) -> dict[str, Any]:
        torch = _torch()
        torch.manual_seed(int(config.seed) % (2**32 - 1))
        rng = np.random.default_rng(int(config.seed))
        train_num, train_cat = self._encode(fit_frame, encoder)
        target_scaled = scaler.transform(np.asarray(fit_frame[target], dtype=np.float64))
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        monitor_state = None
        if monitor is not None:
            m_frame, _, _ = monitor
            m_num, m_cat = self._encode(m_frame, encoder)
            monitor_state = {
                "num": m_num,
                "cat": m_cat,
                "actual": np.asarray(m_frame[target], dtype=np.float64),
            }

        def epoch_step(_epoch: int, batch: np.ndarray) -> float:
            net.train()
            index = torch.as_tensor(np.asarray(batch), dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            prediction = net(train_num[index], train_cat[index])
            target_t = torch.as_tensor(target_scaled[batch], dtype=torch.float32)
            loss = criterion(prediction, target_t)
            if self.spec.selection_sparsity > 0.0:
                penalty = sum(
                    layer.selection().sum() for layer in net.layers
                ) / max(1, self.spec.total_trees)
                loss = loss + self.spec.selection_sparsity * penalty
            loss.backward()
            optimizer.step()
            if not math.isfinite(float(loss.detach().cpu().item())):
                raise FloatingPointError("N line produced a nonfinite training loss")
            return float(loss.detach().cpu().item())

        def epoch_score() -> float:
            if monitor_state is None:
                return float("nan")
            net.eval()
            with torch.no_grad():
                raw = net(monitor_state["num"], monitor_state["cat"])
                predicted = scaler.inverse(raw.detach().cpu().numpy().reshape(-1))
            return float(wmape(monitor_state["actual"], predicted))

        def state_getter():
            return deepcopy(net.state_dict())

        def state_setter(value):
            net.load_state_dict(value)

        if early_stopping:
            stopper, history = regression_epochs(
                n_rows=len(fit_frame), config=config, epoch_step=epoch_step,
                epoch_score=epoch_score, state_getter=state_getter,
                state_setter=state_setter, rng=rng,
            )
            return {
                "best_epoch": int(stopper.best_epoch),
                "stopped_epoch": int(stopper.stopped_epoch),
                "best_validation_wmape": float(stopper.best_value),
                "stop_reason": stopper.stop_reason(int(config.max_epochs)),
                "history": list(history),
            }
        history = []
        for epoch in range(1, int(epochs) + 1):
            order = rng.permutation(len(fit_frame))
            losses = []
            for start in range(0, len(order), int(config.batch_size)):
                losses.append(epoch_step(epoch, order[start:start + int(config.batch_size)]))
            history.append({
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)) if losses else math.inf,
                "validation_wmape": float("nan"),
            })
        return {
            "best_epoch": int(epochs),
            "stopped_epoch": int(epochs),
            "best_validation_wmape": float("nan"),
            "stop_reason": "determined_epoch_refit_no_early_stopping",
            "history": history,
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("NodeEnsembleRegressor must be fitted before prediction")
        torch = _torch()
        frame = frame.reset_index(drop=True)
        num, cat = self._encode(frame, self.encoder_)
        self.model_.eval()
        with torch.no_grad():
            raw = self.model_(num, cat)
        result = self.scaler_.inverse(raw.detach().cpu().numpy().reshape(-1))
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise ValueError("Invalid N-line prediction")
        return result.astype(float, copy=False)

    def predict_chunked(self, frame: pd.DataFrame, *, chunk_size: int) -> np.ndarray:
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        frame = frame.reset_index(drop=True)
        parts = [
            self.predict(frame.iloc[start:start + int(chunk_size)].reset_index(drop=True))
            for start in range(0, len(frame), int(chunk_size))
        ]
        if not parts:
            return np.zeros(0, dtype=float)
        return np.concatenate(parts)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model_.parameters()))

    def _diagnostics(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Routing entropy, leaf usage and parameter-update checks."""
        torch = _torch()
        num, cat = self._encode(frame, self.encoder_)
        self.model_.eval()
        entropies: list[float] = []
        used_fraction: list[float] = []
        selection_mass: list[float] = []
        with torch.no_grad():
            x = torch.cat([num, cat], dim=1)
            for layer in self.model_.layers:
                response = layer.routing(x)
                probabilities = layer.leaf_probabilities(response)
                # Binary routing entropy, normalised to [0, 1] by log(2).
                p = response.clamp(1e-9, 1 - 1e-9)
                entropy = -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2.0)
                entropies.append(float(entropy.mean().item()))
                mean_leaf = probabilities.mean(dim=0)              # (T, L)
                used = (mean_leaf > (1.0 / (2.0 * mean_leaf.shape[1]))).float().mean()
                used_fraction.append(float(used.item()))
                selection_mass.append(float(layer.selection().max(dim=1).values.mean().item()))
                x = torch.cat([x, layer(x)], dim=1)
        leaf_weights = torch.cat([layer.leaf_weights.detach().reshape(-1) for layer in self.model_.layers])
        return {
            "mean_routing_entropy_normalised": round(float(np.mean(entropies)), 6),
            "routing_entropy_per_layer": [round(v, 6) for v in entropies],
            "mean_used_leaf_fraction": round(float(np.mean(used_fraction)), 6),
            "used_leaf_fraction_per_layer": [round(v, 6) for v in used_fraction],
            "mean_max_selection_weight": round(float(np.mean(selection_mass)), 6),
            "leaf_weight_abs_mean": round(float(leaf_weights.abs().mean().item()), 8),
            "collapse_warning": bool(
                float(np.mean(entropies)) < 0.05 or float(np.mean(used_fraction)) < 0.25
            ),
            "note": (
                "Routing collapse is reported, not hidden: an ensemble whose trees all "
                "route identically is not treated as a successful tree ensemble."
            ),
        }
