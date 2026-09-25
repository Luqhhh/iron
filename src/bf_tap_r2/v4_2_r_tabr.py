"""V4.2 R line: retrieval-augmented tabular network (TabR-style).

Evidence basis: the TabR paper (https://arxiv.org/abs/2307.14338) and the
authors' implementation (https://github.com/yandex-research/tabular-dl-tabr).

This is an **adapted implementation, not a line-by-line reproduction**.  What is
kept is the core mechanism: the input representation, the retrieval metric and
the support-sample aggregation are learned jointly, instead of smoothing a fixed
raw-distance residual.  Exact distances in the learned representation replace
FAISS, and the neighbour set is selected over the full legal support with
symmetric tie handling.  No specific paper number is claimed to be reproduced.

Four recipes per target, all sharing one capacity:

===== =========================================== ==================================
ID    structure                                   question the recipe asks
===== =========================================== ==================================
R0    same-capacity FFN, retrieval disabled       is the retrieval path necessary
R1    learned retrieval, support features only     are retrieved features enough
R2    R1 plus open support labels                  does label memory add net gain
R3    R2 with eight-segment PLE numerics           is retrieval sensitive to encoding
===== =========================================== ==================================

Isolation rules enforced here and asserted by the tests:

* the outer evaluation fold never enters a support set;
* the inner early-stopping part ``H`` never enters the support set of the model
  that is early-stopped on it;
* a training query can never retrieve itself or a row with identical features;
* the support set is serialised with the model together with the source-ID and
  fit-row hashes.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    sha256_hex,
    transform_hash,
)
from .v4_2_train import NEURAL_LOSS, FitTimer, TrainConfig, TrainOutcome, regression_epochs

__all__ = ["R_RECIPES", "R_START", "TabRRetrievalRegressor", "TabRSpec", "recipe_spec"]

#: The four pre-registered R recipes.
R_RECIPES: dict[str, dict[str, Any]] = {
    "R0": {
        "recipe_id": "R0",
        "retrieval": False,
        "use_support_labels": False,
        "numeric_encoding": "raw",
        "role": "capacity_control_no_retrieval",
        "counts_as_retrieval_success": False,
    },
    "R1": {
        "recipe_id": "R1",
        "retrieval": True,
        "use_support_labels": False,
        "numeric_encoding": "raw",
        "role": "retrieved_features_only_labels_masked_from_training_start",
        "counts_as_retrieval_success": True,
    },
    "R2": {
        "recipe_id": "R2",
        "retrieval": True,
        "use_support_labels": True,
        "numeric_encoding": "raw",
        "role": "retrieved_features_plus_open_support_labels",
        "counts_as_retrieval_success": True,
    },
    "R3": {
        "recipe_id": "R3",
        "retrieval": True,
        "use_support_labels": True,
        "numeric_encoding": "ple",
        "role": "R2_with_eight_segment_ple",
        "counts_as_retrieval_success": True,
    },
}

#: Fixed starting point pre-registered by the task book.
R_START: dict[str, Any] = {
    "repr_width": 128,
    "n_blocks": 2,
    "n_neighbors": 32,
    "dropout": 0.10,
    "ple_segments": PLE_SEGMENTS,
}


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("The V4.2 R line requires torch") from exc
    return torch


def recipe_spec(recipe_id: str) -> dict[str, Any]:
    key = str(recipe_id).upper()
    if key not in R_RECIPES:
        raise ValueError(f"Unknown R recipe: {recipe_id!r}")
    return deepcopy(R_RECIPES[key])


class TabRSpec:
    """Validated R recipe specification; nothing is silently ignored."""

    def __init__(self, recipe_id: str, overrides: Mapping[str, Any] | None = None) -> None:
        payload = recipe_spec(recipe_id)
        merged = dict(R_START)
        unknown = sorted(set(overrides or {}) - set(R_START))
        if unknown:
            raise ValueError(f"Unknown R starting-point overrides: {unknown}")
        merged.update({str(k): v for k, v in (overrides or {}).items()})
        self.recipe_id = str(payload["recipe_id"])
        self.retrieval = bool(payload["retrieval"])
        self.use_support_labels = bool(payload["use_support_labels"])
        self.numeric_encoding = str(payload["numeric_encoding"])
        self.role = str(payload["role"])
        self.counts_as_retrieval_success = bool(payload["counts_as_retrieval_success"])
        self.repr_width = int(merged["repr_width"])
        self.n_blocks = int(merged["n_blocks"])
        self.n_neighbors = int(merged["n_neighbors"])
        self.dropout = float(merged["dropout"])
        self.ple_segments = int(merged["ple_segments"])
        if self.repr_width <= 0 or self.n_blocks <= 0 or self.n_neighbors <= 0:
            raise ValueError("Invalid R starting point")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Invalid R dropout")
        if not 1 <= self.ple_segments:
            raise ValueError("Invalid PLE segment count")
        if self.use_support_labels and not self.retrieval:
            raise ValueError("Open support labels require retrieval to be enabled")

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "retrieval": bool(self.retrieval),
            "use_support_labels": bool(self.use_support_labels),
            "numeric_encoding": str(self.numeric_encoding),
            "role": str(self.role),
            "counts_as_retrieval_success": bool(self.counts_as_retrieval_success),
            "repr_width": int(self.repr_width),
            "n_blocks": int(self.n_blocks),
            "n_neighbors": int(self.n_neighbors),
            "dropout": float(self.dropout),
            "ple_segments": int(self.ple_segments),
        }


def _make_network(spec: TabRSpec, n_num: int, n_cat: int):
    torch = _torch()

    class _TabRNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(spec.repr_width)
            blocks: list[Any] = []
            previous = int(n_num) + int(n_cat)
            for _ in range(int(spec.n_blocks)):
                blocks.extend([
                    torch.nn.Linear(previous, width),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(spec.dropout)),
                ])
                previous = width
            self.encoder = torch.nn.Sequential(*blocks)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(width * 2, width),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(spec.dropout)),
                torch.nn.Linear(width, 1),
            )
            if spec.retrieval:
                # Learnable retrieval temperature: part of the learned metric.
                self.log_scale = torch.nn.Parameter(
                    torch.tensor(math.log(math.e - 1.0), dtype=torch.float32)
                )

        def encode(self, x_num, x_cat):
            return self.encoder(torch.cat([x_num, x_cat], dim=1))

        def attention(self, q, k_emb, legal):
            """Symmetric top-k attention weights in the learned representation."""
            torch_mod = _torch()
            d2 = (q * q).sum(dim=1, keepdim=True) + (k_emb * k_emb).sum(dim=1)[None, :]
            d2 = d2 - 2.0 * (q @ k_emb.transpose(0, 1))
            # ``clamp_min`` keeps the square-root derivative finite at zero
            # distance: sqrt' is unbounded at exactly zero.
            distance = d2.clamp_min(1e-8).sqrt()
            inf = torch_mod.tensor(float("inf"), dtype=distance.dtype, device=distance.device)
            masked = torch_mod.where(legal, distance, inf)
            legal_count = legal.sum(dim=1)
            if bool((legal_count == 0).any()):
                raise ValueError(
                    "R line: a training query has no legal support row; V4.2 refuses to "
                    "silently include an excluded sample"
                )
            k = int(min(int(spec.n_neighbors), int(legal_count.min().item())))
            kth = torch_mod.topk(masked, k, dim=1, largest=False).values[:, -1]
            # Symmetric ties: every row tied with the k-th distance is admitted,
            # so the cut never favours an arbitrary sample.
            keep = (masked <= kth[:, None]) & legal
            scale = torch_mod.nn.functional.softplus(self.log_scale)
            # Illegal entries are replaced by a *finite* zero before scaling, so
            # masking them below cannot create a 0 * inf NaN gradient.
            finite_distance = torch_mod.where(legal, distance, torch_mod.zeros_like(distance))
            logits = (-scale) * finite_distance
            neg_inf = torch_mod.tensor(
                float("-inf"), dtype=distance.dtype, device=distance.device
            )
            logits = torch_mod.where(keep, logits, neg_inf)
            return torch_mod.softmax(logits, dim=1)

        def forward(self, x_num_q, x_cat_q, x_num_s, x_cat_s, labels, legal):
            q = self.encode(x_num_q, x_cat_q)
            if spec.retrieval:
                k_emb = self.encode(x_num_s, x_cat_s)
                weights = self.attention(q, k_emb, legal)
                agg = weights @ k_emb
                y_term = weights @ labels if labels is not None else None
            else:
                agg = torch.zeros_like(q)
                weights = None
                y_term = None
            base = self.head(torch.cat([q, agg], dim=1)).reshape(-1)
            if spec.use_support_labels:
                if y_term is None:
                    raise ValueError("R recipe requires open support labels but none were supplied")
                return base + y_term
            return base

    return _TabRNet()


class TabRRetrievalRegressor:
    """One TabR-style recipe fitted on a caller-supplied training part."""

    def __init__(
        self,
        recipe_id: str,
        *,
        overrides: Mapping[str, Any] | None = None,
        n_inner_splits: int = 5,
        inner_seed: int = 42,
        torch_threads: int | None = None,
    ) -> None:
        self.spec = TabRSpec(recipe_id, overrides)
        self.n_inner_splits = int(n_inner_splits)
        self.inner_seed = int(inner_seed)
        self.torch_threads = None if torch_threads is None else int(torch_threads)
        self.recipe_id = self.spec.recipe_id

    # -- support plumbing -------------------------------------------------
    def _prepare_support(self, frame: pd.DataFrame, target: str, encoder) -> dict[str, Any]:
        x_num, x_cat = encode_frame(frame, encoder)
        keys = exclusion_group_keys(frame)
        return {
            "x_num": np.asarray(x_num, dtype=np.float32),
            "x_cat": np.asarray(x_cat, dtype=np.float32),
            "y_raw": np.asarray(frame[target], dtype=np.float64),
            "sample_ids": [str(v) for v in frame["sample_id"].tolist()],
            "group_keys": keys,
            "source_id_hash": fit_row_id_hash(frame["sample_id"].tolist()),
            "fit_row_hash": group_hash(keys.tolist()),
            "n_duplicate_rows": int(len(keys) - len(set(keys.tolist()))),
        }

    def _support_tensors(self):
        torch = _torch()
        support, scaler = self.support_, self.scaler_
        labels = None
        y_scaled = scaler.transform(np.asarray(support["y_raw"], dtype=np.float64))
        if self.spec.use_support_labels:
            labels = torch.as_tensor(np.asarray(y_scaled), dtype=torch.float32)
        return (
            torch.as_tensor(np.asarray(support["x_num"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(support["x_cat"]), dtype=torch.float32),
            labels,
        )

    @staticmethod
    def _legal_mask(query_keys, support_keys) -> np.ndarray:
        """``legal[q, s]`` is True when support row ``s`` may be retrieved."""
        q = np.asarray(query_keys, dtype=object)
        s = np.asarray(support_keys, dtype=object)
        if not len(q) or not len(s):
            return np.ones((len(q), len(s)), dtype=bool)
        return q[:, None] != s[None, :]

    # -- fit --------------------------------------------------------------
    def fit(
        self,
        frame: pd.DataFrame,
        target: str,
        *,
        train_config: Mapping[str, Any] | None = None,
    ) -> "TabRRetrievalRegressor":
        torch = _torch()
        if self.torch_threads is not None:
            torch.set_num_threads(max(1, int(self.torch_threads)))
        config = TrainConfig.from_mapping({**dict(train_config or {}), "loss": NEURAL_LOSS})
        frame = frame.reset_index(drop=True)
        y_raw = np.asarray(frame[target], dtype=np.float64)
        if not np.isfinite(y_raw).all():
            raise ValueError("R line requires finite training targets")
        self.target_ = str(target)

        # ---- stage 1: search/early stopping on the inner U -> H split ----
        split = inner_uh_split(frame, n_splits=self.n_inner_splits, seed=self.inner_seed)
        u_frame = frame.iloc[split["u_index"]].reset_index(drop=True)
        h_frame = frame.iloc[split["h_index"]].reset_index(drop=True)
        if len(u_frame) == 0 or len(h_frame) == 0:
            raise ValueError("R line requires a nonempty U/H split")

        self.scaler_ = TargetScaler.fit(np.asarray(u_frame[target], dtype=np.float64))
        u_encoder = build_encoder(self.spec.numeric_encoding, n_bins=self.spec.ple_segments).fit(u_frame)
        u_support = self._prepare_support(u_frame, target, encoder=u_encoder)
        n_num = int(np.asarray(u_support["x_num"]).shape[1])
        n_cat = int(np.asarray(u_support["x_cat"]).shape[1])

        with FitTimer() as timer:
            stage1_net = _make_network(self.spec, n_num, n_cat)
            stage1 = self._run_epochs(
                stage1_net,
                encoder=u_encoder,
                fit_frame=u_frame,
                target=target,
                support=u_support,
                scaler=self.scaler_,
                config=config,
                monitor=(h_frame, u_support, u_encoder),
                epochs=int(config.max_epochs),
                early_stopping=True,
            )
        stage1_seconds, stage1_memory = timer.seconds, timer.report()
        best_epoch = max(1, int(stage1["best_epoch"]))

        # ---- stage 2: refit on the complete training part T --------------
        # The determined epoch count is reused.  V is never consulted and no
        # new early-stopping part is invented.
        self.encoder_ = build_encoder(
            self.spec.numeric_encoding, n_bins=self.spec.ple_segments
        ).fit(frame)
        self.scaler_ = TargetScaler.fit(y_raw)
        support = self._prepare_support(frame, target, encoder=self.encoder_)
        with FitTimer() as timer:
            self.model_ = _make_network(self.spec, n_num, n_cat)
            final = self._run_epochs(
                self.model_,
                encoder=self.encoder_,
                fit_frame=frame,
                target=target,
                support=support,
                scaler=self.scaler_,
                config=config,
                monitor=None,
                epochs=best_epoch,
                early_stopping=False,
            )
        stage2_seconds, stage2_memory = timer.seconds, timer.report()

        self.support_ = support
        self._support_tensor_cache = None
        self.model_.eval()
        self.retrieval_audit_ = self._audit_retrieval()
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
                "n_support": int(len(support["sample_ids"])),
                "n_neighbors_requested": int(self.spec.n_neighbors),
                "distance": "exact_euclidean_in_learned_representation",
                "tie_handling": "symmetric_admit_all_ties_at_kth",
                "faiss_used": False,
                "reference_neighbour_crosscheck": "tests/test_round2_v4_2.py brute-force comparison",
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
            fit_row_id_hash=str(support["source_id_hash"]),
            fit_group_hash=str(support["fit_row_hash"]),
            transform_hash=transform_hash({
                "target_scale": self.scaler_.metadata(),
                "encoder": self.encoder_.metadata(),
                "recipe": self.spec.as_dict(),
            }),
            extra={
                "line": "R",
                "adapter_note": "ADAPTED_TABR_STYLE_IMPLEMENTATION_NOT_PAPER_REPRODUCTION",
                "retrieval_audit": dict(self.retrieval_audit_),
                "inner_uh": {
                    "n_u": int(len(u_frame)),
                    "n_h": int(len(h_frame)),
                    "inner_fold_hash": str(split["inner_fold_hash"]),
                    "inner_group_hash": str(split["group_hash"]),
                },
            },
        )
        return self

    def _run_epochs(
        self,
        net,
        *,
        encoder,
        fit_frame: pd.DataFrame,
        target: str,
        support: Mapping[str, Any],
        scaler: TargetScaler,
        config: TrainConfig,
        monitor,
        epochs: int,
        early_stopping: bool,
    ) -> dict[str, Any]:
        torch = _torch()
        torch.manual_seed(int(config.seed) % (2**32 - 1))
        rng = np.random.default_rng(int(config.seed))
        x_num_q, x_cat_q = encode_frame(fit_frame, encoder)
        s_num, s_cat, s_labels = self._tensors_for(support, scaler)
        support_keys = np.asarray(support["group_keys"], dtype=object)
        legal = torch.as_tensor(
            self._legal_mask(support_keys, support_keys), dtype=torch.bool
        )
        q_num_t = torch.as_tensor(np.asarray(x_num_q), dtype=torch.float32)
        q_cat_t = torch.as_tensor(np.asarray(x_cat_q), dtype=torch.float32)
        target_scaled = scaler.transform(np.asarray(fit_frame[target], dtype=np.float64))
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.AdamW(
            [p for p in net.parameters() if p.requires_grad],
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )

        monitor_state = None
        if monitor is not None:
            m_frame, m_support, m_encoder = monitor
            m_num_q, m_cat_q = encode_frame(m_frame, m_encoder)
            m_num_s, m_cat_s, m_labels = self._tensors_for(m_support, scaler)
            # H queries retrieve the U support only: the early-stopping part is
            # never part of the support set of the model it early-stops.
            monitor_state = {
                "num_q": torch.as_tensor(np.asarray(m_num_q), dtype=torch.float32),
                "cat_q": torch.as_tensor(np.asarray(m_cat_q), dtype=torch.float32),
                "num_s": m_num_s,
                "cat_s": m_cat_s,
                "labels": m_labels,
                "legal": torch.as_tensor(
                    self._legal_mask(
                        exclusion_group_keys(m_frame),
                        np.asarray(m_support["group_keys"], dtype=object),
                    ),
                    dtype=torch.bool,
                ),
                "actual": np.asarray(m_frame[target], dtype=np.float64),
            }

        def epoch_step(_epoch: int, batch: np.ndarray) -> float:
            net.train()
            index = torch.as_tensor(np.asarray(batch), dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            prediction = net(
                q_num_t[index], q_cat_t[index], s_num, s_cat, s_labels, legal[index]
            )
            target_t = torch.as_tensor(target_scaled[batch], dtype=torch.float32)
            loss = criterion(prediction, target_t)
            loss.backward()
            optimizer.step()
            return float(loss.detach().cpu().item())

        def epoch_score() -> float:
            if monitor_state is None:
                return float("nan")
            net.eval()
            with torch.no_grad():
                raw = net(
                    monitor_state["num_q"], monitor_state["cat_q"], monitor_state["num_s"],
                    monitor_state["cat_s"], monitor_state["labels"], monitor_state["legal"],
                )
                predicted = scaler.inverse(raw.detach().cpu().numpy().reshape(-1))
            return float(wmape(monitor_state["actual"], predicted))

        def state_getter():
            return deepcopy(net.state_dict())

        def state_setter(value):
            net.load_state_dict(value)

        if early_stopping:
            stopper, history = regression_epochs(
                n_rows=len(fit_frame),
                config=config,
                epoch_step=epoch_step,
                epoch_score=epoch_score,
                state_getter=state_getter,
                state_setter=state_setter,
                rng=rng,
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

    # -- inference --------------------------------------------------------
    def _tensors_for(self, support: Mapping[str, Any], scaler: TargetScaler):
        torch = _torch()
        y_scaled = scaler.transform(np.asarray(support["y_raw"], dtype=np.float64))
        labels = (
            torch.as_tensor(np.asarray(y_scaled), dtype=torch.float32)
            if self.spec.use_support_labels
            else None
        )
        return (
            torch.as_tensor(np.asarray(support["x_num"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(support["x_cat"]), dtype=torch.float32),
            labels,
        )

    def _cached_support_tensors(self):
        if getattr(self, "_support_tensor_cache", None) is None:
            self._support_tensor_cache = self._tensors_for(self.support_, self.scaler_)
        return self._support_tensor_cache

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        query_group_keys: Sequence[str] | np.ndarray | None = None,
    ) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TabRRetrievalRegressor must be fitted before prediction")
        torch = _torch()
        frame = frame.reset_index(drop=True)
        x_num, x_cat = encode_frame(frame, self.encoder_)
        num_q = torch.as_tensor(np.asarray(x_num), dtype=torch.float32)
        cat_q = torch.as_tensor(np.asarray(x_cat), dtype=torch.float32)
        s_num, s_cat, s_labels = self._cached_support_tensors()
        if query_group_keys is None:
            # External query: no row of it is inside the support set.
            keys = np.asarray([f"external-{i}" for i in range(len(frame))], dtype=object)
        else:
            keys = np.asarray(query_group_keys, dtype=object).reshape(-1)
            if len(keys) != len(frame):
                raise ValueError("query_group_keys must align with the query frame")
        legal = torch.as_tensor(
            self._legal_mask(keys, np.asarray(self.support_["group_keys"], dtype=object)),
            dtype=torch.bool,
        )
        self.model_.eval()
        with torch.no_grad():
            raw = self.model_(num_q, cat_q, s_num, s_cat, s_labels, legal)
        result = self.scaler_.inverse(raw.detach().cpu().numpy().reshape(-1))
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise ValueError("Invalid R-line prediction")
        return result.astype(float, copy=False)

    def predict_chunked(
        self,
        frame: pd.DataFrame,
        *,
        chunk_size: int,
        query_group_keys: Sequence[str] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Chunk-invariant inference, used by the cold-consistency audit."""
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        frame = frame.reset_index(drop=True)
        keys = None if query_group_keys is None else np.asarray(query_group_keys, dtype=object)
        parts: list[np.ndarray] = []
        for start in range(0, len(frame), int(chunk_size)):
            stop = min(len(frame), start + int(chunk_size))
            parts.append(self.predict(
                frame.iloc[start:stop].reset_index(drop=True),
                query_group_keys=None if keys is None else keys[start:stop],
            ))
        if not parts:
            return np.zeros(0, dtype=float)
        return np.concatenate(parts)

    def attention_weights(
        self, frame: pd.DataFrame, *, chunk_size: int = 256
    ) -> np.ndarray:
        """Attention weights over the stored support (retrieval recipes only)."""
        if not self.spec.retrieval:
            raise ValueError("R0 disables retrieval, so it has no attention weights")
        torch = _torch()
        frame = frame.reset_index(drop=True)
        x_num, x_cat = encode_frame(frame, self.encoder_)
        s_num, s_cat, _ = self._cached_support_tensors()
        support_keys = np.asarray(self.support_["group_keys"], dtype=object)
        keys = exclusion_group_keys(frame)
        out = np.zeros((len(frame), len(support_keys)), dtype=np.float64)
        self.model_.eval()
        for start in range(0, len(frame), int(chunk_size)):
            stop = min(len(frame), start + int(chunk_size))
            num_q = torch.as_tensor(np.asarray(x_num[start:stop]), dtype=torch.float32)
            cat_q = torch.as_tensor(np.asarray(x_cat[start:stop]), dtype=torch.float32)
            legal = torch.as_tensor(
                self._legal_mask(keys[start:stop], support_keys), dtype=torch.bool
            )
            with torch.no_grad():
                weights = self.model_.attention(
                    self.model_.encode(num_q, cat_q), self.model_.encode(s_num, s_cat), legal
                )
            out[start:stop] = weights.detach().cpu().numpy()
        return out

    # -- audit and serialisation -----------------------------------------
    def _audit_retrieval(self) -> dict[str, Any]:
        """Verify no training query retrieves itself or an identical duplicate."""
        keys = np.asarray(self.support_["group_keys"], dtype=object)
        n = len(keys)
        unique = len(set(keys.tolist()))
        audit: dict[str, Any] = {
            "exclusion_key": "21_numeric_features_plus_spout_no_exact_bytes",
            "n_support": int(n),
            "distinct_groups": int(unique),
            "duplicate_rows_present": int(n - unique),
            "retrieval_enabled": bool(self.spec.retrieval),
        }
        if not self.spec.retrieval:
            audit.update({
                "self_retrieval_count": 0,
                "duplicate_group_retrieval_count": 0,
                "max_illegal_attention_weight": 0.0,
                "retrieval_gradient_nonzero": False,
                "note": "R0 is the capacity control with retrieval switched off.",
            })
            return audit
        # Weight audit over the stored support itself: query rows are the
        # support rows, so the exclusion rule is exercised directly.
        torch = _torch()
        s_num, s_cat, _ = self._cached_support_tensors()
        illegal_weight = 0.0
        violations = 0
        chunk = 256
        self.model_.eval()
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            legal = torch.as_tensor(self._legal_mask(keys[start:stop], keys), dtype=torch.bool)
            with torch.no_grad():
                weights = self.model_.attention(
                    self.model_.encode(s_num[start:stop], s_cat[start:stop]),
                    self.model_.encode(s_num, s_cat),
                    legal,
                ).detach().cpu().numpy()
            same = keys[start:stop][:, None] == keys[None, :]
            illegal_weight = max(illegal_weight, float(np.abs(weights[same]).max()) if same.any() else 0.0)
            violations += int((np.abs(weights[same]) > 0.0).sum())
        # Gradient check: the learned metric must actually receive gradient.
        net = self.model_
        net.train()
        for p in net.parameters():
            p.grad = None
        take = min(8, n)
        legal = torch.as_tensor(self._legal_mask(keys[:take], keys), dtype=torch.bool)
        out = net(s_num[:take], s_cat[:take], s_num, s_cat, self._cached_support_tensors()[2], legal)
        out.sum().backward()
        scale_grad = None if net.log_scale.grad is None else float(net.log_scale.grad.abs().sum())
        encoder_grad = max(
            (float(p.grad.abs().sum()) for p in net.encoder.parameters() if p.grad is not None),
            default=0.0,
        )
        for p in net.parameters():
            p.grad = None
        net.eval()
        audit.update({
            "self_retrieval_count": int(violations),
            "duplicate_group_retrieval_count": int(violations),
            "max_illegal_attention_weight": round(float(illegal_weight), 12),
            "retrieval_gradient_nonzero": bool(
                (scale_grad or 0.0) > 0.0 and encoder_grad > 0.0
            ),
            "log_scale_gradient_abs_sum": scale_grad,
            "encoder_gradient_abs_sum": round(float(encoder_grad), 8),
        })
        if violations != 0:
            raise AssertionError(
                "R line retrieved an excluded training row; the V4.2 self/duplicate ban is violated"
            )
        return audit

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model_.parameters()))

    def support_state(self) -> dict[str, Any]:
        if not hasattr(self, "support_"):
            raise RuntimeError("No support set; the model is not fitted")
        support = self.support_
        return {
            "source_id_hash": str(support["source_id_hash"]),
            "fit_row_hash": str(support["fit_row_hash"]),
            "n_support": int(len(support["sample_ids"])),
            "support_sample_ids": list(support["sample_ids"]),
            "support_group_keys": [str(v) for v in support["group_keys"]],
            "support_y_raw": [float(v) for v in support["y_raw"]],
            "support_x_num": np.asarray(support["x_num"], dtype=np.float32),
            "support_x_cat": np.asarray(support["x_cat"], dtype=np.float32),
            "note": (
                "Private in-model training memory. It is part of the model state. Cold "
                "inference may read this memory but must not reopen the original training "
                "CSV, and it must never be published to the code repository."
            ),
        }

    def save_support(self, path: Path | str) -> dict[str, Any]:
        target = Path(path)
        if "local" not in target.resolve().parts:
            raise ValueError("Support sets are private and may only be written under local/")
        state = self.support_state()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            x_num=np.asarray(state["support_x_num"], dtype=np.float32),
            x_cat=np.asarray(state["support_x_cat"], dtype=np.float32),
            y_raw=np.asarray(state["support_y_raw"], dtype=np.float64),
            group_keys=np.asarray(state["support_group_keys"], dtype=object),
            sample_ids=np.asarray(state["support_sample_ids"], dtype=object),
        )
        meta = {
            "source_id_hash": state["source_id_hash"],
            "fit_row_hash": state["fit_row_hash"],
            "n_support": state["n_support"],
            "recipe": self.spec.as_dict(),
            "target": str(self.target_),
            "file_sha256": sha256_hex(target.read_bytes()),
            "note": state["note"],
        }
        target.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta
