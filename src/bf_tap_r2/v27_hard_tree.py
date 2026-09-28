"""V27 scratch-trained hard trees adapted from GRANDE's MIT-licensed core.

Upstream: s-marton/GRANDE, commit 07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb.
Copyright (c) 2023 Sascha Marton. See docs/round2_v27_hard_tree/GRANDE_LICENSE.txt.
Only finite-input regression without author embeddings/subsampling is supported.
"""
from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class HardTreeEnsemble(nn.Module):
    """Hard forward routing, soft straight-through derivatives, paired gating."""

    def __init__(self, n_features: int, arm: str, settings: dict):
        super().__init__()
        if arm not in {"GLOBAL", "INSTANCE"}:
            raise ValueError("Unknown weight arm")
        self.arm = arm
        self.n_estimators = int(settings["n_estimators"])
        self.depth = int(settings["depth"])
        self.dropout = float(settings["dropout"])
        if n_features < 1 or self.n_estimators < 2 or self.depth < 1:
            raise ValueError("Invalid hard-tree shape")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")
        seed = int(settings.get("random_seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        fraction = float(settings["selected_variables"])
        count = min(int(fraction), n_features) if fraction > 1 else min(
            n_features, max(10, min(50, int(n_features * fraction))))
        features = torch.stack([torch.tensor(np.random.choice(
            n_features, size=count, replace=False), dtype=torch.long)
            for _ in range(self.n_estimators)])
        self.register_buffer("features_by_estimator", features)
        identifiers, nodes = [], []
        for leaf in range(2 ** self.depth):
            for d in range(1, self.depth + 1):
                identifiers.append((leaf // (2 ** (self.depth-d))) % 2)
                nodes.append(2 ** (d-1) + leaf // (2 ** (self.depth-(d-1))) - 1)
        self.register_buffer("path_identifier_list", torch.tensor(
            np.reshape(identifiers, (-1, self.depth)), dtype=torch.long))
        self.register_buffer("internal_node_index_list", torch.tensor(
            np.reshape(nodes, (-1, self.depth)), dtype=torch.long))
        # Author resets RNG after feature sampling, before all four parameter draws.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        split_shape = (self.n_estimators, 2 ** self.depth-1, count)
        leaf_shape = (self.n_estimators, 2 ** self.depth)
        for name, shape in [
            ("split_values", split_shape), ("split_index_array", split_shape),
            ("estimator_weights", leaf_shape), ("leaf_classes_array", leaf_shape)
        ]:
            param = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
            nn.init.normal_(param, mean=0.0, std=0.05)
            self.register_parameter(name, param)
        with torch.no_grad():
            self.estimator_weights.zero_()

    def routing_weights(self, inputs: torch.Tensor):
        if inputs.ndim != 2 or not torch.isfinite(inputs).all():
            raise ValueError("Finite two-dimensional inputs required")
        selected = inputs[:, self.features_by_estimator]
        indices = F.softmax(self.split_index_array, dim=-1)
        one_hot = F.one_hot(indices.argmax(dim=-1), num_classes=indices.shape[-1])
        indices = indices - (indices-one_hot).detach()
        threshold = torch.einsum("ein,ein->ei", self.split_values, indices)
        values = torch.einsum("ben,ein->bei", selected, indices)
        node = (F.softsign(threshold-values) + 1) / 2
        node = node - (node-torch.round(node)).detach()
        left = node[..., self.internal_node_index_list]
        right = 1.0-left
        p = torch.prod((1-self.path_identifier_list)*left
                       + self.path_identifier_list*right, dim=3)
        if self.arm == "INSTANCE":
            logits = torch.einsum("el,bel->be", self.estimator_weights, p)
        else:
            logits = self.estimator_weights.mean(dim=1).unsqueeze(0).expand(
                inputs.shape[0], -1)
        weights = F.softmax(logits, dim=-1)
        if self.training and self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout, training=True)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return p, weights

    def forward(self, inputs: torch.Tensor):
        p, weights = self.routing_weights(inputs)
        weighted = torch.einsum("bel,be->bel", p, weights)
        layer = torch.einsum("el,bel->be", self.leaf_classes_array, weighted)
        return torch.einsum("be->b", layer)
