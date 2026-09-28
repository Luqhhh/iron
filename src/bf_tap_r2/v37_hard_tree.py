"""Contract hard-tree leaf responses before normalizing tree weights.

Same MIT-attributed GRANDE model as V36; see its reference and license.
"""
import torch
from torch.nn import functional as F

from .v36_hard_tree import HardTreeEnsemble as RoutingEnsemble


def contract_leaves(node, leaves, depth):
    """Bottom-up linear contraction with unchanged straight-through nodes."""
    values = leaves.unsqueeze(0)
    for level in reversed(range(depth)):
        selector = node[..., 2**level-1:2**(level+1)-1]
        values = selector*values[..., 0::2] + (1-selector)*values[..., 1::2]
    return values.squeeze(-1)


class HardTreeEnsemble(RoutingEnsemble):
    def forward(self, inputs):
        if inputs.ndim != 2 or not torch.isfinite(inputs).all():
            raise ValueError("Finite two-dimensional inputs required")
        selected = inputs[:, self.features_by_estimator]
        indices = F.softmax(self.split_index_array, dim=-1)
        one_hot = F.one_hot(indices.argmax(dim=-1), num_classes=indices.shape[-1])
        indices = indices-(indices-one_hot).detach()
        threshold = torch.einsum("ein,ein->ei", self.split_values, indices)
        values = torch.einsum("ben,ein->bei", selected, indices)
        node = (F.softsign(threshold-values)+1)/2
        node = node-(node-torch.round(node)).detach()
        response = contract_leaves(node, self.leaf_classes_array, self.depth)
        if self.arm == "INSTANCE":
            logits = contract_leaves(node, self.estimator_weights, self.depth)
        else:
            logits = self.estimator_weights.mean(dim=1).unsqueeze(0).expand(inputs.shape[0], -1)
        weights = F.softmax(logits, dim=-1)
        if self.training and self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout, training=True)
            weights = weights/weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("be,be->b", response, weights)
