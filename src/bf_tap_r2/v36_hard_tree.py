"""Equivalent hard-tree routing without expanding every root-to-leaf path.

Adapted from the MIT GRANDE core, copyright (c) 2023 Sascha Marton.
See docs/round2_v36/GRANDE_LICENSE.txt and v36_hard_tree_reference.py.
"""
import torch
from torch.nn import functional as F

from .v36_hard_tree_reference import HardTreeEnsemble as ReferenceEnsemble


class HardTreeEnsemble(ReferenceEnsemble):
    def routing_weights(self, inputs):
        if inputs.ndim != 2 or not torch.isfinite(inputs).all():
            raise ValueError("Finite two-dimensional inputs required")
        selected = inputs[:, self.features_by_estimator]
        indices = F.softmax(self.split_index_array, dim=-1)
        one_hot = F.one_hot(indices.argmax(dim=-1), num_classes=indices.shape[-1])
        indices = indices - (indices-one_hot).detach()
        threshold = torch.einsum("ein,ein->ei", self.split_values, indices)
        values = torch.einsum("ben,ein->bei", selected, indices)
        node = (F.softsign(threshold-values)+1)/2
        node = node-(node-torch.round(node)).detach()
        # Each parent path probability is reused for its two children.
        # Stacking left then right preserves the author's binary leaf order.
        p = torch.ones((*node.shape[:2],1),dtype=node.dtype,device=node.device)
        for depth in range(self.depth):
            level = node[...,2**depth-1:2**(depth+1)-1]
            p = torch.stack((p*level,p*(1-level)),dim=-1).flatten(-2)
        if self.arm == "INSTANCE":
            logits = torch.einsum("el,bel->be",self.estimator_weights,p)
        else:
            logits = self.estimator_weights.mean(dim=1).unsqueeze(0).expand(inputs.shape[0],-1)
        weights = F.softmax(logits,dim=-1)
        if self.training and self.dropout > 0:
            weights = F.dropout(weights,p=self.dropout,training=True)
            weights = weights/weights.sum(dim=1,keepdim=True).clamp_min(1e-8)
        return p,weights
