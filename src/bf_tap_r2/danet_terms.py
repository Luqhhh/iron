"""DANet abstract-layer equations and inference-only reparameterization.

Mechanism: Chen et al., AAAI 2022; author commit b007c57121ec9082f6ef19ec7465d9df70767c26.
This independent implementation preserves row-local masks, gated branches,
ghost normalization and branch summation. No data loading or fitting runner.
See docs/danet_abstract/SOURCE_AUDIT.md and licenses/DANet-MIT.txt.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class _Entmax15(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        z=(logits-logits.amax(dim=-1,keepdim=True))/2
        descending=z.sort(dim=-1,descending=True).values
        counts=torch.arange(1,z.shape[-1]+1,dtype=z.dtype,device=z.device)
        mean=descending.cumsum(-1)/counts
        second=descending.square().cumsum(-1)/counts
        spread=counts*(second-mean.square())
        threshold=mean-((1-spread)/counts).clamp_min(0).sqrt()
        support=(threshold<=descending).sum(-1,keepdim=True)
        tau=threshold.gather(-1,support-1)
        answer=(z-tau).clamp_min(0).square()
        ctx.save_for_backward(answer)
        return answer

    @staticmethod
    def backward(ctx, output_gradient):
        probability,=ctx.saved_tensors
        root=probability.sqrt()
        weighted=output_gradient*root
        center=weighted.sum(-1,keepdim=True)/root.sum(-1,keepdim=True)
        return weighted-center*root


def entmax15(logits):
    """Exact alpha=1.5 simplex mapping with its support-aware Jacobian."""
    if (not isinstance(logits,torch.Tensor) or not logits.is_floating_point()
            or logits.ndim<1 or not logits.numel() or not torch.isfinite(logits).all()):
        raise ValueError('Nonempty finite floating logits required')
    return _Entmax15.apply(logits)


class GhostNormalization(nn.Module):
    """Train-only sequential ghost batches; evaluation uses stored statistics."""
    def __init__(self,channels,virtual_batch_size=256):
        super().__init__()
        if type(channels) is not int or channels<1 or type(virtual_batch_size) is not int or virtual_batch_size<2:
            raise ValueError('Positive channels and ghost size at least two required')
        self.virtual_batch_size=virtual_batch_size
        self.bn=nn.BatchNorm1d(channels)

    def forward(self,values):
        if values.ndim!=2 or not len(values) or values.shape[1]!=self.bn.num_features:
            raise ValueError('Aligned nonempty channel matrix required')
        if not self.training:return self.bn(values)
        pieces=values.chunk(math.ceil(len(values)/self.virtual_batch_size),dim=0)
        # Refuse before any partial normalization-statistic update.
        if any(len(part)<2 for part in pieces):raise ValueError('Singleton ghost training batch')
        return torch.cat([self.bn(part) for part in pieces],dim=0)


class AbstractLayer(nn.Module):
    """Identical row-wise groups followed by sigmoid-gated ReLU branches."""
    def __init__(self,input_dim,output_dim,groups=5,virtual_batch_size=256,*,learn_masks=True,bias=True):
        super().__init__()
        if (any(type(v) is not int or v<1 for v in (input_dim,output_dim,groups))
                or type(learn_masks) is not bool or type(bias) is not bool):
            raise ValueError('Valid abstract-layer dimensions/control required')
        self.input_dim,self.output_dim,self.groups=input_dim,output_dim,groups
        self.logits=nn.Parameter(torch.rand(groups,input_dim),requires_grad=learn_masks)
        self.projection=nn.Conv1d(input_dim*groups,2*output_dim*groups,1,groups=groups,bias=bias)
        gain=math.sqrt((input_dim*groups+2*output_dim*groups)/math.sqrt(input_dim*groups))
        nn.init.xavier_normal_(self.projection.weight,gain=gain)
        self.normalization=GhostNormalization(2*output_dim*groups,virtual_batch_size)

    def forward(self,features):
        if (features.ndim!=2 or not len(features) or features.shape[1]!=self.input_dim
                or not features.is_floating_point() or not torch.isfinite(features).all()):
            raise ValueError('Finite aligned row-local features required')
        masked=features[:,None,:]*entmax15(self.logits)[None,:,:]
        projected=self.projection(masked.reshape(len(features),-1,1)).squeeze(-1)
        branches=self.normalization(projected).reshape(len(features),self.groups,2*self.output_dim)
        gate,value=branches.split(self.output_dim,dim=-1)
        return F.relu(gate.sigmoid()*value).sum(dim=1)

    @torch.no_grad()
    def folded(self):
        if self.training:raise ValueError('Only stored evaluation statistics can be folded')
        batch=self.normalization.bn
        weight=self.projection.weight.squeeze(-1).reshape(self.groups,2*self.output_dim,self.input_dim)
        bias=self.projection.bias
        bias=torch.zeros_like(batch.running_mean) if bias is None else bias
        scale=batch.weight/(batch.running_var+batch.eps).sqrt()
        folded_weight=weight*entmax15(self.logits)[:,None,:]*scale.reshape(self.groups,2*self.output_dim,1)
        folded_bias=((bias-batch.running_mean)*scale+batch.bias).reshape(self.groups,2*self.output_dim)
        return FoldedAbstractLayer(folded_weight,folded_bias)


class FoldedAbstractLayer(nn.Module):
    """Fixed inference arrays; masks and normalization merged into each branch."""
    def __init__(self,weight,bias):
        super().__init__()
        if (weight.ndim!=3 or bias.shape!=weight.shape[:2] or weight.shape[1]%2
                or not weight.numel() or not torch.isfinite(weight).all() or not torch.isfinite(bias).all()):
            raise ValueError('Valid finite folded branch arrays required')
        self.register_buffer('weight',weight.detach().clone())
        self.register_buffer('bias',bias.detach().clone())
        self.training=False

    def train(self,mode=True):
        if mode:raise ValueError('Folded abstract layer is inference only')
        return super().train(False)

    def forward(self,features):
        if (features.ndim!=2 or not len(features) or features.shape[1]!=self.weight.shape[-1]
                or not torch.isfinite(features).all()):raise ValueError('Aligned finite folded features required')
        branches=F.linear(features,self.weight.flatten(0,1),self.bias.flatten()).reshape(len(features),*self.bias.shape)
        gate,value=branches.chunk(2,dim=-1)
        return F.relu(gate.sigmoid()*value).sum(1)
