"""MIT-adapted efficient-kan core, pinned in the V28 specification.

Copyright (c) 2024 Huanqi Cao. See docs/round2_v28_edge_kan/KAN_LICENSE.txt.
Only fixed-grid forwarding is used; no grid updates or entropy penalty.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SplineLinear(nn.Module):
    def __init__(self, in_features, out_features, arm):
        super().__init__()
        if arm not in ('EDGE', 'SHARED'):
            raise ValueError('Unknown V28 arm')
        self.in_features, self.out_features, self.arm = in_features, out_features, arm
        self.register_buffer('grid', (torch.arange(-3, 9) * .4 - 1).expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, 8))
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = (torch.rand(6, in_features, out_features) - .5) * .1 / 5
            self.spline_weight.copy_(self.curve2coeff(self.grid.T[3:-3], noise))
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError('Invalid spline input shape')
        grid, x = self.grid, x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, 4):
            bases = ((x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * bases[:, :, :-1]
                     + (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:-k]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        solution = torch.linalg.lstsq(self.b_splines(x).transpose(0, 1), y.transpose(0, 1)).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x):
        coeff = self.spline_weight
        if self.arm == 'SHARED':
            coeff = coeff.mean(0, keepdim=True).expand_as(coeff)
        coeff = coeff * self.spline_scaler.unsqueeze(-1)
        return F.linear(F.silu(x), self.base_weight) + F.linear(
            self.b_splines(x).reshape(len(x), -1), coeff.reshape(self.out_features, -1))


class EdgeKAN(nn.Module):
    def __init__(self, input_count, arm):
        super().__init__()
        torch.manual_seed(42)
        widths = (input_count, 64, 64, 1)
        self.layers = nn.ModuleList([SplineLinear(a, b, arm) for a, b in zip(widths[:-1], widths[1:])])
        with torch.no_grad():
            for layer in self.layers:
                layer.spline_weight.zero_()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x