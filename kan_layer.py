import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KAN_GRID, KAN_ORDER


class KANLinear(nn.Module):
    """
    Efficient-KAN style linear layer.

    This keeps the same KAN idea, learnable B-spline activations per input
    feature, but computes spline bases once and combines them with a single
    matrix multiplication instead of expanding by output feature.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        grid=KAN_GRID,
        order=KAN_ORDER,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=0.5,
        base_activation=nn.SiLU,
        grid_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid = grid
        self.order = order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        step = (grid_range[1] - grid_range[0]) / grid
        knots = (
            torch.arange(-order, grid + order + 1, dtype=torch.float32) * step
            + grid_range[0]
        )
        self.register_buffer("knots", knots.expand(in_dim, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.spline_weight = nn.Parameter(torch.empty(out_dim, in_dim, grid + order))
        self.spline_scaler = nn.Parameter(torch.empty(out_dim, in_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid + 1, self.in_dim, self.out_dim) - 0.5
            ) * self.scale_noise / self.grid
            coeff = self.curve2coeff(
                self.knots.T[self.order:-self.order],
                noise,
            )
            self.spline_weight.copy_(coeff * self.scale_spline)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_dim
        knots = self.knots
        x = x.unsqueeze(-1)
        bases = ((x >= knots[:, :-1]) & (x < knots[:, 1:])).to(x.dtype)

        for k in range(1, self.order + 1):
            left_den = knots[:, k:-1] - knots[:, :-(k + 1)]
            right_den = knots[:, k + 1:] - knots[:, 1:-k]
            bases = (
                (x - knots[:, :-(k + 1)]) / left_den * bases[:, :, :-1]
                + (knots[:, k + 1:] - x) / right_den * bases[:, :, 1:]
            )

        expected = (x.size(0), self.in_dim, self.grid + self.order)
        assert bases.size() == expected
        return bases.contiguous()

    def curve2coeff(self, x, y):
        assert x.dim() == 2 and x.size(1) == self.in_dim
        assert y.size() == (x.size(0), self.in_dim, self.out_dim)
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(a, b).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_dim, self.in_dim, self.grid + self.order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x):
        assert x.size(-1) == self.in_dim
        original_shape = x.shape
        x = x.reshape(-1, self.in_dim)
        x_spline = torch.tanh(x)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x_spline).reshape(x.size(0), -1),
            self.scaled_spline_weight.reshape(self.out_dim, -1),
        )
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_dim)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1 = self.spline_weight.abs().mean(-1)
        activation_loss = l1.sum()
        p = l1 / activation_loss.clamp_min(1e-8)
        entropy_loss = -torch.sum(p * p.clamp_min(1e-8).log())
        return regularize_activation * activation_loss + regularize_entropy * entropy_loss


class KANLayer(KANLinear):
    pass
