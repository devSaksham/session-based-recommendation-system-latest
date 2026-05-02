import torch
import torch.nn as nn
import torch.nn.functional as F
from config import KAN_GRID, KAN_ORDER


class KANLayer(nn.Module):
    """
    Simple KAN layer using B-spline basis.
    Each input feature gets its own learnable spline curve.
    """
    def __init__(self, in_dim, out_dim, grid=KAN_GRID, order=KAN_ORDER):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.grid    = grid
        self.order   = order
        n_coeff      = grid + order

        # Learnable spline coefficients
        self.coeff   = nn.Parameter(torch.randn(out_dim, in_dim, n_coeff) * 0.1)
        # Residual linear path
        self.linear  = nn.Linear(in_dim, out_dim, bias=True)
        # Per-connection scale
        self.scale   = nn.Parameter(torch.ones(out_dim, in_dim))

        knots = torch.linspace(-1, 1, grid + 1)
        self.register_buffer("knots", knots)

    def _bspline(self, x):
        # x: (B, in_dim)  ->  basis: (B, in_dim, n_coeff)
        x   = x.unsqueeze(-1).clamp(-1 + 1e-5, 1 - 1e-5)          # (B, D, 1)
        k0  = self.knots                                             # (G+1,)
        # Order-0: indicator which grid interval x falls in
        b   = ((x >= k0[:-1]) & (x < k0[1:])).float()              # (B, D, G)

        for k in range(1, self.order + 1):
            # Extend knot vector with repeated boundary knots
            pad_l = k0[:1].repeat(k)
            pad_r = k0[-1:].repeat(k)
            kext  = torch.cat([pad_l, k0, pad_r])                   # G+1+2k

            left_k  = kext[:-(k+1)]   # denom bases
            right_k = kext[k+1:]
            denom   = (right_k - left_k).clamp(min=1e-6)            # (G+2k-1,... )

            # Trim to match b width
            w       = b.size(-1)
            alpha   = ((x - kext[:w]) / denom[:w].unsqueeze(0).unsqueeze(0)).clamp(0, 1)
            b_shift = F.pad(b[..., 1:], (0, 1))
            b       = alpha * b + (1 - alpha) * b_shift

            # Ensure correct width
            target  = self.grid + self.order
            if b.size(-1) < target:
                b = F.pad(b, (0, target - b.size(-1)))
            else:
                b = b[..., :target]

        return b   # (B, in_dim, n_coeff)

    def forward(self, x):
        # x: (B, in_dim)
        xn    = torch.tanh(x)                                       # normalize to [-1,1]
        basis = self._bspline(xn)                                    # (B, D, n_coeff)
        # Spline: sum over coefficients weighted by scale
        spline = torch.einsum("bdc,odc->bo", basis, self.coeff * self.scale.unsqueeze(-1))
        return spline + self.linear(x)                               # residual
