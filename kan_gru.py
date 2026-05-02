import torch
import torch.nn as nn
import torch.nn.functional as F
from config import KAN_GRID, KAN_ORDER, DROPOUT


# ── B-spline KAN linear (same as before) ────────────────────────────────────

class KANLinear(nn.Module):
    """KAN layer using B-spline basis functions."""
    def __init__(self, in_dim, out_dim, grid=KAN_GRID, order=KAN_ORDER):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.grid    = grid
        self.order   = order
        n_coeff      = grid + order

        self.coeff  = nn.Parameter(torch.randn(out_dim, in_dim, n_coeff) * 0.1)
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.scale  = nn.Parameter(torch.ones(out_dim, in_dim))
        self.register_buffer("knots", torch.linspace(-1, 1, grid + 1))

    def _bspline(self, x):
        x  = x.unsqueeze(-1).clamp(-1 + 1e-5, 1 - 1e-5)
        k0 = self.knots
        b  = ((x >= k0[:-1]) & (x < k0[1:])).float()
        for k in range(1, self.order + 1):
            kext  = torch.cat([k0[:1].repeat(k), k0, k0[-1:].repeat(k)])
            w     = b.size(-1)
            denom = (kext[k:k+w] - kext[:w]).clamp(min=1e-6)
            alpha = ((x - kext[:w]) / denom).clamp(0, 1)
            b     = alpha * b + (1 - alpha) * F.pad(b[..., 1:], (0, 1))
            target = self.grid + self.order
            b = F.pad(b, (0, target - b.size(-1))) if b.size(-1) < target else b[..., :target]
        return b

    def forward(self, x):
        basis  = self._bspline(torch.tanh(x))
        spline = torch.einsum("bdc,odc->bo", basis, self.coeff * self.scale.unsqueeze(-1))
        return spline + self.linear(x)


# ── KAN-GRU Cell ─────────────────────────────────────────────────────────────

class KANGRUCell(nn.Module):
    """
    GRU cell where sigmoid/tanh activations are replaced by KAN layers.

    Standard GRU:
        z_t = sigmoid( W_z * [x_t, h_{t-1}] )
        r_t = sigmoid( W_r * [x_t, h_{t-1}] )
        n_t = tanh(    W_n * (x_t + r_t * h_{t-1}) )
        h_t = (1 - z_t) * h_{t-1}  +  z_t * n_t

    KAN-GRU (this class):
        z_t = KAN_z( [x_t, h_{t-1}] )   <- KAN replaces Linear+sigmoid
        r_t = KAN_r( [x_t, h_{t-1}] )   <- KAN replaces Linear+sigmoid
        n_t = KAN_n( x_t + r_t * h_{t-1} ) <- KAN replaces Linear+tanh
        h_t = (1 - z_t) * h_{t-1}  +  z_t * n_t   <- same gating formula

    KAN learns the shape of each gate's transformation — not locked to
    sigmoid or tanh. Each input dimension gets its own learnable spline curve.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        combined        = input_dim + hidden_dim

        # Three KAN layers — one per gate
        self.kan_z = KANLinear(combined,    hidden_dim)   # update gate
        self.kan_r = KANLinear(combined,    hidden_dim)   # reset gate
        self.kan_n = KANLinear(hidden_dim,  hidden_dim)   # new/candidate gate

        # Extra transform for x_t in the new gate
        self.kan_x = KANLinear(input_dim,   hidden_dim)

        self.norm_z = nn.LayerNorm(hidden_dim)
        self.norm_r = nn.LayerNorm(hidden_dim)
        self.norm_n = nn.LayerNorm(hidden_dim)

    def forward(self, x_t, h_prev):
        """
        x_t   : (B, input_dim)
        h_prev: (B, hidden_dim)
        returns h_t: (B, hidden_dim)
        """
        xh = torch.cat([x_t, h_prev], dim=-1)          # (B, input_dim + hidden_dim)

        # Update gate  — KAN replaces Linear + sigmoid
        z_t = torch.sigmoid(self.norm_z(self.kan_z(xh)))   # (B, H)

        # Reset gate   — KAN replaces Linear + sigmoid
        r_t = torch.sigmoid(self.norm_r(self.kan_r(xh)))   # (B, H)

        # New gate     — KAN replaces Linear + tanh
        # r_t controls how much of h_prev flows into the candidate
        gated_h = r_t * h_prev                              # (B, H)
        x_part  = self.kan_x(x_t)                          # (B, H)
        n_t     = torch.tanh(self.norm_n(
                      self.kan_n(gated_h) + x_part         # (B, H)
                  ))

        # Final hidden state — same interpolation formula as standard GRU
        h_t = (1.0 - z_t) * h_prev + z_t * n_t            # (B, H)
        return h_t


# ── KAN-GRU Layer (stacks cells, matches nn.GRU interface) ──────────────────

class KANGRULayer(nn.Module):
    """
    Stacks multiple KANGRUCell layers.
    Matches the same interface as nn.GRU (batch_first=True).

    Layer 0: input  = embedding (input_dim)
    Layer 1: input  = h_t from layer 0  (hidden_dim)
    ...

    Returns:
        outputs : (B, T, hidden_dim)  — h_t at every step from last layer
        h_n     : (num_layers, B, hidden_dim) — final hidden state per layer
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cells = nn.ModuleList([
            KANGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, h0=None):
        """
        x  : (B, T, input_dim)
        h0 : (num_layers, B, hidden_dim)  — zeros if None
        """
        B, T, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)

        h_prev   = [h0[i] for i in range(self.num_layers)]
        outputs  = []

        for t in range(T):
            inp = x[:, t, :]                            # (B, input_dim)
            for i, cell in enumerate(self.cells):
                h_t      = cell(inp, h_prev[i])         # (B, hidden_dim)
                inp      = self.drop(h_t)               # next layer input
                h_prev[i] = h_t

            outputs.append(h_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)             # (B, T, H)
        h_n     = torch.stack(h_prev, dim=0)            # (num_layers, B, H)
        return outputs, h_n
