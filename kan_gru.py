import torch
import torch.nn as nn

from config import DROPOUT, KAN_ON_GATES
from kan_layer import KANLinear


class KANGRUCell(nn.Module):
    """
    GRU cell with KAN transforms inside the gates.

    The recurrent update remains GRU-like for sequence modelling, while the
    gate projections use learnable spline activations from KANLinear.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        combined = input_dim + hidden_dim

        gate_layer = KANLinear if KAN_ON_GATES else nn.Linear
        self.gate_z = gate_layer(combined, hidden_dim)
        self.gate_r = gate_layer(combined, hidden_dim)
        self.candidate_kan = KANLinear(combined, hidden_dim)
        self.candidate_linear = nn.Linear(combined, hidden_dim)

        self.norm_z = nn.LayerNorm(hidden_dim)
        self.norm_r = nn.LayerNorm(hidden_dim)
        self.norm_n = nn.LayerNorm(hidden_dim)

        if hasattr(self.gate_z, "bias") and self.gate_z.bias is not None:
            nn.init.constant_(self.gate_z.bias, -1.0)

    def forward(self, x_t, h_prev):
        xh = torch.cat([x_t, h_prev], dim=-1)

        z_t = torch.sigmoid(self.norm_z(self.gate_z(xh)))
        r_t = torch.sigmoid(self.norm_r(self.gate_r(xh)))

        candidate_input = torch.cat([x_t, r_t * h_prev], dim=-1)
        candidate = self.candidate_kan(candidate_input) + self.candidate_linear(candidate_input)
        n_t = torch.tanh(self.norm_n(candidate))

        return (1.0 - z_t) * h_prev + z_t * n_t


class KANGRULayer(nn.Module):
    """Stacked KAN-GRU layer with an nn.GRU-like output contract."""
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            KANGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, h0=None):
        batch, steps, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch, self.hidden_dim, device=x.device)

        h_prev = [h0[i] for i in range(self.num_layers)]
        outputs = []

        for t in range(steps):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h_t = cell(inp, h_prev[i])
                inp = self.drop(h_t) if i + 1 < self.num_layers else h_t
                h_prev[i] = h_t
            outputs.append(h_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), torch.stack(h_prev, dim=0)
