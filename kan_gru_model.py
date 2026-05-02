import torch
import torch.nn as nn
from kan_gru import KANGRULayer
from config import EMBEDDING_DIM, HIDDEN_DIM, DROPOUT


class KANGRUModel(nn.Module):
    """
    Embedding -> KAN-GRU -> last hidden state -> FC

    Identical structure to the RNN model but every gate inside
    the GRU (update, reset, new) uses a KAN layer instead of
    a fixed Linear + sigmoid/tanh.

    This is the deepest KAN integration:
    - RNN model    : fixed activations inside gates
    - KAN-GRU model: learnable spline activations inside gates
    """
    def __init__(self, n_items):
        super().__init__()
        self.emb  = nn.Embedding(n_items, EMBEDDING_DIM, padding_idx=0)
        self.gru  = KANGRULayer(EMBEDDING_DIM, HIDDEN_DIM, num_layers=2, dropout=DROPOUT)
        self.drop = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(HIDDEN_DIM, n_items - 1)

    def forward(self, x):
        lengths    = (x != 0).sum(1).clamp(min=1)
        emb        = self.drop(self.emb(x))             # (B, T, E)
        out, _     = self.gru(emb)                      # (B, T, H)
        idx        = (lengths - 1).view(-1, 1, 1).expand(-1, 1, HIDDEN_DIM)
        last       = out.gather(1, idx).squeeze(1)      # (B, H)
        return self.fc(self.drop(last))
