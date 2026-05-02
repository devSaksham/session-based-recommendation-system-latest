import torch
import torch.nn as nn
from kan_layer import KANLayer
from config import *


class HybridModel(nn.Module):
    """
    Embedding -> GRU -> KAN -> FC

    GRU captures sequential order.
    KAN then refines GRU hidden states with learnable spline functions
    before the final item prediction.
    """
    def __init__(self, n_items):
        super().__init__()
        self.emb  = nn.Embedding(n_items, EMBEDDING_DIM, padding_idx=0)
        self.gru  = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, num_layers=2,
                           batch_first=True, dropout=DROPOUT)
        self.kan  = KANLayer(HIDDEN_DIM, HIDDEN_DIM)   # KAN refines GRU output
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(HIDDEN_DIM, n_items - 1)

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1)

        # Step 1: GRU over the sequence
        out, _  = self.gru(self.drop(self.emb(x)))              # (B, T, H)

        # Step 2: pick last non-pad hidden state
        idx     = (lengths - 1).view(-1, 1, 1).expand(-1, 1, HIDDEN_DIM)
        last    = out.gather(1, idx).squeeze(1)                  # (B, H)

        # Step 3: KAN refines the GRU hidden state
        refined = self.norm(self.kan(last))                      # (B, H)

        return self.fc(self.drop(refined))
