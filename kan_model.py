import torch
import torch.nn as nn
from kan_layer import KANLayer
from config import *


class KANModel(nn.Module):
    """Embedding + PosEnc -> KAN -> attention pool -> FC  (no GRU)"""
    def __init__(self, n_items):
        super().__init__()
        self.emb     = nn.Embedding(n_items, EMBEDDING_DIM, padding_idx=0)
        self.pos     = nn.Embedding(MAX_SEQ_LEN + 1, EMBEDDING_DIM)
        self.kan     = KANLayer(EMBEDDING_DIM, HIDDEN_DIM)
        self.norm    = nn.LayerNorm(HIDDEN_DIM)
        self.attn    = nn.Linear(HIDDEN_DIM, 1)
        self.drop    = nn.Dropout(DROPOUT)
        self.fc      = nn.Linear(HIDDEN_DIM, n_items - 1)

    def forward(self, x):
        B, T   = x.shape
        mask   = (x != 0).float()
        pos_id = torch.arange(T, device=x.device).unsqueeze(0).clamp(max=MAX_SEQ_LEN)
        e      = self.drop(self.emb(x) + self.pos(pos_id))      # (B, T, E)

        # KAN over each time step
        flat   = e.reshape(B * T, EMBEDDING_DIM)
        h      = self.norm(self.kan(flat)).reshape(B, T, HIDDEN_DIM)

        # Masked attention pool
        scores = self.attn(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        w      = torch.softmax(scores, -1).unsqueeze(-1)
        pooled = (h * w).sum(1)
        return self.fc(self.drop(pooled))
