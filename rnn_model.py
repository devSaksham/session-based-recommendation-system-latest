import torch
import torch.nn as nn
from config import *


class RNNModel(nn.Module):
    """Embedding -> GRU -> last hidden -> FC"""
    def __init__(self, n_items):
        super().__init__()
        self.emb  = nn.Embedding(n_items, EMBEDDING_DIM, padding_idx=0)
        self.gru  = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, num_layers=2,
                           batch_first=True, dropout=DROPOUT)
        self.drop = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(HIDDEN_DIM, n_items - 1)

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1)
        out, _  = self.gru(self.drop(self.emb(x)))
        idx     = (lengths - 1).view(-1, 1, 1).expand(-1, 1, HIDDEN_DIM)
        last    = out.gather(1, idx).squeeze(1)
        return self.fc(self.drop(last))
