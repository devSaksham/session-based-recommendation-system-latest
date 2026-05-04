import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_gru import KANGRULayer
from kan_layer import KANLinear
from config import (
    EMBEDDING_DIM, HIDDEN_DIM, DROPOUT, KAN_GRU_LAYERS, MAX_SEQ_LEN,
    KAN_USE_POSITION_EMB, KAN_USE_ATTENTION_READOUT, KAN_USE_KAN_READOUT,
    USE_MULTI_VIEW_READOUT, USE_TIED_OUTPUT, USE_LAST_ITEM_RESIDUAL,
    USE_TWO_TOWER_OUTPUT, TWO_TOWER_LAST_WEIGHT_INIT, USE_REPEAT_BOOST,
    REPEAT_BOOST_INIT, USE_REPEAT_TRANSITION_MIXTURE,
    REPEAT_MIXTURE_STRENGTH_INIT, USE_UNTIED_LOGIT_RESIDUAL,
    LOGIT_RESIDUAL_INIT, USE_MARKOV_TRANSITION_PRIOR,
    MARKOV_PRIOR_WEIGHT_INIT
)


class KANGRUModel(nn.Module):
    """
    Embedding -> KAN-GRU -> last hidden state -> repeat/transition scorer

    The recurrent core is still a KAN-GRU. The default readout is deliberately
    simple again so changes focus on the recurrent KAN cell and output scorer.
    """
    def __init__(self, n_items):
        super().__init__()
        self.use_position_emb = KAN_USE_POSITION_EMB
        self.use_attention_readout = KAN_USE_ATTENTION_READOUT
        self.use_kan_readout = KAN_USE_KAN_READOUT
        self.use_multi_view_readout = USE_MULTI_VIEW_READOUT

        self.emb = nn.Embedding(n_items, EMBEDDING_DIM, padding_idx=0)
        if self.use_position_emb:
            self.pos_emb = nn.Embedding(MAX_SEQ_LEN, EMBEDDING_DIM)

        self.gru = KANGRULayer(
            EMBEDDING_DIM,
            HIDDEN_DIM,
            num_layers=KAN_GRU_LAYERS,
            dropout=DROPOUT,
        )
        self.drop = nn.Dropout(DROPOUT)

        if self.use_attention_readout:
            self.attn_state = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.attn_last = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
            self.attn_score = nn.Linear(HIDDEN_DIM, 1, bias=False)
            readout_dim = HIDDEN_DIM * 3
        elif self.use_multi_view_readout:
            readout_dim = HIDDEN_DIM * 3
        else:
            readout_dim = HIDDEN_DIM

        if self.use_kan_readout:
            self.readout_kan = KANLinear(readout_dim, HIDDEN_DIM)
            self.readout_linear = nn.Linear(readout_dim, HIDDEN_DIM)
            self.readout_norm = nn.LayerNorm(HIDDEN_DIM)
            classifier_dim = HIDDEN_DIM
        else:
            self.readout_linear = nn.Linear(readout_dim, HIDDEN_DIM)
            self.readout_norm = nn.LayerNorm(HIDDEN_DIM)
            classifier_dim = HIDDEN_DIM

        self.use_tied_output = USE_TIED_OUTPUT
        self.use_last_item_residual = USE_LAST_ITEM_RESIDUAL
        self.use_two_tower_output = USE_TWO_TOWER_OUTPUT
        self.use_repeat_boost = USE_REPEAT_BOOST
        self.use_repeat_mixture = USE_REPEAT_TRANSITION_MIXTURE
        self.use_markov_prior = USE_MARKOV_TRANSITION_PRIOR
        self.use_logit_residual = USE_UNTIED_LOGIT_RESIDUAL
        self.register_buffer("markov_indices", torch.zeros(0, 0, dtype=torch.long), persistent=False)
        self.register_buffer("markov_scores", torch.zeros(0, 0), persistent=False)

        if self.use_tied_output:
            self.output_proj = nn.Linear(classifier_dim, EMBEDDING_DIM)
            self.output_norm = nn.LayerNorm(EMBEDDING_DIM)
            self.output_bias = nn.Parameter(torch.zeros(n_items - 1))
            self.last_item_alpha = nn.Parameter(torch.tensor(0.2))
            self.logit_scale = nn.Parameter(torch.tensor(10.0))
            if self.use_two_tower_output:
                self.last_output_proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
                self.last_output_norm = nn.LayerNorm(EMBEDDING_DIM)
                self.last_tower_alpha = nn.Parameter(torch.tensor(TWO_TOWER_LAST_WEIGHT_INIT))
            if self.use_repeat_boost:
                self.repeat_logit_alpha = nn.Parameter(torch.tensor(REPEAT_BOOST_INIT))
            if self.use_repeat_mixture:
                self.repeat_gate = nn.Linear(classifier_dim, 1)
                self.repeat_strength = nn.Parameter(torch.tensor(REPEAT_MIXTURE_STRENGTH_INIT))
            if self.use_markov_prior:
                self.markov_prior_alpha = nn.Parameter(torch.tensor(MARKOV_PRIOR_WEIGHT_INIT))
            if self.use_logit_residual:
                self.logit_residual = nn.Linear(classifier_dim, n_items - 1)
                self.logit_residual_alpha = nn.Parameter(torch.tensor(LOGIT_RESIDUAL_INIT))
        else:
            self.fc = nn.Linear(classifier_dim, n_items - 1)

    def set_markov_transition_prior(self, indices, scores):
        self.markov_indices = indices.to(dtype=torch.long)
        self.markov_scores = scores.to(dtype=torch.float)

    def _add_position_embedding(self, x, emb):
        if not self.use_position_emb:
            return emb
        steps = emb.size(1)
        positions = torch.arange(steps, device=emb.device).clamp(max=MAX_SEQ_LEN - 1)
        pos = self.pos_emb(positions).unsqueeze(0)
        return emb + pos * (x != 0).unsqueeze(-1)

    def _attention_readout(self, out, last, mask):
        if not self.use_attention_readout:
            return last
        scores = self.attn_score(
            torch.tanh(self.attn_state(out) + self.attn_last(last).unsqueeze(1))
        ).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (weights * out).sum(dim=1)
        return torch.cat([last, context, last * context], dim=-1)

    def _session_readout(self, out, last, mask):
        if self.use_attention_readout:
            return self._attention_readout(out, last, mask)
        if self.use_multi_view_readout:
            mask_f = mask.unsqueeze(-1).to(out.dtype)
            mean = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            return torch.cat([last, mean, last * mean], dim=-1)
        return last

    def _score_items(self, session, x, lengths):
        if not self.use_tied_output:
            return self.fc(self.drop(session))

        dropped_session = self.drop(session)
        query = self.output_norm(self.output_proj(dropped_session))
        last_item_ids = None
        if (
            self.use_last_item_residual
            or self.use_two_tower_output
            or self.use_repeat_boost
            or self.use_repeat_mixture
            or self.use_markov_prior
        ):
            last_item_ids = x.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        if self.use_last_item_residual:
            query = query + self.last_item_alpha * self.emb(last_item_ids)

        query = F.normalize(query, dim=-1)
        item_weight = F.normalize(self.emb.weight[1:], dim=-1)
        scale = self.logit_scale.clamp(1.0, 30.0)
        logits = scale * query @ item_weight.t() + self.output_bias

        if self.use_two_tower_output:
            last_query = self.last_output_norm(self.last_output_proj(self.emb(last_item_ids)))
            last_query = F.normalize(last_query, dim=-1)
            last_logits = scale * last_query @ item_weight.t()
            logits = logits + self.last_tower_alpha.clamp(0.0, 2.0) * last_logits

        if self.use_logit_residual:
            logits = logits + self.logit_residual_alpha * self.logit_residual(dropped_session)
        if self.use_repeat_boost:
            repeat_indices = (last_item_ids - 1).clamp(min=0).unsqueeze(1)
            repeat_boost = torch.zeros_like(logits)
            repeat_boost.scatter_add_(
                1,
                repeat_indices,
                self.repeat_logit_alpha.expand_as(repeat_indices).to(logits.dtype),
            )
            logits = logits + repeat_boost
        if self.use_repeat_mixture:
            repeat_indices = (last_item_ids - 1).clamp(min=0).unsqueeze(1)
            repeat_gate = torch.sigmoid(self.repeat_gate(session))
            repeat_strength = F.softplus(self.repeat_strength)
            repeat_boost = torch.zeros_like(logits)
            repeat_boost.scatter_add_(
                1,
                repeat_indices,
                (repeat_gate * repeat_strength).to(logits.dtype),
            )
            logits = logits + repeat_boost
        if self.use_markov_prior and self.markov_scores.numel():
            prior_indices = self.markov_indices[last_item_ids]
            prior_scores = self.markov_scores[last_item_ids].to(logits.dtype)
            prior_logits = torch.zeros_like(logits)
            prior_logits.scatter_add_(1, prior_indices, prior_scores)
            logits = logits + self.markov_prior_alpha * prior_logits
        return logits

    def forward(self, x):
        mask = x != 0
        lengths = mask.sum(1).clamp(min=1)
        emb = self._add_position_embedding(x, self.emb(x))
        out, _ = self.gru(self.drop(emb))
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, HIDDEN_DIM)
        last = out.gather(1, idx).squeeze(1)

        session = self._session_readout(out, last, mask)
        if self.use_kan_readout:
            session = self.readout_norm(self.readout_kan(session) + self.readout_linear(session))
        else:
            session = self.readout_norm(self.readout_linear(session))
        session = torch.tanh(session)
        return self._score_items(session, x, lengths)
