import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LABEL_SMOOTHING, LOSS_TYPE


LOSS_ALIASES = {
    "ce": "cross_entropy",
    "crossentropy": "cross_entropy",
    "cross-entropy": "cross_entropy",
    "cross_entropy": "cross_entropy",
    "bpr": "bpr",
    "top1": "top1",
    "top-1": "top1",
    "top_1": "top1",
}


def normalize_loss_type(loss_type=None):
    value = LOSS_TYPE if loss_type is None else loss_type
    key = str(value).strip().lower().replace(" ", "_")
    if key not in LOSS_ALIASES:
        valid = ", ".join(sorted(set(LOSS_ALIASES.values())))
        raise ValueError(f"Unknown LOSS_TYPE={loss_type!r}. Use one of: {valid}")
    return LOSS_ALIASES[key]


def _negative_logits(logits, targets):
    if logits.size(1) <= 1:
        raise ValueError("Ranking losses need at least two item classes.")
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets.view(-1, 1), False)
    return logits.masked_select(mask).view(logits.size(0), logits.size(1) - 1)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss over all non-target items."""

    display_name = "BPR"

    def forward(self, logits, targets):
        pos = logits.gather(1, targets.view(-1, 1))
        neg = _negative_logits(logits, targets)
        return -F.logsigmoid(pos - neg).mean()


class TOP1Loss(nn.Module):
    """GRU4Rec TOP1 ranking loss over all non-target items."""

    display_name = "TOP1"

    def forward(self, logits, targets):
        pos = logits.gather(1, targets.view(-1, 1))
        neg = _negative_logits(logits, targets)
        return (torch.sigmoid(neg - pos) + torch.sigmoid(neg.pow(2))).mean()


def build_loss(loss_type=None):
    loss_type = normalize_loss_type(loss_type)
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    if loss_type == "bpr":
        return BPRLoss()
    if loss_type == "top1":
        return TOP1Loss()
    raise AssertionError(f"Unhandled normalized loss type: {loss_type}")


def loss_display_name(loss_type=None):
    loss_type = normalize_loss_type(loss_type)
    return {
        "cross_entropy": "Cross-entropy",
        "bpr": "BPR",
        "top1": "TOP1",
    }[loss_type]
