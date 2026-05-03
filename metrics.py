import torch
from config import TOP_K


def top1_acc(logits, targets):
    return (logits.argmax(-1) == targets).float().mean().item()


def hr_at_k(logits, targets, k=TOP_K):
    _, topk = torch.topk(logits, k, dim=-1)
    return (topk == targets.unsqueeze(1)).any(1).float().mean().item()


def ndcg_at_k(logits, targets, k=TOP_K):
    _, topk = torch.topk(logits, k, dim=-1)
    hits    = (topk == targets.unsqueeze(1)).float()
    disc    = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=logits.device))
    return (hits * disc).sum(-1).mean().item()


def mrr_at_k(logits, targets, k=TOP_K):
    _, topk = torch.topk(logits, k, dim=-1)
    hits = topk == targets.unsqueeze(1)
    ranks = torch.arange(1, k + 1, dtype=torch.float, device=logits.device).unsqueeze(0)
    reciprocal = torch.where(hits, 1.0 / ranks, torch.zeros_like(ranks))
    return reciprocal.max(dim=1).values.mean().item()


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
