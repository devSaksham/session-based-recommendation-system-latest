import copy, time

import torch
import torch.nn as nn

from config import (
    DEVICE, EPOCHS, LR, WEIGHT_DECAY, LABEL_SMOOTHING, MIN_DELTA,
    MONITOR_METRIC, MONITOR_MODE, LR_PATIENCE, EARLY_STOPPING_PATIENCE,
    KAN_REG_ACTIVATION, KAN_REG_ENTROPY
)
from metrics import top1_acc, hr_at_k, ndcg_at_k, mrr_at_k
from model_inspector import display_training_setup


def _kan_regularization(model):
    loss = None
    for module in model.modules():
        if module is model or not hasattr(module, "regularization_loss"):
            continue
        value = module.regularization_loss(
            regularize_activation=KAN_REG_ACTIVATION,
            regularize_entropy=KAN_REG_ENTROPY,
        )
        loss = value if loss is None else loss + value
    return loss


def _optimizer_param_groups(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("bias") or "norm" in name.lower() or "emb." in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def run_epoch(model, loader, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    tot_loss = tot_acc = tot_hr = tot_ndcg = tot_mrr = 0.0
    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss   = criterion(logits, y)
            if optimizer:
                reg_loss = _kan_regularization(model)
                if reg_loss is not None:
                    loss = loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot_loss += loss.item()
            tot_acc  += top1_acc(logits.detach(), y)
            tot_hr   += hr_at_k(logits.detach(), y)
            tot_ndcg += ndcg_at_k(logits.detach(), y)
            tot_mrr  += mrr_at_k(logits.detach(), y)
    n = len(loader)
    return tot_loss/n, tot_acc/n, tot_hr/n, tot_ndcg/n, tot_mrr/n


def train(model, train_loader, val_loader, name="Model"):
    model.to(DEVICE)
    opt  = torch.optim.AdamW(_optimizer_param_groups(model), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=MONITOR_MODE, factor=0.5, patience=LR_PATIENCE
    )
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    hist = {k: [] for k in ["tr_loss","tr_acc","vl_loss","vl_acc","vl_hr","vl_ndcg","vl_mrr"]}
    if MONITOR_METRIC not in hist:
        raise ValueError(f"MONITOR_METRIC must be one of {sorted(hist)}")
    if MONITOR_MODE not in {"min", "max"}:
        raise ValueError("MONITOR_MODE must be 'min' or 'max'")

    display_training_setup(model, train_loader, val_loader, opt, sch, crit, name)

    print(f"\n-- {name} ----------------------------------------------")
    print(f"{'Ep':>3} {'TrLoss':>7} {'TrAcc':>7} {'VlLoss':>7} {'VlAcc':>7} {'HR@10':>7} {'NDCG':>7} {'MRR':>7}")

    t0 = time.time()
    best_score = float("inf") if MONITOR_MODE == "min" else -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for ep in range(1, EPOCHS + 1):
        tl, ta, _, _, _             = run_epoch(model, train_loader, crit, opt)
        vl, va, vhr, vndcg, vmrr    = run_epoch(model, val_loader,   crit)
        for k, v in zip(hist, [tl, ta, vl, va, vhr, vndcg, vmrr]):
            hist[k].append(v)
        monitor_value = hist[MONITOR_METRIC][-1]
        sch.step(monitor_value)
        print(f"{ep:>3} {tl:>7.4f} {ta:>7.4f} {vl:>7.4f} {va:>7.4f} {vhr:>7.4f} {vndcg:>7.4f} {vmrr:>7.4f}")

        improved = (
            monitor_value < best_score - MIN_DELTA
            if MONITOR_MODE == "min"
            else monitor_value > best_score + MIN_DELTA
        )
        if improved:
            best_score = monitor_value
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping: no {MONITOR_METRIC} improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    print(f"  Best {MONITOR_METRIC}: {best_score:.4f}")
    print(f"  Training time: {elapsed:.1f}s")
    return hist, elapsed


_run_epoch = run_epoch
