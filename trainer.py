import time, torch, torch.nn as nn
from config import DEVICE, EPOCHS, LR
from metrics import top1_acc, hr_at_k, ndcg_at_k


def _run_epoch(model, loader, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    tot_loss = tot_acc = tot_hr = tot_ndcg = 0.0
    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss   = criterion(logits, y)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot_loss += loss.item()
            tot_acc  += top1_acc(logits.detach(), y)
            tot_hr   += hr_at_k(logits.detach(), y)
            tot_ndcg += ndcg_at_k(logits.detach(), y)
    n = len(loader)
    return tot_loss/n, tot_acc/n, tot_hr/n, tot_ndcg/n


def train(model, train_loader, val_loader, name="Model"):
    model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    crit = nn.CrossEntropyLoss()
    hist = {k: [] for k in ["tr_loss","tr_acc","vl_loss","vl_acc","vl_hr","vl_ndcg"]}

    print(f"\n── {name} ──────────────────────────────────────────────")
    print(f"{'Ep':>3} {'TrLoss':>7} {'TrAcc':>7} {'VlLoss':>7} {'VlAcc':>7} {'HR@10':>7} {'NDCG':>7}")

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tl, ta, _, _          = _run_epoch(model, train_loader, crit, opt)
        vl, va, vhr, vndcg    = _run_epoch(model, val_loader,   crit)
        sch.step()
        for k, v in zip(hist, [tl, ta, vl, va, vhr, vndcg]):
            hist[k].append(v)
        print(f"{ep:>3} {tl:>7.4f} {ta:>7.4f} {vl:>7.4f} {va:>7.4f} {vhr:>7.4f} {vndcg:>7.4f}")

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    return hist, elapsed
