"""
4-Way Comparison: RNN vs KAN vs Hybrid vs KAN-GRU
Dataset: YooChoose
Run:  python compare.py
"""
import os, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate
import torch.nn as nn

from config        import EPOCHS, SEED, DEVICE
from data_loader   import load_data, get_loaders
from rnn_model     import RNNModel
from kan_model     import KANModel
from hybrid_model  import HybridModel
from kan_gru_model import KANGRUModel
from trainer       import train, _run_epoch
from metrics       import n_params

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────
print("\n[Data] Loading YooChoose...")
train_seqs, test_seqs, n_items = load_data()
train_loader, test_loader      = get_loaders(train_seqs, test_seqs)

# ── Train all four ───────────────────────────────────────────────
MODELS = [
    ("RNN",     RNNModel),
    ("KAN",     KANModel),
    ("Hybrid",  HybridModel),
    ("KAN-GRU", KANGRUModel),
]

results = {}
for name, ModelClass in MODELS:
    model         = ModelClass(n_items)
    hist, elapsed = train(model, train_loader, test_loader, name)
    torch.save(model.state_dict(), f"checkpoints/{name.lower().replace('-','_')}.pt")
    crit = nn.CrossEntropyLoss()
    vl, va, vhr, vndcg = run_epoch(model, test_loader, crit)
    results[name] = dict(hist=hist, time=elapsed, params=n_params(model),
                         loss=vl, acc=va, hr=vhr, ndcg=vndcg)

# ── Plots ─────────────────────────────────────────────────────────
COLORS = {
    "RNN":     "#E74C3C",
    "KAN":     "#2ECC71",
    "Hybrid":  "#3498DB",
    "KAN-GRU": "#F39C12",
}
epochs = range(1, EPOCHS + 1)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("YooChoose: RNN vs KAN vs Hybrid vs KAN-GRU", fontsize=15, fontweight="bold")
axes = axes.flatten()

panels = [
    ("tr_loss", "Training Loss",   "Loss"),
    ("vl_loss", "Validation Loss", "Loss"),
    ("vl_acc",  "Validation Acc",  "Top-1 Accuracy"),
    ("vl_hr",   "HR@10",           "HR@10"),
    ("vl_ndcg", "NDCG@10",         "NDCG@10"),
]

for ax, (key, title, ylabel) in zip(axes, panels):
    for name, r in results.items():
        ax.plot(epochs, r["hist"][key], "o-", label=name,
                color=COLORS[name], ms=4, linewidth=1.8)
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)

ax    = axes[5]
names = [n for n, _ in MODELS]
hrs   = [results[n]["hr"] for n in names]
bars  = ax.bar(names, hrs, color=[COLORS[n] for n in names], width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=3)
ax.set_title("Final HR@10"); ax.set_ylabel("HR@10")
ax.set_ylim(0, max(hrs) * 1.3); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("plots/comparison.png", dpi=150)
print("\nPlot saved -> plots/comparison.png")

# ── Results Table ─────────────────────────────────────────────────
def d(a, b, low=False):
    diff = (b - a) if low else (a - b)
    return f"{'up' if diff >= 0 else 'dn'} {abs(diff):.4f}"

rnn, kan, hyb, kgru = results["RNN"], results["KAN"], results["Hybrid"], results["KAN-GRU"]

rows = [
    ["Val Loss",   f"{rnn['loss']:.4f}",  f"{kan['loss']:.4f}",  f"{hyb['loss']:.4f}",  f"{kgru['loss']:.4f}",  d(kan['loss'],  rnn['loss'], True),  d(hyb['loss'],  rnn['loss'], True),  d(kgru['loss'], rnn['loss'], True)],
    ["Accuracy",   f"{rnn['acc']:.4f}",   f"{kan['acc']:.4f}",   f"{hyb['acc']:.4f}",   f"{kgru['acc']:.4f}",   d(kan['acc'],   rnn['acc']),          d(hyb['acc'],   rnn['acc']),          d(kgru['acc'],  rnn['acc'])],
    ["HR@10",      f"{rnn['hr']:.4f}",    f"{kan['hr']:.4f}",    f"{hyb['hr']:.4f}",    f"{kgru['hr']:.4f}",    d(kan['hr'],    rnn['hr']),            d(hyb['hr'],    rnn['hr']),            d(kgru['hr'],   rnn['hr'])],
    ["NDCG@10",    f"{rnn['ndcg']:.4f}",  f"{kan['ndcg']:.4f}",  f"{hyb['ndcg']:.4f}",  f"{kgru['ndcg']:.4f}",  d(kan['ndcg'],  rnn['ndcg']),          d(hyb['ndcg'],  rnn['ndcg']),          d(kgru['ndcg'], rnn['ndcg'])],
    ["Time(s)",    f"{rnn['time']:.1f}",  f"{kan['time']:.1f}",  f"{hyb['time']:.1f}",  f"{kgru['time']:.1f}",  f"{kan['time']-rnn['time']:+.1f}",     f"{hyb['time']-rnn['time']:+.1f}",     f"{kgru['time']-rnn['time']:+.1f}"],
    ["Params",     f"{rnn['params']:,}",  f"{kan['params']:,}",  f"{hyb['params']:,}",  f"{kgru['params']:,}",  f"{kan['params']-rnn['params']:+,}",   f"{hyb['params']-rnn['params']:+,}",   f"{kgru['params']-rnn['params']:+,}"],
]

headers = ["Metric", "RNN", "KAN", "Hybrid", "KAN-GRU", "KAN vs RNN", "Hybrid vs RNN", "KAN-GRU vs RNN"]

print("\n" + "=" * 90)
print("  FINAL RESULTS:  RNN  vs  KAN  vs  Hybrid  vs  KAN-GRU  (YooChoose)")
print("=" * 90)
print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

scores = {n: 0 for n, _ in MODELS}
for m in ["hr", "ndcg", "acc"]:
    best = max(results, key=lambda n: results[n][m])
    scores[best] += 1

winner = max(scores, key=scores.get)
print(f"\n  Winner: {winner}")
print(f"  Wins -> " + "  ".join(f"{n}: {scores[n]}" for n, _ in MODELS))
print()
