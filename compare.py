"""
2-Way Comparison: vanilla GRU/RNN vs GRU with KAN layers.
Dataset: YooChoose when available, otherwise patterned synthetic fallback.
Run:  python compare.py
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

from config import SEED
from data_loader import load_data, get_loaders
from rnn_model import RNNModel
from kan_gru_model import KANGRUModel
from trainer import train, run_epoch
from metrics import n_params


torch.manual_seed(SEED)
np.random.seed(SEED)

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PLOT_DIR = BASE_DIR / "plots"
CHECKPOINT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

print("\n[Data] Loading YooChoose...")
train_seqs, test_seqs, n_items = load_data()
train_loader, test_loader = get_loaders(train_seqs, test_seqs)

MODELS = [
    ("Vanilla-GRU", RNNModel),
    ("KAN-GRU", KANGRUModel),
]

results = {}
for name, ModelClass in MODELS:
    model = ModelClass(n_items)
    hist, elapsed = train(model, train_loader, test_loader, name)
    checkpoint_path = CHECKPOINT_DIR / f"{name.lower().replace('-', '_')}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    crit = nn.CrossEntropyLoss()
    vl, va, vhr, vndcg, vmrr = run_epoch(model, test_loader, crit)
    results[name] = dict(
        hist=hist,
        time=elapsed,
        params=n_params(model),
        loss=vl,
        acc=va,
        hr=vhr,
        ndcg=vndcg,
        mrr=vmrr,
        checkpoint=checkpoint_path,
    )

COLORS = {
    "Vanilla-GRU": "#E74C3C",
    "KAN-GRU": "#2E86AB",
}

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Session Recommendation: Vanilla GRU vs KAN-GRU", fontsize=15, fontweight="bold")
axes = axes.flatten()

panels = [
    ("tr_loss", "Training Loss", "Loss"),
    ("vl_loss", "Validation Loss", "Loss"),
    ("vl_acc", "Validation Accuracy", "Top-1 Accuracy"),
    ("vl_hr", "HR@10", "HR@10"),
    ("vl_ndcg", "NDCG@10", "NDCG@10"),
    ("vl_mrr", "MRR@10", "MRR@10"),
]

for ax, (key, title, ylabel) in zip(axes, panels):
    for name, result in results.items():
        epochs = range(1, len(result["hist"][key]) + 1)
        ax.plot(epochs, result["hist"][key], "o-", label=name,
                color=COLORS[name], ms=4, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

ax = axes[6]
names = [n for n, _ in MODELS]
hrs = [results[n]["hr"] for n in names]
bars = ax.bar(names, hrs, color=[COLORS[n] for n in names], width=0.45)
ax.bar_label(bars, fmt="%.4f", padding=3)
ax.set_title("Best-Checkpoint HR@10")
ax.set_ylabel("HR@10")
ax.set_ylim(0, max(max(hrs) * 1.3, 0.01))
ax.grid(True, alpha=0.3, axis="y")

axes[7].axis("off")

plt.tight_layout()
plot_path = PLOT_DIR / "comparison.png"
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved -> {plot_path}")


def diff(a, b, low=False):
    value = (b - a) if low else (a - b)
    return f"{'up' if value >= 0 else 'dn'} {abs(value):.4f}"


base = results["Vanilla-GRU"]
kan = results["KAN-GRU"]
rows = [
    ["Val Loss", f"{base['loss']:.4f}", f"{kan['loss']:.4f}", diff(kan["loss"], base["loss"], True)],
    ["Accuracy", f"{base['acc']:.4f}", f"{kan['acc']:.4f}", diff(kan["acc"], base["acc"])],
    ["HR@10", f"{base['hr']:.4f}", f"{kan['hr']:.4f}", diff(kan["hr"], base["hr"])],
    ["NDCG@10", f"{base['ndcg']:.4f}", f"{kan['ndcg']:.4f}", diff(kan["ndcg"], base["ndcg"])],
    ["MRR@10", f"{base['mrr']:.4f}", f"{kan['mrr']:.4f}", diff(kan["mrr"], base["mrr"])],
    ["Time(s)", f"{base['time']:.1f}", f"{kan['time']:.1f}", f"{kan['time'] - base['time']:+.1f}"],
    ["Params", f"{base['params']:,}", f"{kan['params']:,}", f"{kan['params'] - base['params']:+,}"],
]

print("\n" + "=" * 78)
print("  FINAL RESULTS:  Vanilla GRU  vs  KAN-GRU")
print("=" * 78)
print(tabulate(rows, headers=["Metric", "Vanilla-GRU", "KAN-GRU", "KAN-GRU vs Vanilla"], tablefmt="grid"))
print("\nCheckpoints:")
for name in names:
    print(f"  {name}: {results[name]['checkpoint']}")
print()
