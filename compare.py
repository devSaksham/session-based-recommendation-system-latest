"""
2-Way Comparison: vanilla GRU/RNN vs GRU with KAN layers.
Dataset: YooChoose when available, otherwise patterned synthetic fallback.
Run:  python compare.py
"""
import os
import sys
from datetime import datetime
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

from config import SEED, TOP_KS, USE_MARKOV_TRANSITION_PRIOR, MARKOV_TOP_K, LOSS_TYPE, LOSS_TYPES
from data_loader import load_data, get_loaders, build_markov_transition_prior
from rnn_model import RNNModel
from kan_gru_model import KANGRUModel
from trainer import train, run_epoch
from metrics import n_params
from losses import build_loss, loss_display_name, normalize_loss_type


torch.manual_seed(SEED)
np.random.seed(SEED)

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PLOT_DIR = BASE_DIR / "plots"
LOG_DIR = BASE_DIR / "training_logs"
CHECKPOINT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


log_path = LOG_DIR / f"training_history_{datetime.now():%Y%m%d_%H%M%S}.txt"
log_file = open(log_path, "w", encoding="utf-8")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = TeeLogger(original_stdout, log_file)
sys.stderr = TeeLogger(original_stderr, log_file)

print(f"Training log -> {log_path}")

print("\n[Data] Loading YooChoose...")
train_seqs, test_seqs, n_items = load_data()
train_loader, test_loader = get_loaders(train_seqs, test_seqs)
markov_prior = None
if USE_MARKOV_TRANSITION_PRIOR:
    markov_prior = build_markov_transition_prior(train_seqs, n_items, MARKOV_TOP_K)

MODELS = [
    ("Vanilla-GRU", RNNModel),
    ("KAN-GRU", KANGRUModel),
]


def _configured_loss_types():
    values = LOSS_TYPES if LOSS_TYPES else (LOSS_TYPE,)
    if isinstance(values, str):
        values = (values,)
    return tuple(dict.fromkeys(normalize_loss_type(value) for value in values))


LOSS_RUNS = _configured_loss_types()
results = {}
for loss_type in LOSS_RUNS:
    loss_name = loss_display_name(loss_type)
    for model_name, ModelClass in MODELS:
        run_name = f"{model_name} | {loss_name}"
        model = ModelClass(n_items)
        if markov_prior is not None and hasattr(model, "set_markov_transition_prior"):
            model.set_markov_transition_prior(*markov_prior)
        hist, elapsed = train(model, train_loader, test_loader, run_name, loss_type=loss_type)
        checkpoint_name = f"{model_name.lower().replace('-', '_')}_{loss_type}.pt"
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name
        torch.save(model.state_dict(), checkpoint_path)

        crit = build_loss(loss_type)
        train_eval_values = run_epoch(model, train_loader, crit)
        eval_values = run_epoch(model, test_loader, crit)
        tr_loss, tr_acc = train_eval_values[:2]
        vl, va = eval_values[:2]
        cutoff_metrics = {}
        offset = 2
        for k in TOP_KS:
            cutoff_metrics[k] = dict(
                hr=eval_values[offset],
                ndcg=eval_values[offset + 1],
                mrr=eval_values[offset + 2],
                recall=eval_values[offset + 3],
            )
            offset += 4
        primary = cutoff_metrics[TOP_KS[0]]

        results[run_name] = dict(
            model=model_name,
            loss_type=loss_type,
            loss_name=loss_name,
            hist=hist,
            time=elapsed,
            params=n_params(model),
            tr_loss=tr_loss,
            tr_acc=tr_acc,
            loss=vl,
            acc=va,
            hr=primary["hr"],
            recall=primary["recall"],
            ndcg=primary["ndcg"],
            mrr=primary["mrr"],
            cutoff_metrics=cutoff_metrics,
            checkpoint=checkpoint_path,
        )

COLORS = {
    "Vanilla-GRU": "#E74C3C",
    "KAN-GRU": "#2E86AB",
}
LOSS_STYLES = {
    "cross_entropy": "-",
    "bpr": "--",
    "top1": ":",
}
LOSS_MARKERS = {
    "cross_entropy": "o",
    "bpr": "s",
    "top1": "^",
}

fig, axes = plt.subplots(3, 4, figsize=(24, 14))
fig.suptitle("Session Recommendation: Vanilla GRU vs KAN-GRU", fontsize=15, fontweight="bold")
axes = axes.flatten()

panels = [
    ("tr_loss", "Training Loss", "TrLoss"),
    ("tr_acc", "Training Accuracy", "TrAcc"),
    ("vl_loss", "Validation Loss", "VlLoss"),
    ("vl_acc", "Validation Accuracy", "VlAcc"),
    ("vl_hr10", "HR@10", "HR@10"),
    ("vl_recall10", "Recall@10", "Recall@10"),
    ("vl_mrr10", "MRR@10", "MRR@10"),
    ("vl_ndcg10", "NDCG@10", "NDCG@10"),
    ("vl_hr20", "HR@20", "HR@20"),
    ("vl_recall20", "Recall@20", "Recall@20"),
    ("vl_mrr20", "MRR@20", "MRR@20"),
    ("vl_ndcg20", "NDCG@20", "NDCG@20"),
]

for ax, (key, title, ylabel) in zip(axes, panels):
    for name, result in results.items():
        epochs = range(1, len(result["hist"][key]) + 1)
        ax.plot(
            epochs,
            result["hist"][key],
            label=name,
            color=COLORS[result["model"]],
            linestyle=LOSS_STYLES.get(result["loss_type"], "-"),
            marker=LOSS_MARKERS.get(result["loss_type"], "o"),
            ms=4,
            linewidth=1.8,
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = PLOT_DIR / "comparison.png"
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved -> {plot_path}")


rows = []
for loss_type in LOSS_RUNS:
    loss_name = loss_display_name(loss_type)
    for model_name, _ in MODELS:
        run_name = f"{model_name} | {loss_name}"
        result = results[run_name]
        row = [
            model_name,
            loss_name,
            f"{result['tr_loss']:.4f}",
            f"{result['tr_acc']:.4f}",
            f"{result['loss']:.4f}",
            f"{result['acc']:.4f}",
        ]
        for k in TOP_KS:
            metrics = result["cutoff_metrics"][k]
            row.extend([
                f"{metrics['hr']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['mrr']:.4f}",
                f"{metrics['ndcg']:.4f}",
            ])
        row.extend([f"{result['time']:.1f}", f"{result['params']:,}"])
        rows.append(row)

headers = ["Model", "Loss", "TrLoss", "TrAcc", "VlLoss", "VlAcc"]
for k in TOP_KS:
    headers.extend([f"HR@{k}", f"Recall@{k}", f"MRR@{k}", f"NDCG@{k}"])
headers.extend(["Time(s)", "Params"])

print("\n" + "=" * 78)
print("  FINAL RESULTS:  Model and Loss Comparison")
print("=" * 78)
print(tabulate(rows, headers=headers, tablefmt="grid"))

rank_k = 20 if 20 in TOP_KS else TOP_KS[-1]
best_hr_name, best_hr = max(
    results.items(), key=lambda item: item[1]["cutoff_metrics"][rank_k]["recall"]
)
best_mrr_name, best_mrr = max(
    results.items(), key=lambda item: item[1]["cutoff_metrics"][rank_k]["mrr"]
)
print(
    f"\nBest Recall@{rank_k}: {best_hr_name} = "
    f"{best_hr['cutoff_metrics'][rank_k]['recall']:.4f}"
)
print(
    f"Best MRR@{rank_k}: {best_mrr_name} = "
    f"{best_mrr['cutoff_metrics'][rank_k]['mrr']:.4f}"
)
print("\nCheckpoints:")
for name, result in results.items():
    print(f"  {name}: {result['checkpoint']}")
print()

sys.stdout = original_stdout
sys.stderr = original_stderr
log_file.close()
