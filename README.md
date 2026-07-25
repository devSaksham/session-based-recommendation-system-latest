# YooChoose: RNN vs KAN vs Hybrid (3-Way Comparison)

A PyTorch research project comparing three architectures for **session-based recommendation** on the [YOOCHOOSE](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015) e-commerce clickstream dataset: a classic GRU-based RNN, a from-scratch Kolmogorov-Arnold Network (KAN), and a Hybrid model that lets a KAN layer refine GRU output. All three are trained and evaluated under a shared pipeline and benchmarked on HR@10, NDCG@10, and accuracy.

## Project Structure
```
├── compare.py          ← ★ Run this
├── config.py           ← All hyperparameters
├── data_loader.py      ← YooChoose download + preprocessing
├── trainer.py          ← Shared train/eval loop
├── metrics.py          ← HR@10, NDCG@10, accuracy
├── requirements.txt
└── models/
    ├── rnn_model.py    ← Embedding → GRU → FC
    ├── kan_model.py    ← Embedding → KAN → Attention Pool → FC
    ├── hybrid_model.py ← Embedding → GRU → KAN → FC  ★ new
    └── kan_layer.py    ← B-spline KAN layer (from scratch)
```

## How to Run
```bash
pip install -r requirements.txt
python compare.py
```

## What Each Model Does
| Model   | Architecture |
|---------|-------------|
| RNN     | Embedding → 2-layer GRU → last hidden → FC |
| KAN     | Embedding + PosEnc → KAN layer → attention pool → FC |
| Hybrid  | Embedding → GRU → **then KAN refines GRU output** → FC |

## Output
- `plots/comparison.png` — 6-panel plot (loss, acc, HR, NDCG, bar chart)
- `checkpoints/` — saved model weights
- Terminal — formatted 3-way comparison table with winner
