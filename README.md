# Session-Based Recommendation: GRU vs KAN-GRU

Does putting Kolmogorov-Arnold Network (KAN) layers inside a GRU help next-item recommendation? This PyTorch project trains a standard GRU recommender and a **KAN-GRU** (a GRU whose gate projections are B-spline KAN layers, written from scratch) on the same YooChoose clickstream data, with the same pipeline, and compares them on HR@K, NDCG@K, MRR@K and accuracy.

## Results

Latest shared run (`training_logs/training_history_latest.txt`): about 14.2K sessions after filtering (11.3K train, 2.8K validation), giving 47.5K training and 10.8K validation samples over ~10K items, with an input window of 3 clicks. That run had position embeddings and attention readout switched on, LR 0.007 and label smoothing 0.01. The committed `config.py` has since moved on, so `python compare.py` won't reproduce these exact numbers.

| Metric | Vanilla GRU | KAN-GRU |
|---|---|---|
| HR@10 | 0.4487 | **0.4703** |
| NDCG@10 | 0.2951 | **0.3115** |
| MRR@10 | 0.2469 | **0.2616** |
| Accuracy | 0.1600 | **0.1699** |
| Training time | 108 s | 72 s |
| Parameters | 2.22M | 2.32M |

KAN-GRU came out ahead on every metric with about 98K extra parameters. It peaked at epoch 9 and then began to overfit, so early stopping matters. Two earlier runs are in `training_logs/` too, and both point the same way.

## How it works

- **Data** (`data_loader.py`): reads `yoochoose-clicks.dat`, keeps sessions of at least 4 clicks (up to `MAX_SESSIONS`), and turns each session into sliding windows of 3 clicks → next item. Sessions are split 80/20, and validation targets never seen in training are dropped.
- **Vanilla GRU** (`rnn_model.py`): embedding → 2-layer GRU → output layer.
- **KAN-GRU** (`kan_gru.py`, `kan_gru_model.py`): the GRU's update, reset and candidate gates use `KANLinear` (`kan_layer.py`: learnable B-spline functions plus a SiLU base path) instead of `nn.Linear`. The output layer is tied to the item embeddings, with a last-item tower.
- **Training** (`trainer.py`): AdamW, LR-on-plateau and early stopping on validation NDCG, and KAN activation and entropy regularisation. It can train with cross-entropy, BPR or TOP1 loss (`losses.py`).
- **Metrics** (`metrics.py`): HR, Recall, NDCG and MRR at 10 and 20, plus top-1 accuracy.

Every switch (KAN grid size, readout variants, Markov transition prior, repeat boost, loss types, split mode) is in `config.py`.

## Earlier three-way study

An earlier version of this project also compared a standalone KAN recommender (`kan_model.py`: embedding + positional encoding → KAN → attention pooling) and a hybrid (`hybrid_model.py`: GRU output refined by a KAN layer). Those models are kept for reference. `compare.py` now runs the GRU vs KAN-GRU comparison only.

## Run it

```bash
git clone https://github.com/devSaksham/session-based-recommendation-system-latest.git
cd session-based-recommendation-system-latest
pip install -r requirements.txt
python compare.py
```

The data loader looks for `yoochoose-clicks.dat` in this order: the path in `YOOCHOOSE_CLICKS`, `Dataset/` or `data/`. If it finds nothing, it downloads the file from the RecSysDatasets mirror. If the download also fails, it generates synthetic sessions so the pipeline still runs, but those numbers mean nothing.

`compare.py` trains both models once for each loss type in `LOSS_TYPES`. It then prints a comparison table and writes:

- `plots/`: a 3×4 grid of training curves
- `checkpoints/`: the best weights for each model
- `training_logs/training_history_<timestamp>.txt`: the full console log

`python dataset_diagnostics.py` plots session-length distributions for the current split settings.

## Files

```
compare.py             run the experiment
config.py              hyperparameters and feature switches
data_loader.py         loading, windowing, splitting, Markov prior
rnn_model.py           vanilla GRU model
kan_model.py           standalone KAN model (earlier study)
hybrid_model.py        GRU + KAN hybrid (earlier study)
kan_layer.py           B-spline KAN layer
kan_gru.py             KAN-GRU cell and layer
kan_gru_model.py       KAN-GRU recommender
trainer.py             training loop, early stopping, KAN regularisation
losses.py              cross-entropy, BPR, TOP1
metrics.py             HR / Recall / NDCG / MRR / accuracy
model_inspector.py     prints each model's training setup
dataset_diagnostics.py session statistics and plots
training_logs/         logs from earlier runs
```

## License

MIT
