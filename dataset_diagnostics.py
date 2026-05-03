"""
Dataset diagnostics for the session recommendation experiment.

Run:
    python dataset_diagnostics.py

Outputs:
    plots/session_length_histogram.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import INPUT_WINDOW, MIN_SEQ_LEN, SEED, SPLIT_MODE, TRAIN_RATIO
from data_loader import download, _read_clicks


BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def make_sequences(items):
    sequences = []
    for start in range(0, len(items) - INPUT_WINDOW):
        end = start + INPUT_WINDOW
        sequences.append((items[start:end], items[end]))
    return sequences


def split_sessions(sessions):
    sessions = list(sessions)
    if SPLIT_MODE == "random_session":
        rng = np.random.default_rng(SEED)
        rng.shuffle(sessions)
    elif SPLIT_MODE == "temporal_session":
        pass
    else:
        raise ValueError("SPLIT_MODE must be 'random_session' or 'temporal_session'")

    train_end = int(len(sessions) * TRAIN_RATIO)
    remaining = len(sessions) - train_end
    val_count = remaining // 2
    val_end = train_end + val_count
    return sessions[:train_end], sessions[train_end:val_end], sessions[val_end:]


def flatten_labels(session_rows):
    labels = []
    sample_count = 0
    for _, seqs in session_rows:
        sample_count += len(seqs)
        labels.extend(target for _, target in seqs)
    return labels, sample_count


def main():
    data_file = download()
    df = _read_clicks(data_file)

    raw_lengths = df.groupby("SessionId")["ItemId"].count()
    min_required_len = max(MIN_SEQ_LEN, INPUT_WINDOW + 1)
    filtered_df = df[df["SessionId"].isin(raw_lengths[raw_lengths >= min_required_len].index)]
    lengths = filtered_df.groupby("SessionId")["ItemId"].count()

    sessions = []
    for session_id, group in filtered_df.groupby("SessionId", sort=False):
        items = group["ItemId"].tolist()
        seqs = make_sequences(items)
        if seqs:
            sessions.append((session_id, seqs))

    train_sessions, val_sessions, test_sessions = split_sessions(sessions)
    y_train, train_samples = flatten_labels(train_sessions)
    y_val, val_samples = flatten_labels(val_sessions)
    y_test, test_samples = flatten_labels(test_sessions)

    hist_path = PLOT_DIR / "session_length_histogram.png"
    counts_by_length = lengths.value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    plt.bar(counts_by_length.index, counts_by_length.values, width=0.85)
    plt.title("Count of Sessions vs Session Length")
    plt.xlabel("Session Length")
    plt.ylabel("Count of Sessions")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    print("\nDataset diagnostics")
    print("=" * 70)
    print(f"Data file: {data_file}")
    print(f"Total sessions read: {raw_lengths.size:,}")
    print(f"Sessions after minimum length={min_required_len}: {lengths.size:,}")
    print(f"1. Number of unique items: {filtered_df['ItemId'].nunique():,}")
    print(f"2. Minimum length of any session: {int(lengths.min()):,}")
    print(f"3. Maximum length of any session: {int(lengths.max()):,}")
    print(f"4. Histogram saved to: {hist_path}")
    print(f"5. Window size used as model input: {INPUT_WINDOW}")
    print("   Note: fixed-length sliding windows are used; each next item is the label.")
    print("6. Unique label item counts:")
    print(f"   y_train: {len(set(y_train)):,} unique labels from {train_samples:,} samples")
    print(f"   y_val:   {len(set(y_val)):,} unique labels from {val_samples:,} samples")
    print(f"   y_test:  {len(set(y_test)):,} unique labels from {test_samples:,} samples")
    print()
    print("Split details")
    print("-" * 70)
    print(f"Split mode: {SPLIT_MODE}")
    print(f"Session split: train={len(train_sessions):,}, val={len(val_sessions):,}, test={len(test_sessions):,}")
    print(f"Validation labels seen in train: {len(set(y_val) & set(y_train)) / max(1, len(set(y_val))):.1%}")
    print(f"Test labels seen in train:       {len(set(y_test) & set(y_train)) / max(1, len(set(y_test))):.1%}")


if __name__ == "__main__":
    main()
