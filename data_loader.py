import os, gzip, shutil
from pathlib import Path

import requests, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from config import *

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2] if len(BASE_DIR.parents) >= 3 else BASE_DIR
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "yoochoose-clicks.dat"
LOCAL_DATA_CANDIDATES = [
    Path(os.environ["YOOCHOOSE_CLICKS"]) if os.environ.get("YOOCHOOSE_CLICKS") else None,
    PROJECT_ROOT / "Dataset" / "yoochoose-clicks.dat",
    BASE_DIR / "data" / "yoochoose-clicks.dat",
]
GZ_URLS = [
    "https://github.com/RUCAIBox/RecSysDatasets/raw/master/dataset/yoochoose/yoochoose-clicks.dat.gz",
]


def _synthetic():
    DATA_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    rows = []
    for sid in range(1, (MAX_SESSIONS or 40000) + 1):
        length = np.random.randint(3, 15)
        theme = np.random.randint(0, 20)
        start = np.random.randint(0, SYNTHETIC_ITEMS)
        items = []
        for i in range(length):
            # Learnable fallback: sessions follow repeatable transition motifs
            # with a little noise, instead of being independent random IDs.
            if i and np.random.rand() < 0.15:
                item = np.random.randint(0, SYNTHETIC_ITEMS)
            else:
                item = (start + theme * 17 + i * (theme + 3)) % SYNTHETIC_ITEMS
            items.append(item + 1)
        for i, item in enumerate(items):
            rows.append([sid, i, item, "S"])
    df = pd.DataFrame(rows, columns=["SessionId", "Timestamp", "ItemId", "Category"])
    df.to_csv(DATA_FILE, index=False, header=False)
    print("Patterned synthetic data created.")


def _best_local_file():
    candidates = LOCAL_DATA_CANDIDATES
    if isinstance(candidates, (str, os.PathLike)):
        candidates = [candidates]

    normalized = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate).strip())
        normalized.append(path)

    existing = [p for p in normalized if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_size)


def download():
    DATA_DIR.mkdir(exist_ok=True)
    local_file = _best_local_file()
    if local_file:
        print(f"Using local YooChoose file: {local_file}")
        return local_file

    print("Downloading YooChoose...")
    last_error = None
    for url in GZ_URLS:
        try:
            gz = DATA_FILE.with_suffix(DATA_FILE.suffix + ".gz")
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with open(gz, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            with gzip.open(gz, "rb") as fi, open(DATA_FILE, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(gz)
            print("Download done.")
            return DATA_FILE
        except Exception as e:
            last_error = e

    print(f"Download failed ({last_error}). Using patterned synthetic data.")
    _synthetic()
    return DATA_FILE


def _read_clicks(path):
    usecols = ["SessionId", "ItemId"]
    names = ["SessionId", "Timestamp", "ItemId", "Category"]

    if not MAX_SESSIONS:
        return pd.read_csv(path, names=names, usecols=usecols,
                           dtype={"SessionId": int, "ItemId": int}).dropna()

    chunks, seen = [], set()
    for chunk in pd.read_csv(path, names=names, usecols=usecols,
                             dtype={"SessionId": int, "ItemId": int},
                             chunksize=1_000_000):
        chunk = chunk.dropna()
        before = len(seen)
        for session_id in chunk["SessionId"].unique():
            if session_id in seen:
                continue
            if len(seen) >= MAX_SESSIONS:
                break
            seen.add(session_id)

        selected = chunk[chunk["SessionId"].isin(seen)]
        if not selected.empty:
            chunks.append(selected)
        if len(seen) >= MAX_SESSIONS and before == len(seen) and selected.empty:
            break

    df = pd.concat(chunks, ignore_index=True)
    keep = df["SessionId"].drop_duplicates().head(MAX_SESSIONS)
    return df[df["SessionId"].isin(keep)]


def _sequence_stats(split):
    targets = [target for _, target in split]
    input_items = {item for seq, _ in split for item in seq}
    return {
        "samples": len(split),
        "target_items": set(targets),
        "input_items": input_items,
    }


def _print_split_report(train_sessions, test_sessions, train, test):
    train_stats = _sequence_stats(train)
    test_stats = _sequence_stats(test)
    train_targets = train_stats["target_items"]
    test_targets = test_stats["target_items"]
    target_overlap = len(train_targets & test_targets)
    target_coverage = target_overlap / max(1, len(test_targets))

    print(
        f"  split_mode={SPLIT_MODE}  train_ratio={float(TRAIN_RATIO):.3f}  "
        f"train_sessions={len(train_sessions)}  val_sessions={len(test_sessions)}"
    )
    print(
        "  train_samples={train_samples}  val_samples={val_samples}  "
        "avg_train_samples/session={avg_train:.2f}  avg_val_samples/session={avg_val:.2f}".format(
            train_samples=len(train),
            val_samples=len(test),
            avg_train=len(train) / max(1, len(train_sessions)),
            avg_val=len(test) / max(1, len(test_sessions)),
        )
    )
    print(
        "  unique_train_targets={train_targets}  unique_val_targets={val_targets}  "
        "val_target_seen_in_train={coverage:.1%}".format(
            train_targets=len(train_targets),
            val_targets=len(test_targets),
            coverage=target_coverage,
        )
    )


def _filter_unseen_targets(train, test):
    train_targets = {target for _, target in train}
    filtered = [(seq, target) for seq, target in test if target in train_targets]
    dropped = len(test) - len(filtered)
    if dropped:
        print(
            "  filtered_unseen_val_targets={dropped}  kept_val_samples={kept}  "
            "cold_start_removed={pct:.1%}".format(
                dropped=dropped,
                kept=len(filtered),
                pct=dropped / max(1, len(test)),
            )
        )
    return filtered


def _split_cutoff(n_sessions):
    try:
        train_ratio = float(TRAIN_RATIO)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TRAIN_RATIO must be a number between 0 and 1, got {TRAIN_RATIO!r}") from exc

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            f"TRAIN_RATIO must be between 0 and 1. Current value: {TRAIN_RATIO!r}. "
            "Use TRAIN_RATIO = 0.8 for an 80/20 train/validation split."
        )

    if n_sessions < 2:
        raise ValueError(f"Need at least 2 valid sessions after filtering, got {n_sessions}.")

    cut = int(n_sessions * train_ratio)
    if cut <= 0 or cut >= n_sessions:
        raise ValueError(
            f"TRAIN_RATIO={train_ratio} with {n_sessions} sessions creates an empty split. "
            "Use a value such as 0.8."
        )
    return cut


def load_data():
    data_file = download()
    df = _read_clicks(data_file)

    lens   = df.groupby("SessionId")["ItemId"].count()
    min_required_len = max(MIN_SEQ_LEN, INPUT_WINDOW + 1)
    df     = df[df["SessionId"].isin(lens[lens >= min_required_len].index)]

    enc          = LabelEncoder()
    df["ItemId"] = enc.fit_transform(df["ItemId"]) + 1   # 0 = pad
    n_items      = len(enc.classes_) + 1

    sessions = []
    for _, g in df.groupby("SessionId", sort=False):
        items = g["ItemId"].tolist()
        session_seqs = []
        for start in range(0, len(items) - INPUT_WINDOW):
            end = start + INPUT_WINDOW
            session_seqs.append((items[start:end], items[end]))
        if session_seqs:
            sessions.append(session_seqs)

    if SPLIT_MODE == "random_session":
        np.random.seed(SEED)
        np.random.shuffle(sessions)
    elif SPLIT_MODE == "temporal_session":
        pass
    else:
        raise ValueError("SPLIT_MODE must be 'random_session' or 'temporal_session'")

    cut_sessions = _split_cutoff(len(sessions))
    train_sessions = sessions[:cut_sessions]
    test_sessions = sessions[cut_sessions:]
    train = [seq for sess in train_sessions for seq in sess]
    test = [seq for sess in test_sessions for seq in sess]
    if FILTER_UNSEEN_VAL_TARGETS:
        test = _filter_unseen_targets(train, test)
    print(f"  n_items={n_items}  train={len(train)}  test={len(test)}")
    _print_split_report(train_sessions, test_sessions, train, test)
    return train, test, n_items


class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        s, t = self.seqs[i]
        return torch.tensor(s, dtype=torch.long), torch.tensor(t - 1, dtype=torch.long)


def collate(batch):
    seqs, tgts = zip(*batch)
    pad = pad_sequence(seqs, batch_first=True, padding_value=0)
    if pad.size(1) > MAX_SEQ_LEN:
        pad = pad[:, -MAX_SEQ_LEN:]
    return pad, torch.stack(tgts)


def get_loaders(train, test):
    kw = dict(batch_size=BATCH_SIZE, collate_fn=collate)
    return (DataLoader(SeqDataset(train), shuffle=True,  **kw),
            DataLoader(SeqDataset(test),  shuffle=False, **kw))
