import os, gzip, shutil, requests, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from config import *

DATA_DIR  = "./data"
DATA_FILE = os.path.join(DATA_DIR, "yoochoose-clicks.dat")
GZ_URL    = "https://github.com/RUCAIBox/RecSysDatasets/raw/master/dataset/yoochoose/yoochoose-clicks.dat.gz"


def _synthetic():
    np.random.seed(SEED)
    rows = []
    for sid in range(1, (MAX_SESSIONS or 40000) + 1):
        length = np.random.randint(3, 15)
        items  = np.random.choice(5000, length, replace=False) + 1
        for i, item in enumerate(items):
            rows.append([sid, i, item, "S"])
    df = pd.DataFrame(rows, columns=["SessionId", "Timestamp", "ItemId", "Category"])
    df.to_csv(DATA_FILE, index=False, header=False)
    print("Synthetic data created.")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        return
    print("Downloading YooChoose...")
    try:
        gz = DATA_FILE + ".gz"
        r  = requests.get(GZ_URL, timeout=60, stream=True)
        r.raise_for_status()
        with open(gz, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        with gzip.open(gz, "rb") as fi, open(DATA_FILE, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(gz)
        print("Download done.")
    except Exception as e:
        print(f"Download failed ({e}). Using synthetic data.")
        _synthetic()


def load_data():
    download()
    df = pd.read_csv(DATA_FILE, names=["SessionId", "Timestamp", "ItemId", "Category"],
                     dtype={"SessionId": int, "ItemId": int})[["SessionId", "ItemId"]].dropna()

    if MAX_SESSIONS:
        sess = df["SessionId"].unique()[:MAX_SESSIONS]
        df   = df[df["SessionId"].isin(sess)]

    lens   = df.groupby("SessionId")["ItemId"].count()
    df     = df[df["SessionId"].isin(lens[lens >= MIN_SEQ_LEN].index)]

    enc          = LabelEncoder()
    df["ItemId"] = enc.fit_transform(df["ItemId"]) + 1   # 0 = pad
    n_items      = len(enc.classes_) + 1

    seqs = []
    for _, g in df.groupby("SessionId"):
        items = g["ItemId"].tolist()
        for end in range(2, len(items) + 1):
            s = items[max(0, end - MAX_SEQ_LEN - 1): end]
            seqs.append((s[:-1], s[-1]))

    np.random.seed(SEED)
    np.random.shuffle(seqs)
    cut   = int(len(seqs) * TRAIN_RATIO)
    train, test = seqs[:cut], seqs[cut:]
    print(f"  n_items={n_items}  train={len(train)}  test={len(test)}")
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
