import torch

MAX_SESSIONS   = 40000
MIN_SEQ_LEN    = 3
MAX_SEQ_LEN    = 20

EMBEDDING_DIM  = 64
HIDDEN_DIM     = 128
KAN_GRID       = 5
KAN_ORDER      = 3
DROPOUT        = 0.2

BATCH_SIZE     = 256
EPOCHS         = 15
LR             = 1e-3
TRAIN_RATIO    = 0.8
TOP_K          = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
