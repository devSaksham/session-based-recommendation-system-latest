import torch

MAX_SESSIONS   = 40000
MIN_SEQ_LEN    = 4
INPUT_WINDOW   = 3
MAX_SEQ_LEN    = 20
SYNTHETIC_ITEMS = 1000

EMBEDDING_DIM  = 64
HIDDEN_DIM     = 128
RNN_LAYERS     = 2
KAN_GRU_LAYERS = 1
KAN_GRID       = 5
KAN_ORDER      = 3
KAN_ON_GATES   = False
KAN_REG_ACTIVATION = 1e-5
KAN_REG_ENTROPY    = 1e-5
DROPOUT        = 0.2

BATCH_SIZE     = 256
EPOCHS         = 100
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTHING = 0.02
MIN_DELTA      = 1e-4
MONITOR_METRIC = "vl_ndcg"
MONITOR_MODE   = "max"
LR_PATIENCE    = 3
EARLY_STOPPING_PATIENCE = 8
TRAIN_RATIO    = 0.8
SPLIT_MODE     = "random_session"  # random_session or temporal_session
FILTER_UNSEEN_VAL_TARGETS = True
TOP_K          = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
