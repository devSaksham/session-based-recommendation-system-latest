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
KAN_GRID       = 3
KAN_ORDER      = 3
KAN_ON_GATES   = True
KAN_REG_ACTIVATION = 1e-5
KAN_REG_ENTROPY    = 1e-5
KAN_USE_POSITION_EMB = False
KAN_USE_ATTENTION_READOUT = False
KAN_USE_KAN_READOUT = False
USE_MULTI_VIEW_READOUT = True
USE_TIED_OUTPUT = True
USE_LAST_ITEM_RESIDUAL = False
USE_TWO_TOWER_OUTPUT = True
TWO_TOWER_LAST_WEIGHT_INIT = 0.5
USE_REPEAT_BOOST = False
REPEAT_BOOST_INIT = 1.0
USE_REPEAT_TRANSITION_MIXTURE = False
REPEAT_MIXTURE_STRENGTH_INIT = 2.0
USE_MARKOV_TRANSITION_PRIOR = False
MARKOV_TOP_K = 20
MARKOV_PRIOR_WEIGHT_INIT = 0.5
MARKOV_PRIOR_SMOOTHING = 0.1
MARKOV_PRIOR_MAX_SCORE = 5.0
USE_UNTIED_LOGIT_RESIDUAL = False
LOGIT_RESIDUAL_INIT = 0.05
DROPOUT        = 0.3

BATCH_SIZE     = 256
EPOCHS         = 100
LR             = 3e-3
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTHING = 0.0
# Set LOSS_TYPES to one or more of: "cross_entropy", "bpr", "top1".
# Use LOSS_TYPES = ("cross_entropy",) when you only want one training run per model.
LOSS_TYPE      = "cross_entropy"
LOSS_TYPES     = ("cross_entropy", "bpr", "top1")
MIN_DELTA      = 1e-4
MONITOR_METRIC = "vl_ndcg"
MONITOR_MODE   = "max"
LR_PATIENCE    = 3
EARLY_STOPPING_PATIENCE = 6
TRAIN_RATIO    = 0.8
SPLIT_MODE     = "random_session"  # random_session or temporal_session
FILTER_UNSEEN_VAL_TARGETS = True
TOP_K          = 10
TOP_KS         = (10, 20)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
