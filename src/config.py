from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

GENUINE_DIR = BASE_DIR / "data" / "genuine" / "chanel"
SPLIT_DIR = BASE_DIR / "data" / "split"

TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"

FAKE_HARD_DIR = BASE_DIR / "data" / "fake_hard"
BAG_DIR = BASE_DIR / "handbag" / "112"

MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODELS_DIR / "handbag_oneclass_efficientnet.h5"
CENTER_PATH = MODELS_DIR / "oneclass_center.pt"
METRICS_PATH = METRICS_DIR / "oneclass_metrics.npz"
THRESHOLD_PATH = METRICS_DIR / "oneclass_threshold.json"
ROC_PATH = OUTPUTS_DIR / "roc_curve.png"
LOSS_PATH = OUTPUTS_DIR / "loss_curve.png"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EMBEDDING_DIM = 256
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6

PERCENTILE = 97.5
FAKES_PER_BAG = 1
RANDOM_SEED = 42
