from pathlib import Path

# Dataset configuration
DATA_MODE = "full"  # "partial" or "full"

# Colab runtime image directory
IMG_DIR = "/content/celebahq256/all"

# Existing Drive split files
ACTIVE_SPLIT_DIR = "/content/drive/MyDrive/FYP_FaceCompletion/splits_active"

# Experiment configuration
if DATA_MODE == "partial":
    EXPERIMENT_NAME = "partial_3000"
elif DATA_MODE == "full":
    EXPERIMENT_NAME = "full_dataset"
else:
    raise ValueError(f"Invalid DATA_MODE: {DATA_MODE}")

# Persistent outputs on Drive
RESULT_DIR = Path(
    "/content/drive/MyDrive/FYP_FaceCompletion/results"
) / EXPERIMENT_NAME

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
BASE_CHANNELS = 32
NUM_WORKERS = 2
SEED = 42

CHECKPOINT_DIR = str(Path(RESULT_DIR) / "checkpoints")
PREVIEW_DIR = str(Path(RESULT_DIR) / "previews")
