from pathlib import Path

# Colab runtime image directory
IMG_DIR = "/content/celebahq256/all"

# Existing Drive split files
ACTIVE_SPLIT_DIR = "/content/drive/MyDrive/FYP_FaceCompletion/splits_active"

# Persistent outputs on Drive
RESULT_DIR = "/content/drive/MyDrive/FYP_FaceCompletion/results"

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
BASE_CHANNELS = 32
NUM_WORKERS = 2
SEED = 42

CHECKPOINT_DIR = str(Path(RESULT_DIR) / "checkpoints")
PREVIEW_DIR = str(Path(RESULT_DIR) / "previews")
