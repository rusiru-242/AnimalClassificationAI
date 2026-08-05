import os


# =====================
# Dataset Configuration
# =====================

DATASET_PATH = "dataset"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8

VALIDATION_SPLIT = 0.2

SEED = 42


# =====================
# Training Configuration
# =====================

EPOCHS = 20

LEARNING_RATE = 0.0001


# =====================
# Model Configuration
# =====================

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "skin_tone_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)


# =====================
# Output Configuration
# =====================

OUTPUT_DIR = "outputs"


ACCURACY_GRAPH = os.path.join(
    OUTPUT_DIR,
    "accuracy.png"
)

LOSS_GRAPH = os.path.join(
    OUTPUT_DIR,
    "loss.png"
)


# =====================
# Classes
# =====================

NUM_CLASSES = 4