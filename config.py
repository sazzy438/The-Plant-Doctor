"""Central configuration for Plant Doctor.

Everything that the application, the training pipeline and the tests share
lives here so that there is exactly one source of truth for paths,
preprocessing constants and safety thresholds.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

MODEL_PATH = Path(os.environ.get("PLANT_DOCTOR_MODEL", ARTIFACTS_DIR / "plant_doctor_efficientnet_b0.pt"))
CLASS_MAP_PATH = ARTIFACTS_DIR / "class_mapping.json"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
PREPROCESSING_CONFIG_PATH = ARTIFACTS_DIR / "preprocessing_config.json"
TRAINING_METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"
TRAINING_LOG_PATH = ARTIFACTS_DIR / "training_log.json"
EVALUATION_REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.md"
EVALUATION_METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
OOD_STATS_PATH = ARTIFACTS_DIR / "ood_statistics.npz"
DISEASE_INFO_PATH = DATA_DIR / "disease_information.json"

# --------------------------------------------------------------------------- #
# Model / preprocessing (must match training exactly)
# --------------------------------------------------------------------------- #
ARCHITECTURE = "efficientnet_b0"
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 38

# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #
SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MAX_FILE_SIZE_MB = 25
MIN_IMAGE_SIDE = 64          # pixels; anything smaller cannot show leaf detail
MAX_IMAGE_SIDE = 8000        # pixels; larger images are downscaled before use
MAX_PIXELS = 60_000_000      # decompression-bomb guard
MIN_CROP_SIDE = 32           # pixels in the original image

# --------------------------------------------------------------------------- #
# Abstention / safety thresholds (all configurable)
# --------------------------------------------------------------------------- #
# Softmax probability below which the app refuses to name a condition.
MIN_CONFIDENCE = float(os.environ.get("PLANT_DOCTOR_MIN_CONFIDENCE", 0.60))
# Variance of the Laplacian on a 512px grayscale version; below = too blurry.
BLUR_THRESHOLD = float(os.environ.get("PLANT_DOCTOR_BLUR_THRESHOLD", 25.0))
# Feature-space novelty score: fraction of the calibrated 99th percentile.
# Values > 1.0 mean "more unusual than 99% of validation leaf images".
OOD_SCORE_LIMIT = float(os.environ.get("PLANT_DOCTOR_OOD_LIMIT", 1.0))
# Fraction of (near) green/vegetation pixels below which we warn the user.
MIN_VEGETATION_FRACTION = float(os.environ.get("PLANT_DOCTOR_MIN_VEG", 0.05))
# Below this fraction there is essentially no plant-coloured pixel at all (documents,
# sky, grey walls); validation leaves never fall below ~1.3 %, so this is a hard stop.
MIN_VEGETATION_HARD = float(os.environ.get("PLANT_DOCTOR_MIN_VEG_HARD", 0.01))

ABSTAIN_MESSAGE = (
    "We couldn't confidently identify a plant condition from this image. "
    "Please upload a clearer photograph showing the leaf."
)

# --------------------------------------------------------------------------- #
# Product
# --------------------------------------------------------------------------- #
APP_NAME = "Plant Doctor"
TAGLINE = "Understand your plants. Protect your harvest."
VERSION = "1.0.0"
HISTORY_LIMIT = 12
