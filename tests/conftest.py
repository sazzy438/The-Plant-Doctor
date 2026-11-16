"""Shared fixtures. Tests use the real trained checkpoint when present; otherwise a
temporary untrained checkpoint is created so that the *plumbing* can still be
verified (those tests are marked and reported accordingly)."""
from __future__ import annotations

import glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

DATASET_ROOT = Path(os.environ.get("PLANTVILLAGE_ROOT", ROOT.parent / "datasets" / "plantvillage")) / "raw" / "color"


def dataset_image(class_name: str) -> Path | None:
    files = sorted(glob.glob(str(DATASET_ROOT / class_name / "*")))
    return Path(files[0]) if files else None


@pytest.fixture(scope="session")
def trained_model_available(predictor) -> bool:
    """True only for a checkpoint that went through at least one training epoch."""
    return int(predictor.meta.get("epoch", 0) or 0) >= 1


@pytest.fixture(scope="session")
def model_path(tmp_path_factory) -> Path:
    if config.MODEL_PATH.exists():
        return config.MODEL_PATH
    # Fallback: untrained head so the pipeline can still be exercised.
    import json
    from models.model import build_model, save_checkpoint
    names = json.loads((config.ARTIFACTS_DIR / "class_mapping.json").read_text())["index_to_class"]
    names = [names[str(i)] for i in range(len(names))]
    p = tmp_path_factory.mktemp("model") / "untrained.pt"
    save_checkpoint(build_model(len(names), pretrained=False), p, names, extra={"epoch": 0})
    return p


@pytest.fixture(scope="session")
def predictor(model_path):
    from models.inference import Predictor
    return Predictor(model_path)


@pytest.fixture(scope="session")
def healthy_leaf() -> Image.Image:
    p = dataset_image("Tomato___healthy")
    if p is None:
        pytest.skip("PlantVillage dataset not available")
    return Image.open(p).convert("RGB")


@pytest.fixture(scope="session")
def diseased_leaf() -> Image.Image:
    p = dataset_image("Tomato___Early_blight")
    if p is None:
        pytest.skip("PlantVillage dataset not available")
    return Image.open(p).convert("RGB")


@pytest.fixture(scope="session")
def diseased_leaf_path() -> Path:
    p = dataset_image("Potato___Late_blight")
    if p is None:
        pytest.skip("PlantVillage dataset not available")
    return p


@pytest.fixture
def blurry_leaf(diseased_leaf) -> Image.Image:
    return diseased_leaf.filter(ImageFilter.GaussianBlur(radius=12))


@pytest.fixture
def unrelated_image() -> Image.Image:
    """A synthetic 'street scene': grey asphalt with white lane markings and a blue sky."""
    rng = np.random.default_rng(0)
    arr = np.zeros((400, 600, 3), dtype=np.uint8)
    arr[:180] = (135, 170, 220)                     # sky
    arr[180:] = (90, 90, 95)                        # asphalt
    arr[180:] = np.clip(arr[180:] + rng.integers(-15, 15, arr[180:].shape), 0, 255)
    arr[280:290, 50:200] = 240; arr[280:290, 300:450] = 240  # lane markings
    arr[60:120, 420:560] = (235, 235, 235)          # cloud
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def multi_leaf(healthy_leaf, diseased_leaf) -> Image.Image:
    w, h = healthy_leaf.size
    canvas = Image.new("RGB", (w * 2, h * 2), (245, 245, 240))
    canvas.paste(healthy_leaf, (0, 0)); canvas.paste(diseased_leaf, (w, 0))
    canvas.paste(diseased_leaf.rotate(90), (0, h)); canvas.paste(healthy_leaf.rotate(180), (w, h))
    return canvas


@pytest.fixture
def tmp_image_file(tmp_path):
    def _make(name: str, image: Image.Image | None = None, fmt: str | None = None, content: bytes | None = None) -> Path:
        p = tmp_path / name
        if content is not None:
            p.write_bytes(content)
        else:
            image.save(p, format=fmt)
        return p
    return _make
