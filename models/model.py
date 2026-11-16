"""Model definition and checkpoint handling for Plant Doctor.

The classifier is a torchvision EfficientNet-B0, initialised from ImageNet
weights, whose final linear layer is replaced by a 38-way output layer.
The same ``build_model`` function is used for training and for inference so
that the two can never drift apart.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

import config


class ModelLoadError(RuntimeError):
    """Raised when a trained checkpoint is missing, corrupt or incompatible."""


def build_model(num_classes: int = config.NUM_CLASSES, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    """Return an EfficientNet-B0 whose classifier head has ``num_classes`` outputs."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes))
    return model


def target_layer(model: nn.Module) -> nn.Module:
    """The last convolutional feature map, used by Grad-CAM (7x7x1280 at 224px)."""
    return model.features[-1]


def save_checkpoint(model: nn.Module, path: Path, class_names: list[str], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "format": "plant-doctor-checkpoint-v1",
        "architecture": config.ARCHITECTURE,
        "image_size": config.IMAGE_SIZE,
        "mean": config.IMAGENET_MEAN,
        "std": config.IMAGENET_STD,
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path | str = config.MODEL_PATH, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load the trained model strictly.

    Raises ``ModelLoadError`` with a human-readable explanation instead of
    silently falling back to an untrained network.
    """
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(
            f"Trained model checkpoint not found at '{path}'. "
            "Run the training pipeline (python -m training.train) or copy the "
            "checkpoint into the artifacts/ folder."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - we want to surface everything
        raise ModelLoadError(f"Checkpoint '{path}' could not be read: {exc}") from exc

    if not isinstance(payload, dict) or "state_dict" not in payload or "class_names" not in payload:
        raise ModelLoadError(f"Checkpoint '{path}' is not a Plant Doctor checkpoint (missing state_dict/class_names).")
    if payload.get("architecture") != config.ARCHITECTURE:
        raise ModelLoadError(
            f"Checkpoint architecture '{payload.get('architecture')}' does not match expected '{config.ARCHITECTURE}'."
        )

    class_names = list(payload["class_names"])
    model = build_model(num_classes=len(class_names), pretrained=False)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(f"Checkpoint weights are incompatible with the model definition: {exc}") from exc

    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, meta


def write_class_mapping(class_names: list[str], path: Path = config.CLASS_MAP_PATH) -> None:
    mapping = {
        "num_classes": len(class_names),
        "index_to_class": {str(i): name for i, name in enumerate(class_names)},
        "class_to_index": {name: i for i, name in enumerate(class_names)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2))
