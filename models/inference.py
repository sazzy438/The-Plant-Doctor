"""Cached, thread-safe prediction engine for Plant Doctor.

``get_predictor()`` loads the trained checkpoint exactly once per process and
reuses it for every request. All inference runs under ``torch.inference_mode``
except Grad-CAM, which needs gradients and uses ``torch.enable_grad`` locally.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from core.image_processing import sharpness_score, vegetation_fraction
from models.gradcam import GradCAM, GradCAMResult
from models.model import ModelLoadError, load_checkpoint, target_layer
from models.quality import NoveltyDetector, QualityAssessment, assess
from training.preprocessing import get_eval_transforms

_lock = threading.Lock()
_predictor: "Predictor | None" = None


def split_class_name(class_name: str) -> tuple[str, str, bool]:
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight', False)."""
    crop, _, condition = class_name.partition("___")
    crop = crop.replace("_", " ").replace(",", ",").replace("(including sour)", "(incl. sour)").strip()
    cond = condition.replace("_", " ").strip()
    healthy = cond.lower() == "healthy"
    cond = "Healthy" if healthy else " ".join(cond.split())
    return crop, cond, healthy


@dataclass
class TopPrediction:
    class_name: str
    crop: str
    condition: str
    is_healthy: bool
    probability: float

    def as_dict(self) -> dict:
        return {"class_name": self.class_name, "crop": self.crop, "condition": self.condition,
                "is_healthy": self.is_healthy, "probability": round(self.probability, 6)}


@dataclass
class PredictionResult:
    top: TopPrediction
    alternatives: list[TopPrediction]       # top-3 excluding the winner
    probabilities: np.ndarray                # all 38 softmax probabilities
    features: np.ndarray                     # 1280-d penultimate features
    quality: QualityAssessment
    inference_ms: float
    input_size: tuple[int, int]
    gradcam: GradCAMResult | None = None
    extra: dict = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.quality.abstain

    def as_dict(self) -> dict:
        return {
            "top": self.top.as_dict(),
            "alternatives": [a.as_dict() for a in self.alternatives],
            "quality": self.quality.as_dict(),
            "abstained": self.abstained,
            "inference_ms": round(self.inference_ms, 1),
            "input_size": list(self.input_size),
        }


class Predictor:
    def __init__(self, model_path: Path = config.MODEL_PATH, device: str | None = None,
                 ood_path: Path = config.OOD_STATS_PATH):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.meta = load_checkpoint(model_path, self.device)
        self.model_path = Path(model_path)
        self.class_names: list[str] = list(self.meta["class_names"])
        self.transform = get_eval_transforms(int(self.meta.get("image_size", config.IMAGE_SIZE)))
        self.novelty: NoveltyDetector | None = NoveltyDetector.load(ood_path) if Path(ood_path).exists() else None
        self._features: torch.Tensor | None = None
        self.model.avgpool.register_forward_hook(self._capture_features)
        self._gradcam = GradCAM(self.model, target_layer(self.model))
        self._infer_lock = threading.Lock()
        # Warm-up so the first user request is fast.
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=self.device))

    # ------------------------------------------------------------------ #
    def _capture_features(self, _m, _i, output: torch.Tensor) -> None:
        self._features = output.flatten(1).detach()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """RGB PIL image -> (1, 3, 224, 224) normalised tensor. This is the exact model input."""
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def crops(self) -> list[str]:
        return sorted({split_class_name(c)[0] for c in self.class_names})

    # ------------------------------------------------------------------ #
    def predict(self, image: Image.Image, top_k: int = 3, with_gradcam: bool = True) -> PredictionResult:
        """Run the full pipeline on an RGB image (already cropped if the user cropped)."""
        t0 = time.perf_counter()
        x = self.preprocess(image)
        with self._infer_lock:
            with torch.inference_mode():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                feats = self._features[0].cpu().numpy() if self._features is not None else np.zeros(1280, np.float32)
            order = np.argsort(-probs)
            preds = [TopPrediction(self.class_names[i], *split_class_name(self.class_names[i]), float(probs[i])) for i in order[: top_k + 1]]
            novelty = self.novelty.score(feats) if self.novelty is not None else None
            quality = assess(
                sharpness=sharpness_score(image), vegetation=vegetation_fraction(image),
                novelty=novelty, top_confidence=preds[0].probability,
            )
            cam = None
            if with_gradcam:
                cam = self._gradcam(x, class_index=int(order[0]), output_size=image.size)
        ms = (time.perf_counter() - t0) * 1000
        return PredictionResult(top=preds[0], alternatives=preds[1 : top_k + 1], probabilities=probs, features=feats,
                                quality=quality, inference_ms=ms, input_size=image.size, gradcam=cam)

    def explain(self, image: Image.Image, class_index: int) -> GradCAMResult:
        """Grad-CAM for an arbitrary class (used when the user inspects an alternative)."""
        with self._infer_lock:
            return self._gradcam(self.preprocess(image), class_index=class_index, output_size=image.size)

    @torch.inference_mode()
    def extract_features(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Batch helper for the evaluation pipeline: returns (logits, features)."""
        logits = self.model(batch.to(self.device))
        return logits.cpu().numpy(), self._features.cpu().numpy()


def get_predictor(model_path: Path | None = None, force_reload: bool = False) -> Predictor:
    """Process-wide cached predictor. Raises ``ModelLoadError`` if the checkpoint is unusable."""
    global _predictor
    with _lock:
        if _predictor is None or force_reload or (model_path and Path(model_path) != _predictor.model_path):
            _predictor = Predictor(model_path or config.MODEL_PATH)
        return _predictor


__all__ = ["Predictor", "PredictionResult", "TopPrediction", "get_predictor", "ModelLoadError", "split_class_name"]
