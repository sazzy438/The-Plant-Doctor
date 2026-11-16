"""Image-quality and out-of-distribution (OOD) safeguards.

Plant Doctor refuses to name a condition when the evidence is weak. The
decision combines four independent, configurable signals:

1. **Sharpness** – variance of the Laplacian (very blurry photos are rejected).
2. **Feature-space novelty** – Mahalanobis distance of the network's 1280-d
   penultimate features to the nearest PlantVillage class centre, using a
   shared covariance fitted on training features and a threshold calibrated on
   validation images (99th percentile). Unrelated photographs and unsupported
   crops typically sit far away from every class centre.
3. **Softmax confidence** – the top probability must exceed ``MIN_CONFIDENCE``.
4. **Vegetation fraction** – a weak colour heuristic: below 1 % (no plant-like
   pixels at all, e.g. documents or sky) it rejects; below 5 % it only warns.

Limitations (documented honestly): the novelty detector is trained only on
PlantVillage-style photos (single leaf, plain background). Photos of leaves in
the field may be flagged as unusual even when they show a supported crop, and
a leaf of an unsupported crop that resembles a supported one can slip through.
None of these checks is a substitute for an expert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config


@dataclass
class QualityAssessment:
    sharpness: float
    vegetation_fraction: float
    novelty_score: float | None        # normalised: 1.0 == calibrated 99th-percentile distance
    novelty_raw: float | None
    top_confidence: float
    abstain: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sharpness": round(self.sharpness, 2),
            "vegetation_fraction": round(self.vegetation_fraction, 3),
            "novelty_score": None if self.novelty_score is None else round(self.novelty_score, 3),
            "top_confidence": round(self.top_confidence, 4),
            "abstain": self.abstain,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


class NoveltyDetector:
    """Mahalanobis-distance novelty score in penultimate feature space."""

    def __init__(self, means: np.ndarray, precision: np.ndarray, threshold: float, percentiles: dict[str, float]):
        self.means = means.astype(np.float32)
        self.precision = precision.astype(np.float32)
        self.threshold = float(threshold)
        self.percentiles = percentiles

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, calib_features: np.ndarray, shrinkage: float = 0.1) -> "NoveltyDetector":
        classes = np.unique(labels)
        means = np.stack([features[labels == c].mean(0) for c in classes])
        centred = features - means[np.searchsorted(classes, labels)]
        cov = centred.T @ centred / len(features)
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
        precision = np.linalg.inv(cov)
        det = cls(means, precision, threshold=1.0, percentiles={})
        calib = det.raw_scores(calib_features)
        pct = {f"p{p}": float(np.percentile(calib, p)) for p in (50, 90, 95, 99, 99.9)}
        det.threshold = pct["p99"]
        det.percentiles = pct
        return det

    def raw_scores(self, features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(features).astype(np.float32)
        d2 = np.empty((len(features), len(self.means)), dtype=np.float32)
        for i, mu in enumerate(self.means):
            diff = features - mu
            d2[:, i] = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
        return np.sqrt(np.clip(d2.min(1), 0, None))

    def score(self, feature: np.ndarray) -> tuple[float, float]:
        raw = float(self.raw_scores(feature)[0])
        return raw / self.threshold, raw

    def save(self, path: Path) -> None:
        np.savez_compressed(path, means=self.means, precision=self.precision, threshold=self.threshold,
                            percentile_keys=np.array(list(self.percentiles.keys())),
                            percentile_values=np.array(list(self.percentiles.values())))

    @classmethod
    def load(cls, path: Path) -> "NoveltyDetector":
        z = np.load(path)
        pct = {str(k): float(v) for k, v in zip(z["percentile_keys"], z["percentile_values"])}
        return cls(z["means"], z["precision"], float(z["threshold"]), pct)


def assess(sharpness: float, vegetation: float, novelty: tuple[float, float] | None, top_confidence: float,
           min_confidence: float = config.MIN_CONFIDENCE, blur_threshold: float = config.BLUR_THRESHOLD,
           ood_limit: float = config.OOD_SCORE_LIMIT, min_vegetation: float = config.MIN_VEGETATION_FRACTION,
           min_vegetation_hard: float = config.MIN_VEGETATION_HARD) -> QualityAssessment:
    reasons, warnings = [], []
    if vegetation < min_vegetation_hard:
        reasons.append("No plant-coloured area was detected in the image; it does not appear to show a leaf.")
    if sharpness < blur_threshold:
        reasons.append("The photo looks too blurry for a reliable analysis.")
    nov_score, nov_raw = (None, None) if novelty is None else novelty
    if nov_score is not None and nov_score > ood_limit:
        reasons.append("The image does not resemble the leaf photographs the model was trained on (it may not show a supported crop, or may not be a leaf).")
    if top_confidence < min_confidence:
        reasons.append(f"The model's confidence ({top_confidence*100:.0f}%) is below the {min_confidence*100:.0f}% threshold.")
    if min_vegetation_hard <= vegetation < min_vegetation:
        warnings.append("Very little plant-coloured area was detected; make sure the leaf fills most of the frame.")
    if nov_score is not None and ood_limit * 0.75 < nov_score <= ood_limit:
        warnings.append("This image is somewhat unusual compared with the training photographs; treat the result with extra caution.")
    return QualityAssessment(
        sharpness=sharpness, vegetation_fraction=vegetation, novelty_score=nov_score, novelty_raw=nov_raw,
        top_confidence=top_confidence, abstain=bool(reasons), reasons=reasons, warnings=warnings,
    )
