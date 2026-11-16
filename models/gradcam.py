"""Grad-CAM (Selvaraju et al., 2017) for the Plant Doctor EfficientNet-B0.

The class-activation map is computed from the *actual* gradients of the
selected class score with respect to the last convolutional feature map.
Nothing here is synthetic: if the gradients are zero the map is empty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torch import nn


@dataclass
class GradCAMResult:
    cam: np.ndarray            # float32, HxW in [0, 1], same size as the analysed image
    class_index: int
    class_score: float         # logit of the explained class
    raw_cam_shape: tuple[int, int]


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [target_layer.register_forward_hook(self._forward_hook)]

    def _forward_hook(self, _module, _inputs, output: torch.Tensor) -> None:
        self._activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, input_tensor: torch.Tensor, class_index: int | None = None, output_size: tuple[int, int] | None = None) -> GradCAMResult:
        """``input_tensor``: (1, 3, H, W) preprocessed. ``output_size``: (width, height) to upsample the map to."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = input_tensor.clone().requires_grad_(True)
            logits = self.model(x)
            if class_index is None:
                class_index = int(logits.argmax(dim=1).item())
            score = logits[0, class_index]
            score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients; wrong target layer?")
        acts = self._activations.detach()[0]          # (C, h, w)
        grads = self._gradients.detach()[0]           # (C, h, w)
        weights = grads.mean(dim=(1, 2), keepdim=True)  # alpha_k^c
        cam = F.relu((weights * acts).sum(dim=0))      # (h, w)
        raw_shape = tuple(cam.shape)
        if cam.max() > 0:
            cam = cam / cam.max()
        if output_size is None:
            output_size = (input_tensor.shape[-1], input_tensor.shape[-2])
        w, h = output_size
        cam_up = F.interpolate(cam[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        self.model.zero_grad(set_to_none=True)
        return GradCAMResult(cam=cam_up.cpu().numpy().astype(np.float32), class_index=class_index,
                             class_score=float(score.item()), raw_cam_shape=raw_shape)


def cam_to_heatmap(cam: np.ndarray, colormap: str = "inferno") -> Image.Image:
    """Colourise a [0,1] map as an RGB image."""
    rgba = colormaps[colormap](np.clip(cam, 0, 1))
    return Image.fromarray((rgba[..., :3] * 255).astype(np.uint8), mode="RGB")


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45, colormap: str = "inferno") -> Image.Image:
    """Blend the heatmap over ``image``. ``cam`` must already match ``image.size``."""
    if cam.shape[:2] != (image.height, image.width):
        raise ValueError(f"CAM shape {cam.shape[:2]} does not match image size {(image.height, image.width)}")
    heat = np.asarray(cam_to_heatmap(cam, colormap), dtype=np.float32)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
