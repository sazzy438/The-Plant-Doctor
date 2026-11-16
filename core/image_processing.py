"""Upload validation, safe decoding, cropping and image-quality heuristics.

Nothing in this module touches the neural network; it prepares pixels and
computes cheap signals (sharpness, vegetation fraction) that the abstention
logic in ``models.quality`` combines with model-based signals.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy import ndimage

import config

Image.MAX_IMAGE_PIXELS = config.MAX_PIXELS


class ImageValidationError(ValueError):
    """User-facing problem with an uploaded image (format, size, corruption)."""


@dataclass
class LoadedImage:
    image: Image.Image           # RGB, EXIF-corrected, possibly downscaled for processing
    original_size: tuple[int, int]
    file_name: str
    format: str
    was_downscaled: bool


def _open(source: str | Path | bytes | Image.Image) -> tuple[Image.Image, str]:
    if isinstance(source, Image.Image):
        return source, "memory"
    if isinstance(source, (bytes, bytearray)):
        if len(source) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ImageValidationError(f"The file is larger than {config.MAX_FILE_SIZE_MB} MB. Please upload a smaller image.")
        return Image.open(io.BytesIO(source)), "upload"
    path = Path(source)
    if not path.exists():
        raise ImageValidationError("The uploaded file could not be found. Please try uploading again.")
    if path.stat().st_size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ImageValidationError(f"The file is larger than {config.MAX_FILE_SIZE_MB} MB. Please upload a smaller image.")
    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"'{path.suffix or 'no extension'}' is not a supported file type. Please upload a JPG, PNG or WEBP image."
        )
    return Image.open(path), path.name


def load_and_validate(source: str | Path | bytes | Image.Image, max_side: int = 2048) -> LoadedImage:
    """Decode an upload defensively and return an RGB working copy.

    Raises ``ImageValidationError`` for anything a user can fix.
    """
    try:
        im, name = _open(source)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "This file could not be read as an image. It may be corrupted or in an unsupported format."
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("This image has too many pixels to process safely.") from exc

    fmt = (im.format or "RGB").upper()
    if isinstance(source, Image.Image):
        fmt = "RGB"
    elif fmt == "MPO":  # multi-picture JPEGs from phones
        fmt = "JPEG"
    elif fmt not in config.SUPPORTED_FORMATS:
        raise ImageValidationError(f"{fmt} images are not supported. Please upload a JPG, PNG or WEBP image.")

    try:
        im.load()  # forces the full decode; corrupted files fail here
        im = ImageOps.exif_transpose(im)
        rgb = im.convert("RGB")
    except (OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError("This image appears to be corrupted and could not be decoded.") from exc

    w, h = rgb.size
    if min(w, h) < config.MIN_IMAGE_SIDE:
        raise ImageValidationError(
            f"The image is too small ({w}x{h}px). Please upload an image at least {config.MIN_IMAGE_SIDE}px on each side."
        )
    if max(w, h) > config.MAX_IMAGE_SIDE:
        raise ImageValidationError(
            f"The image is too large ({w}x{h}px). Please upload an image no larger than {config.MAX_IMAGE_SIDE}px on a side."
        )
    downscaled = False
    if max(w, h) > max_side:
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        downscaled = True
    return LoadedImage(image=rgb, original_size=(w, h), file_name=name, format=fmt, was_downscaled=downscaled)


def normalise_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp an (x0, y0, x1, y1) box to the image and validate its size."""
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0 = int(max(0, round(x0))), int(max(0, round(y0)))
    x1, y1 = int(min(width, round(x1))), int(min(height, round(y1)))
    if x1 - x0 < config.MIN_CROP_SIDE or y1 - y0 < config.MIN_CROP_SIDE:
        raise ImageValidationError(
            f"The selected region is too small. Please select an area at least {config.MIN_CROP_SIDE}px wide and tall."
        )
    return x0, y0, x1, y1


def crop_image(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Return the crop defined by ``box`` (pixel coordinates). The crop is what the model sees."""
    return image.crop(normalise_box(box, *image.size))


def sharpness_score(image: Image.Image, working_side: int = 512) -> float:
    """Variance of the Laplacian on a normalised grayscale copy (higher = sharper).

    Images are resized to a fixed working size so the score is comparable
    across uploads of different resolutions.
    """
    g = image.convert("L")
    g.thumbnail((working_side, working_side), Image.Resampling.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)
    return float(ndimage.laplace(arr).var())


def vegetation_fraction(image: Image.Image, working_side: int = 256) -> float:
    """Fraction of pixels that look like plant tissue (green through yellow-brown hues).

    This is a weak heuristic: it is used to *warn*, never to decide by itself.
    """
    im = image.copy()
    im.thumbnail((working_side, working_side))
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    plant = (h >= 20) & (h <= 170) & (s >= 0.12) & (v >= 0.12)
    return float(plant.mean())


def to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
