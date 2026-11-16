import numpy as np
import pytest
from PIL import Image

import config
from core.image_processing import (ImageValidationError, crop_image, load_and_validate, normalise_box,
                                   sharpness_score, vegetation_fraction)


def test_valid_jpeg_png_webp_load(tmp_image_file, diseased_leaf):
    for name, fmt in (("a.jpg", "JPEG"), ("b.png", "PNG"), ("c.webp", "WEBP")):
        loaded = load_and_validate(tmp_image_file(name, diseased_leaf, fmt))
        assert loaded.image.mode == "RGB" and loaded.image.size == diseased_leaf.size
        assert loaded.format in config.SUPPORTED_FORMATS


def test_unsupported_format_rejected(tmp_image_file, diseased_leaf):
    p = tmp_image_file("leaf.gif", diseased_leaf, "GIF")
    with pytest.raises(ImageValidationError, match="not a supported file type"):
        load_and_validate(p)
    # Correct extension but wrong content (a BMP renamed to .jpg) is also rejected
    p2 = tmp_image_file("fake.jpg", diseased_leaf, "BMP")
    with pytest.raises(ImageValidationError, match="not supported"):
        load_and_validate(p2)


def test_corrupted_file_rejected(tmp_image_file, diseased_leaf_path):
    data = diseased_leaf_path.read_bytes()
    truncated = tmp_image_file("trunc.jpg", content=data[: len(data) // 3])
    garbage = tmp_image_file("garbage.jpg", content=b"\x00\x01not an image" * 100)
    for p in (truncated, garbage):
        with pytest.raises(ImageValidationError, match="corrupted|could not be read"):
            load_and_validate(p)


def test_too_small_and_very_large(tmp_image_file):
    tiny = Image.new("RGB", (20, 20), (30, 120, 40))
    with pytest.raises(ImageValidationError, match="too small"):
        load_and_validate(tmp_image_file("tiny.png", tiny, "PNG"))
    big = Image.new("RGB", (4500, 3000), (30, 120, 40))
    loaded = load_and_validate(tmp_image_file("big.jpg", big, "JPEG"))
    assert loaded.was_downscaled and max(loaded.image.size) == 2048 and loaded.original_size == (4500, 3000)
    huge = Image.new("RGB", (config.MAX_IMAGE_SIDE + 10, 100))
    with pytest.raises(ImageValidationError, match="too large"):
        load_and_validate(tmp_image_file("huge.png", huge, "PNG"))


def test_file_size_limit(tmp_path):
    p = tmp_path / "bloated.jpg"
    p.write_bytes(b"\xff\xd8" + b"0" * (config.MAX_FILE_SIZE_MB * 1024 * 1024 + 1))
    with pytest.raises(ImageValidationError, match="larger than"):
        load_and_validate(p)


def test_crop_changes_pixels_and_validates(diseased_leaf):
    c = crop_image(diseased_leaf, (10, 20, 110, 140))
    assert c.size == (100, 120)
    assert np.array_equal(np.asarray(c), np.asarray(diseased_leaf)[20:140, 10:110])
    with pytest.raises(ImageValidationError, match="too small"):
        crop_image(diseased_leaf, (0, 0, 10, 10))
    assert normalise_box((-50, -50, 5000, 5000), 256, 256) == (0, 0, 256, 256)


def test_sharpness_detects_blur(diseased_leaf, blurry_leaf):
    sharp, blurry = sharpness_score(diseased_leaf), sharpness_score(blurry_leaf)
    assert sharp > config.BLUR_THRESHOLD > blurry


def test_vegetation_fraction(diseased_leaf, unrelated_image):
    assert vegetation_fraction(diseased_leaf) > 0.3
    assert vegetation_fraction(unrelated_image) < 0.1
