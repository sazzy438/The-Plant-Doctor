import numpy as np
import pytest
import torch
from PIL import Image

import config
from core.image_processing import crop_image
from models.inference import PredictionResult, split_class_name


def test_split_class_name():
    assert split_class_name("Tomato___Early_blight") == ("Tomato", "Early blight", False)
    assert split_class_name("Pepper,_bell___healthy") == ("Pepper, bell", "Healthy", True)
    assert split_class_name("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")[1] == "Cercospora leaf spot Gray leaf spot"


def test_predict_structure_and_probabilities(predictor, diseased_leaf):
    r = predictor.predict(diseased_leaf)
    assert isinstance(r, PredictionResult)
    assert r.probabilities.shape == (predictor.num_classes,)
    assert abs(r.probabilities.sum() - 1.0) < 1e-4
    assert 0 <= r.top.probability <= 1 and len(r.alternatives) == 3
    assert r.top.probability >= max(a.probability for a in r.alternatives)
    assert r.gradcam is not None and r.gradcam.cam.shape == (diseased_leaf.height, diseased_leaf.width)
    assert r.features.shape == (1280,)
    assert r.inference_ms > 0
    d = r.as_dict()
    assert set(d) >= {"top", "alternatives", "quality", "abstained", "inference_ms"}


def test_prediction_is_deterministic_and_repeatable(predictor, diseased_leaf):
    a = predictor.predict(diseased_leaf, with_gradcam=False)
    b = predictor.predict(diseased_leaf, with_gradcam=False)
    assert np.allclose(a.probabilities, b.probabilities, atol=1e-6)


def test_crop_changes_model_input_and_output(predictor, diseased_leaf):
    full = predictor.preprocess(diseased_leaf)
    crop = crop_image(diseased_leaf, (0, 0, 128, 128))
    cropped = predictor.preprocess(crop)
    assert full.shape == cropped.shape == (1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert not torch.equal(full, cropped)
    # The preprocessed crop must equal preprocessing the crop independently (no hidden full-image path)
    expected = predictor.transform(crop).unsqueeze(0)
    assert torch.allclose(cropped, expected)
    p_full = predictor.predict(diseased_leaf, with_gradcam=False).probabilities
    p_crop = predictor.predict(crop, with_gradcam=False).probabilities
    assert not np.allclose(p_full, p_crop)


def test_predictions_come_from_the_network_not_metadata(predictor, diseased_leaf):
    """Perturbing the network weights must change the output; filenames/labels are never consulted."""
    base = predictor.predict(diseased_leaf, with_gradcam=False).probabilities
    layer = predictor.model.classifier[1]
    saved = layer.weight.detach().clone()
    with torch.no_grad():
        layer.weight.add_(torch.randn_like(layer.weight) * 0.5)
    try:
        perturbed = predictor.predict(diseased_leaf, with_gradcam=False).probabilities
    finally:
        with torch.no_grad():
            layer.weight.copy_(saved)
    assert not np.allclose(base, perturbed)


def test_blurry_image_abstains(predictor, blurry_leaf):
    r = predictor.predict(blurry_leaf, with_gradcam=False)
    assert r.abstained and any("blurry" in s for s in r.quality.reasons)


def test_large_and_multi_leaf_images_run(predictor, multi_leaf):
    r = predictor.predict(multi_leaf, with_gradcam=True)
    assert r.gradcam.cam.shape == (multi_leaf.height, multi_leaf.width)
    big = multi_leaf.resize((2048, 2048))
    r2 = predictor.predict(big, with_gradcam=True)
    assert r2.gradcam.cam.shape == (2048, 2048)


# ----------------------------------------------------------------------------- trained-model-only checks
def _needs_trained(trained_model_available):
    if not trained_model_available:
        pytest.skip("requires the trained checkpoint")


def test_trained_model_recognises_healthy_leaf(predictor, trained_model_available, healthy_leaf):
    _needs_trained(trained_model_available)
    r = predictor.predict(healthy_leaf)
    assert not r.abstained, r.quality.reasons
    assert r.top.is_healthy and r.top.crop == "Tomato"


def test_trained_model_recognises_diseased_leaf(predictor, trained_model_available, diseased_leaf):
    _needs_trained(trained_model_available)
    r = predictor.predict(diseased_leaf)
    assert not r.abstained, r.quality.reasons
    assert r.top.class_name == "Tomato___Early_blight"


def test_trained_model_cropped_leaf(predictor, trained_model_available, diseased_leaf):
    _needs_trained(trained_model_available)
    w, h = diseased_leaf.size
    crop = crop_image(diseased_leaf, (w * 0.15, h * 0.15, w * 0.85, h * 0.85))
    r = predictor.predict(crop)
    assert r.top.crop == "Tomato"


def test_trained_model_flags_unrelated_photo(predictor, trained_model_available, unrelated_image):
    _needs_trained(trained_model_available)
    r = predictor.predict(unrelated_image, with_gradcam=False)
    assert r.abstained, (r.top, r.quality.as_dict())


def test_novelty_detector_present_after_evaluation(predictor, trained_model_available):
    _needs_trained(trained_model_available)
    if not config.OOD_STATS_PATH.exists():
        pytest.skip("novelty statistics not generated yet (run training.evaluate)")
    assert predictor.novelty is not None and predictor.novelty.threshold > 0
