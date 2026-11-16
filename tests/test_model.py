import json

import pytest
import torch

import config
from models.gradcam import GradCAM, cam_to_heatmap, overlay_heatmap
from models.model import ModelLoadError, build_model, load_checkpoint, save_checkpoint, target_layer


def test_build_model_shape():
    m = build_model(38, pretrained=False).eval()
    with torch.inference_mode():
        out = m(torch.zeros(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    assert out.shape == (2, 38)


def test_checkpoint_roundtrip_and_strict_errors(tmp_path):
    m = build_model(5, pretrained=False)
    p = tmp_path / "m.pt"
    save_checkpoint(m, p, ["a", "b", "c", "d", "e"], extra={"epoch": 3})
    m2, meta = load_checkpoint(p)
    assert meta["class_names"] == ["a", "b", "c", "d", "e"] and meta["epoch"] == 3
    for k, v in m.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k])
    with pytest.raises(ModelLoadError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")
    (tmp_path / "bad.pt").write_bytes(b"junk")
    with pytest.raises(ModelLoadError, match="could not be read"):
        load_checkpoint(tmp_path / "bad.pt")
    torch.save({"foo": 1}, tmp_path / "wrong.pt")
    with pytest.raises(ModelLoadError, match="not a Plant Doctor checkpoint"):
        load_checkpoint(tmp_path / "wrong.pt")


def test_artifacts_consistent():
    """The checkpoint, class mapping and model config must agree."""
    if not config.MODEL_PATH.exists():
        pytest.skip("trained checkpoint not present")
    _, meta = load_checkpoint(config.MODEL_PATH)
    mapping = json.loads(config.CLASS_MAP_PATH.read_text())
    assert [mapping["index_to_class"][str(i)] for i in range(mapping["num_classes"])] == meta["class_names"]
    assert len(meta["class_names"]) == config.NUM_CLASSES == json.loads(config.MODEL_CONFIG_PATH.read_text())["num_classes"]


def test_gradcam_uses_real_gradients():
    torch.manual_seed(0)
    m = build_model(38, pretrained=False).eval()
    cam = GradCAM(m, target_layer(m))
    x = torch.rand(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    r1 = cam(x, class_index=3, output_size=(64, 48))
    r2 = cam(x, class_index=7, output_size=(64, 48))
    assert r1.cam.shape == (48, 64) and r1.raw_cam_shape == (7, 7)
    assert 0.0 <= r1.cam.min() and r1.cam.max() <= 1.0
    # Different classes -> different gradients -> different maps
    assert not (abs(r1.cam - r2.cam) < 1e-6).all()
    # Zeroing the classifier weights of the class removes its gradient signal -> empty map
    with torch.no_grad():
        m.classifier[1].weight[3].zero_(); m.classifier[1].bias[3].zero_()
    r3 = cam(x, class_index=3, output_size=(64, 48))
    assert r3.cam.max() == 0.0
    heat = cam_to_heatmap(r1.cam)
    assert heat.size == (64, 48)
    from PIL import Image
    over = overlay_heatmap(Image.new("RGB", (64, 48)), r1.cam)
    assert over.size == (64, 48)
    with pytest.raises(ValueError):
        overlay_heatmap(Image.new("RGB", (10, 10)), r1.cam)
