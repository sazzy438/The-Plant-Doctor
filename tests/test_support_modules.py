"""Cropper, novelty detector, PDF report and session history."""
import numpy as np
from PIL import Image

import config
from core import cropper, history
from core.report import build_report
from models.quality import NoveltyDetector, assess


def test_cropper_box_conversions_and_preview(diseased_leaf):
    img = diseased_leaf
    box = cropper.box_from_percent(img, 25, 25, 75, 75)
    assert box == (64, 64, 192, 192)
    assert cropper.is_full(img, None) and cropper.is_full(img, (0, 0, 256, 256)) and not cropper.is_full(img, box)
    pct = cropper.percent_from_box(img, box)
    assert [round(v) for v in pct] == [25, 25, 75, 75]
    plain = cropper.draw_selection(img, None)
    assert plain.size == img.size and np.array_equal(np.asarray(plain), np.asarray(img))
    marked = cropper.draw_selection(img, box, pending=None)
    arr_m, arr_o = np.asarray(marked).astype(int), np.asarray(img).astype(int)
    # Outside the box the preview is darker; inside it is unchanged (except the outline).
    assert arr_m[10, 10].sum() < arr_o[10, 10].sum()
    assert np.array_equal(arr_m[120:140, 120:140], arr_o[120:140, 120:140])
    pending = cropper.draw_selection(img, None, pending=(30, 30))
    assert pending.size == img.size
    big = Image.new("RGB", (3000, 2000), (40, 90, 50))
    assert max(cropper.draw_selection(big, (10, 10, 500, 500)).size) == 1024


def test_novelty_detector_separates_far_points(tmp_path):
    rng = np.random.default_rng(0)
    d, n = 16, 400
    means = rng.normal(0, 5, size=(3, d))
    feats = np.concatenate([rng.normal(m, 1.0, size=(n, d)) for m in means]).astype(np.float32)
    labels = np.repeat(np.arange(3), n)
    calib = np.concatenate([rng.normal(m, 1.0, size=(50, d)) for m in means]).astype(np.float32)
    det = NoveltyDetector.fit(feats, labels, calib)
    in_dist, _ = det.score(rng.normal(means[1], 1.0, size=d).astype(np.float32))
    far, _ = det.score((means[1] + 40).astype(np.float32))
    assert in_dist < 1.5 and far > 3
    det.save(tmp_path / "ood.npz")
    det2 = NoveltyDetector.load(tmp_path / "ood.npz")
    assert abs(det2.threshold - det.threshold) < 1e-6 and det2.percentiles["p99"] == det.percentiles["p99"]


def test_assess_rules():
    q = assess(sharpness=1000, vegetation=0.5, novelty=(0.5, 10), top_confidence=0.9)
    assert not q.abstain and not q.reasons
    q = assess(sharpness=1, vegetation=0.5, novelty=None, top_confidence=0.9)
    assert q.abstain and "blurry" in q.reasons[0]
    q = assess(sharpness=1000, vegetation=0.01, novelty=(2.0, 50), top_confidence=0.3)
    assert q.abstain and len(q.reasons) == 2 and q.warnings
    q = assess(sharpness=1000, vegetation=0.5, novelty=(0.9, 10), top_confidence=0.9)
    assert not q.abstain and any("unusual" in w for w in q.warnings)
    q = assess(sharpness=4000, vegetation=0.0, novelty=(0.99, 60), top_confidence=0.75)  # a text document
    assert q.abstain and any("plant-coloured" in r for r in q.reasons)
    assert set(q.as_dict()) >= {"sharpness", "novelty_score", "abstain", "reasons", "warnings"}


def test_report_builds_pdf(tmp_path, diseased_leaf):
    top = {"class_name": "Tomato___Early_blight", "crop": "Tomato", "condition": "Early blight", "is_healthy": False, "probability": 0.91}
    alts = [{"class_name": "Tomato___Target_Spot", "crop": "Tomato", "condition": "Target spot", "is_healthy": False, "probability": 0.05}]
    from core.disease_info import get_info
    out = build_report(tmp_path / "r.pdf", analysed_image=diseased_leaf, heatmap=diseased_leaf, overlay=diseased_leaf, file_name="leaf.jpg",
                       top=top, alternatives=alts, quality={"sharpness": 500, "vegetation_fraction": 0.6, "novelty_score": 0.4, "reasons": [], "warnings": []},
                       info=get_info("Tomato___Early_blight"), abstained=False, region_note="Full image", model_meta={"epoch": 3, "val_metrics": {"macro_f1": 0.98}})
    data = out.read_bytes()
    assert data[:4] == b"%PDF" and len(data) > 20_000
    out2 = build_report(tmp_path / "r2.pdf", analysed_image=diseased_leaf, heatmap=None, overlay=None, file_name="x.png", top=top, alternatives=[],
                        quality={"reasons": ["too blurry"], "warnings": []}, info=None, abstained=True, region_note="Full image", model_meta={})
    assert out2.read_bytes()[:4] == b"%PDF"


def test_history_push_limit_and_find(diseased_leaf):
    h = []
    for i in range(config.HISTORY_LIMIT + 3):
        h = history.push(h, history.new_entry(diseased_leaf, f"f{i}.jpg", "Tomato — Early blight", 0.9, False, {"i": i}))
    assert len(h) == config.HISTORY_LIMIT and h[0].file_name == f"f{config.HISTORY_LIMIT + 2}.jpg"
    assert history.find(h, h[3].id).payload["i"] == h[3].payload["i"]
    assert history.find(h, "nope") is None and h[0].thumbnail_b64
