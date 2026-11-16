"""End-to-end tests of the application layer (handlers called directly, no browser)."""
import os

import pytest
from PIL import Image

import config
from core import disease_info


@pytest.fixture(scope="module")
def app_module(model_path):
    os.environ["PLANT_DOCTOR_MODEL"] = str(model_path)
    import importlib
    import config as cfg
    importlib.reload(cfg)
    import app
    importlib.reload(app)
    return app


def test_disease_database_covers_all_classes(predictor):
    classes = disease_info.all_classes()
    for name in predictor.class_names:
        info = disease_info.get_info(name)
        assert info["common_name"] and info["symptoms"] and info["prevention"] and info["references"]
    assert set(classes) == set(predictor.class_names)


def test_upload_validation_in_app(app_module, tmp_image_file, diseased_leaf):
    state, _, _, note, col, *_ = app_module.on_upload(str(tmp_image_file("leaf.png", diseased_leaf, "PNG")))
    assert state and state["image"].size == diseased_leaf.size and "leaf.png" in note
    state, _, _, note, *_ = app_module.on_upload(str(tmp_image_file("bad.jpg", content=b"nope")))
    assert state is None and "couldn" in note
    assert app_module.on_upload(None)[0] is None


def test_full_analysis_flow_and_history(app_module, tmp_image_file, diseased_leaf):
    app = app_module
    state, sel, preview, _, _, *sliders = app.on_upload(str(tmp_image_file("leaf.jpg", diseased_leaf, "JPEG")))
    assert sel["box"] is None and preview.size == diseased_leaf.size and sliders == [0, 0, 100, 100]
    out = app.analyze("full", state, sel, [], progress=lambda *a, **k: None)
    result_html, chart_html, slider, gallery, quality_html, info_html, compare_html, report, history, history_html, dd, col, tabs = out
    assert "pd-" in result_html and "Model predictions" in chart_html
    assert isinstance(slider, tuple) and len(gallery) == 3
    assert report and report.endswith(".pdf") and os.path.getsize(report) > 10_000
    assert len(history) == 1 and history[0].file_name == "leaf.jpg" and history[0].thumbnail_b64
    assert "leaf.jpg" in history_html
    # two-click selection defines the crop box; the crop is what the model sees
    class Evt:  # minimal stand-in for gr.SelectData
        def __init__(self, x, y): self.index = [x, y]
    sel, prev, *sl = app.on_preview_select(state, sel, Evt(10, 20))
    assert sel["pending"] == (10, 20) and sel["box"] is None
    sel, prev, *sl = app.on_preview_select(state, sel, Evt(130, 140))
    assert sel["box"] == (10, 20, 130, 140) and sel["pending"] is None
    assert prev.size == diseased_leaf.size and sl != [0, 0, 100, 100]
    out2 = app.analyze("selection", state, sel, history, progress=lambda *a, **k: None)
    assert "Selected region · 120×120" in out2[0]
    assert len(out2[8]) == 2
    # compare mode
    out3 = app.analyze("compare", state, sel, out2[8], progress=lambda *a, **k: None)
    assert "Full image vs selected region" in out3[6]
    # reopen from history
    reopened = app.reopen(out3[8][0].id, out3[8])
    assert reopened[0] == out3[0]
    # sliders drive the same box; full-range sliders mean "no crop"
    sel2, prev2 = app.on_sliders(state, sel, 25, 25, 75, 75)
    assert sel2["box"] == (64, 64, 192, 192)
    sel3, _ = app.on_sliders(state, sel2, 0, 0, 100, 100)
    assert sel3["box"] is None
    # too-small crop is refused gracefully
    sel_small = {"box": (0, 0, 8, 8), "pending": None}
    out4 = app.analyze("selection", state, sel_small, [], progress=lambda *a, **k: None)
    assert "too small" in out4[0]
    # reset
    reset = app.reset_crop(state)
    assert reset[0]["box"] is None and list(reset[2:]) == [0, 0, 100, 100]


def test_analyze_without_upload(app_module):
    out = app_module.analyze("full", None, None, [], progress=lambda *a, **k: None)
    assert "upload a leaf image first" in out[0]


def test_disease_info_matches_prediction(app_module, diseased_leaf, trained_model_available):
    if not trained_model_available:
        pytest.skip("requires the trained checkpoint")
    payload = app_module._run(diseased_leaf, "x.jpg", "Full image")
    html = app_module._render(payload)[5]
    assert disease_info.get_info(payload["class_name"])["common_name"] in html
