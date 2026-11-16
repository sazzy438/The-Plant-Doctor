"""Plant Doctor — AI-powered plant health screening (Gradio application).

Run locally:   python app.py
Hugging Face:  this file is the Space entry point (see README.md).
"""
from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

import gradio as gr
from PIL import Image

import config
from core import history as hist
from core import cropper
from core.image_processing import ImageValidationError, crop_image, load_and_validate
from core.report import build_report
from models.gradcam import cam_to_heatmap, overlay_heatmap
from models.inference import ModelLoadError, get_predictor
from ui import render
from ui.styles import CSS, HEAD_HTML

# --------------------------------------------------------------------------- #
# Model bootstrap (never silently falls back)
# --------------------------------------------------------------------------- #
PREDICTOR = None
MODEL_ERROR: str | None = None
try:
    PREDICTOR = get_predictor()
except ModelLoadError as exc:
    MODEL_ERROR = str(exc)
except Exception as exc:  # noqa: BLE001
    MODEL_ERROR = f"Unexpected error while loading the model: {exc}"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except json.JSONDecodeError:
        return None


MODEL_META = PREDICTOR.meta if PREDICTOR else {}
EVAL_METRICS = _read_json(config.EVALUATION_METRICS_PATH)
TRAINING_LOG = _read_json(config.TRAINING_LOG_PATH)
CROPS = PREDICTOR.crops() if PREDICTOR else []
REPORT_DIR = Path(tempfile.mkdtemp(prefix="plant_doctor_reports_"))

if PREDICTOR:
    vm = MODEL_META.get("val_metrics", {})
    MODEL_NOTE = f"Model loaded · EfficientNet-B0 · {PREDICTOR.num_classes} classes · val F1 {vm.get('macro_f1', 0)*100:.1f}% · {PREDICTOR.device.type.upper()}"
else:
    MODEL_NOTE = "Model unavailable — analysis disabled"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _shrink(image: Image.Image, max_side: int = 1024) -> Image.Image:
    im = image.copy()
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


def _run(image: Image.Image, file_name: str, region_note: str) -> dict:
    """Run the full pipeline and return a JSON-friendly payload (plus PIL images)."""
    res = PREDICTOR.predict(image, top_k=3, with_gradcam=True)
    heat = cam_to_heatmap(res.gradcam.cam)
    over = overlay_heatmap(image, res.gradcam.cam)
    top = res.top.as_dict()
    return {
        "file_name": file_name, "region_note": region_note,
        "top": top, "alternatives": [a.as_dict() for a in res.alternatives],
        "quality": res.quality.as_dict(), "abstained": res.abstained,
        "inference_ms": res.inference_ms, "input_size": list(res.input_size),
        "class_name": res.top.class_name, "cam_shape": list(res.gradcam.raw_cam_shape),
        "images": {"analysed": _shrink(image), "heatmap": _shrink(heat), "overlay": _shrink(over)},
    }


def _render(payload: dict, compare: dict | None = None):
    top, q = payload["top"], payload["quality"]
    imgs = payload["images"]
    if payload["abstained"]:
        result_html = render.abstain_card(q, top, payload["region_note"])
        info_html = ""
    else:
        result_html = render.result_card(top, q, payload["region_note"], payload["inference_ms"])
        info_html = render.info_panel(payload["class_name"])
    chart_html = render.predictions_chart(top, payload["alternatives"])
    gallery = [(imgs["analysed"], "Analysed image"), (imgs["heatmap"], "Grad-CAM heatmap"), (imgs["overlay"], "Heatmap overlay")]
    compare_html = render.compare_panel(compare["full"], compare["crop"]) if compare else ""
    return result_html, chart_html, (imgs["analysed"], imgs["overlay"]), gallery, render.quality_panel(q), info_html, compare_html


def _make_report(payload: dict) -> str:
    info = None
    if not payload["abstained"]:
        from core.disease_info import get_info
        info = get_info(payload["class_name"])
    out = REPORT_DIR / f"plant_doctor_report_{os.getpid()}_{abs(hash(payload['file_name'] + payload['region_note'] + str(payload['inference_ms']))) % 10**8}.pdf"
    build_report(out, analysed_image=payload["images"]["analysed"], heatmap=payload["images"]["heatmap"], overlay=payload["images"]["overlay"],
                 file_name=payload["file_name"], top=payload["top"], alternatives=payload["alternatives"], quality=payload["quality"],
                 info=info, abstained=payload["abstained"], region_note=payload["region_note"], model_meta=MODEL_META)
    return str(out)


def _history_choices(entries):
    return [(f"{e.timestamp} · {e.prediction} · {e.file_name}", e.id) for e in (entries or [])]


# --------------------------------------------------------------------------- #
# Event handlers
# --------------------------------------------------------------------------- #
def _no_selection():
    return {"box": None, "pending": None}


def on_upload(file_path):
    """Validate the upload and prime the crop preview."""
    if not file_path:
        return None, _no_selection(), None, "", gr.Column(visible=False), 0, 0, 100, 100
    try:
        loaded = load_and_validate(file_path)
    except ImageValidationError as exc:
        return None, _no_selection(), None, render.error_box(str(exc)), gr.Column(visible=False), 0, 0, 100, 100
    except Exception as exc:  # noqa: BLE001
        return None, _no_selection(), None, render.error_box(f"Unexpected error while reading the image: {exc}"), gr.Column(visible=False), 0, 0, 100, 100
    w, h = loaded.original_size
    note = f"<div class='pd-note'><b>{render.esc(loaded.file_name)}</b> · {w}×{h} px · {loaded.format}" + \
           (f" · downscaled to {loaded.image.width}×{loaded.image.height} for processing" if loaded.was_downscaled else "") + \
           " · Click two opposite corners on the preview (or use the sliders) to select a region, or analyse the full image.</div>"
    state = {"image": loaded.image, "file_name": loaded.file_name}
    return state, _no_selection(), cropper.draw_selection(loaded.image, None), note, gr.Column(visible=True), 0, 0, 100, 100


def on_preview_select(state, sel, evt: gr.SelectData):
    """Two clicks on the preview define the crop box."""
    if not state:
        return sel, None, 0, 0, 100, 100
    img = state["image"]
    x, y = int(evt.index[0]), int(evt.index[1])
    x, y = max(0, min(img.width, x)), max(0, min(img.height, y))
    sel = dict(sel or _no_selection())
    if sel.get("pending") is None:
        sel["pending"] = (x, y)
        box = sel.get("box")
    else:
        px, py = sel["pending"]
        x0, x1 = sorted((px, x)); y0, y1 = sorted((py, y))
        if x1 - x0 < config.MIN_CROP_SIDE or y1 - y0 < config.MIN_CROP_SIDE:
            # Too small: keep the first corner and wait for a better second click.
            return sel, cropper.draw_selection(img, sel.get("box"), sel["pending"]), *_pct(img, sel.get("box"))
        sel["box"], sel["pending"] = (x0, y0, x1, y1), None
        box = sel["box"]
    return sel, cropper.draw_selection(img, box, sel.get("pending")), *_pct(img, box)


def _pct(img, box):
    if box is None:
        return 0, 0, 100, 100
    return tuple(round(v, 1) for v in cropper.percent_from_box(img, box))


def on_sliders(state, sel, left, top, right, bottom):
    if not state:
        return sel, None
    img = state["image"]
    box = cropper.box_from_percent(img, left, top, right, bottom)
    sel = dict(sel or _no_selection())
    sel["pending"] = None
    sel["box"] = None if cropper.is_full(img, box) else box
    return sel, cropper.draw_selection(img, sel["box"])


def reset_crop(state):
    if not state:
        return _no_selection(), None, 0, 0, 100, 100
    return _no_selection(), cropper.draw_selection(state["image"], None), 0, 0, 100, 100


def analyze(mode: str, state, sel, history, progress=gr.Progress()):
    """mode: 'full' | 'selection' | 'compare'."""
    empty = ("", "", None, None, "", "", "")
    if PREDICTOR is None:
        return (*empty[:0], render.error_box(f"The trained model could not be loaded. {MODEL_ERROR}"), "", None, None, "", "", "",
                None, history, render.history_list(history), gr.Dropdown(choices=_history_choices(history)), gr.Column(visible=True), gr.Tabs())
    if not state:
        return (render.error_box("Please upload a leaf image first."), "", None, None, "", "", "",
                None, history, render.history_list(history), gr.Dropdown(choices=_history_choices(history)), gr.Column(visible=True), gr.Tabs())
    try:
        progress(0.05, desc="Preparing image")
        full_img: Image.Image = state["image"]
        box = (sel or {}).get("box")
        is_cropped = not cropper.is_full(full_img, box)
        if mode == "compare" and not is_cropped:
            raise ImageValidationError("To compare, first select a region on the preview (otherwise the two images are identical).")
        if mode == "full" or (mode == "selection" and not is_cropped):
            target, note = full_img, f"Full image · {full_img.width}×{full_img.height} px"
        else:
            sel_img = crop_image(full_img, box)  # validates the minimum size; this crop IS the model input
            target, note = sel_img, f"Selected region · {sel_img.width}×{sel_img.height} px at ({box[0]}, {box[1]})"

        compare = None
        if mode == "compare":
            progress(0.25, desc="Analysing full image")
            full_payload = _run(full_img, state["file_name"], f"Full image · {full_img.width}×{full_img.height} px")
            progress(0.55, desc="Analysing selected region")
            payload = _run(target, state["file_name"], note)
            compare = {"full": full_payload, "crop": payload}
        else:
            progress(0.3, desc="Running the model and Grad-CAM")
            payload = _run(target, state["file_name"], note)
        progress(0.85, desc="Building report")
        report_path = _make_report(payload)
        outputs = _render(payload, compare)
        label = "No confident result" if payload["abstained"] else f"{payload['top']['crop']} — {payload['top']['condition']}"
        entry = hist.new_entry(payload["images"]["analysed"], state["file_name"], label,
                               None if payload["abstained"] else payload["top"]["probability"], payload["abstained"],
                               {"payload": payload, "compare": compare, "report": report_path})
        history = hist.push(history, entry)
        progress(1.0, desc="Done")
        return (*outputs, report_path, history, render.history_list(history),
                gr.Dropdown(choices=_history_choices(history), value=entry.id), gr.Column(visible=True), gr.Tabs(selected="analyze"))
    except ImageValidationError as exc:
        return (render.error_box(str(exc)), "", None, None, "", "", "", None, history, render.history_list(history),
                gr.Dropdown(choices=_history_choices(history)), gr.Column(visible=True), gr.Tabs(selected="analyze"))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return (render.error_box(f"The analysis failed unexpectedly: {exc}"), "", None, None, "", "", "", None, history,
                render.history_list(history), gr.Dropdown(choices=_history_choices(history)), gr.Column(visible=True), gr.Tabs(selected="analyze"))


def reopen(entry_id, history):
    e = hist.find(history, entry_id)
    if not e:
        return (render.error_box("That history entry is no longer available."), "", None, None, "", "", "", None, gr.Column(visible=True), gr.Tabs(selected="analyze"))
    outputs = _render(e.payload["payload"], e.payload.get("compare"))
    return (*outputs, e.payload.get("report"), gr.Column(visible=True), gr.Tabs(selected="analyze"))


def clear_all():
    return (None, _no_selection(), None, None, "", gr.Column(visible=False), 0, 0, 100, 100, "", "", None, None, "", "", "", None, gr.Column(visible=False))


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def build() -> gr.Blocks:
    with gr.Blocks(title=f"{config.APP_NAME} — {config.TAGLINE}") as demo:
        image_state = gr.State(None)
        selection_state = gr.State(_no_selection())
        history_state = gr.State([])
        gr.HTML(render.masthead(PREDICTOR is not None, MODEL_NOTE))
        if MODEL_ERROR:
            gr.HTML(render.error_box(f"The trained model could not be loaded, so analysis is disabled. {MODEL_ERROR}"))

        with gr.Tabs() as tabs:
            # ----------------------------------------------------------- Home
            with gr.Tab("Home", id="home"):
                gr.HTML(render.hero(PREDICTOR.num_classes if PREDICTOR else config.NUM_CLASSES, CROPS))
                with gr.Row():
                    start_btn = gr.Button("Upload Leaf Image", variant="primary", size="lg", scale=0, min_width=240)
                gr.HTML('<div class="pd-section-title">Supported crops and conditions</div><div class="pd-sub">The model recognises these 38 categories and nothing else.</div>')
                gr.HTML(render.crops_grid())
                gr.HTML("<div style='height:14px'></div>" + render.disclaimer_box())

            # ------------------------------------------------------- Analyse
            with gr.Tab("Analyse", id="analyze"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320):
                        gr.HTML('<div class="pd-section-title" style="margin-top:6px">Analysis workspace</div><div class="pd-sub">Upload a leaf photo, optionally crop to the area of concern, then analyse.</div>')
                        upload = gr.Image(type="filepath", label="Leaf image (JPG, PNG, WEBP)", sources=["upload", "clipboard"],
                                          height=300, elem_classes=["pd-upload"], image_mode=None)
                        upload_note = gr.HTML("")
                        sample_files = sorted(str(p) for p in (config.ASSETS_DIR / "samples").glob("*.jpg"))
                        if sample_files:
                            gr.Examples(examples=[[f] for f in sample_files], inputs=upload, label="Or try a sample leaf (PlantVillage test images, never used for training)",
                                        examples_per_page=6)
                        with gr.Column(visible=False) as crop_col:
                            preview = gr.Image(type="pil", label="Select a region (optional) — click two opposite corners", interactive=False,
                                               sources=[], height=380, elem_id="pd-preview", buttons=["fullscreen"])
                            with gr.Accordion("Fine-tune the selection", open=False):
                                with gr.Row():
                                    sl_left = gr.Slider(0, 100, value=0, step=0.5, label="Left %")
                                    sl_top = gr.Slider(0, 100, value=0, step=0.5, label="Top %")
                                with gr.Row():
                                    sl_right = gr.Slider(0, 100, value=100, step=0.5, label="Right %")
                                    sl_bottom = gr.Slider(0, 100, value=100, step=0.5, label="Bottom %")
                            with gr.Row():
                                analyze_full_btn = gr.Button("Analyse full image", variant="primary")
                                analyze_sel_btn = gr.Button("Analyse selection", variant="secondary")
                            with gr.Row():
                                compare_btn = gr.Button("Compare full vs selection", variant="secondary", size="sm")
                                reset_btn = gr.Button("Reset crop", variant="secondary", size="sm")
                                clear_btn = gr.Button("Clear", variant="secondary", size="sm")
                    with gr.Column(scale=7, min_width=360):
                        with gr.Column(visible=False) as results_col:
                            gr.HTML('<div class="pd-section-title" style="margin-top:6px">Results</div>')
                            result_html = gr.HTML("")
                            compare_html = gr.HTML("")
                            chart_html = gr.HTML("")
                            gr.HTML('<div class="pd-section-title" style="font-size:18px">Visual explanation</div><div class="pd-sub">Drag the slider to compare the analysed image with the Grad-CAM overlay.</div>')
                            slider = gr.ImageSlider(type="pil", label="Analysed image ↔ Grad-CAM overlay", height=360, buttons=["fullscreen"])
                            gallery = gr.Gallery(label="Original · Heatmap · Overlay", columns=3, rows=1, height=200, object_fit="contain", type="pil", buttons=["download", "fullscreen"])
                            gr.HTML('<div class="pd-note">The highlighted regions show where the model focused when producing its prediction. This visualization does not prove that the highlighted region contains the disease.</div>')
                            quality_html = gr.HTML("")
                            info_html = gr.HTML("")
                            with gr.Row():
                                report_btn = gr.DownloadButton("Download PDF report", variant="secondary", size="sm")
                                another_btn = gr.Button("Analyse another image", variant="primary", size="sm")
                                reset_all_btn = gr.Button("Reset analysis", variant="secondary", size="sm")
                            with gr.Accordion("How did the AI decide?", open=False):
                                gr.HTML(render.educational())
                        gr.HTML(render.disclaimer_box())

            # ------------------------------------------------------- History
            with gr.Tab("History", id="history"):
                gr.HTML('<div class="pd-section-title" style="margin-top:6px">Session history</div><div class="pd-sub">Recent analyses from this browser session. Nothing is stored on the server.</div>')
                with gr.Row():
                    history_dd = gr.Dropdown(choices=[], label="Reopen a previous result", scale=4)
                    reopen_btn = gr.Button("Reopen", variant="primary", scale=1, min_width=120)
                history_html = gr.HTML(render.history_list([]))

            # --------------------------------------------------------- About
            with gr.Tab("About & model", id="about"):
                gr.HTML(render.about_page(MODEL_META, EVAL_METRICS, TRAINING_LOG))

        # ----------------------------------------------------------- Wiring
        analysis_outputs = [result_html, chart_html, slider, gallery, quality_html, info_html, compare_html, report_btn,
                            history_state, history_html, history_dd, results_col, tabs]
        start_btn.click(lambda: gr.Tabs(selected="analyze"), outputs=tabs)
        sliders = [sl_left, sl_top, sl_right, sl_bottom]
        upload.change(on_upload, inputs=upload, outputs=[image_state, selection_state, preview, upload_note, crop_col, *sliders])
        preview.select(on_preview_select, inputs=[image_state, selection_state], outputs=[selection_state, preview, *sliders])
        for sl in sliders:
            sl.release(on_sliders, inputs=[image_state, selection_state, *sliders], outputs=[selection_state, preview])
        reset_btn.click(reset_crop, inputs=image_state, outputs=[selection_state, preview, *sliders])
        analyze_full_btn.click(lambda s, e, h, p=gr.Progress(): analyze("full", s, e, h, p), inputs=[image_state, selection_state, history_state], outputs=analysis_outputs)
        analyze_sel_btn.click(lambda s, e, h, p=gr.Progress(): analyze("selection", s, e, h, p), inputs=[image_state, selection_state, history_state], outputs=analysis_outputs)
        compare_btn.click(lambda s, e, h, p=gr.Progress(): analyze("compare", s, e, h, p), inputs=[image_state, selection_state, history_state], outputs=analysis_outputs)
        reopen_btn.click(reopen, inputs=[history_dd, history_state],
                         outputs=[result_html, chart_html, slider, gallery, quality_html, info_html, compare_html, report_btn, results_col, tabs])
        clear_outputs = [image_state, selection_state, upload, preview, upload_note, crop_col, *sliders, result_html, chart_html, slider, gallery, quality_html, info_html, compare_html, report_btn, results_col]
        clear_btn.click(clear_all, outputs=clear_outputs)
        another_btn.click(clear_all, outputs=clear_outputs)
        reset_all_btn.click(clear_all, outputs=clear_outputs)
    return demo


demo = build()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=port,
        inbrowser=os.environ.get("PLANT_DOCTOR_OPEN_BROWSER", "0") == "1",
        theme=gr.themes.Base(primary_hue="green", neutral_hue="stone", font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]),
        css=CSS,
        head=HEAD_HTML,
        show_error=True,
        max_file_size=f"{config.MAX_FILE_SIZE_MB}mb",
        footer_links=[],
    )
