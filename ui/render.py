"""HTML renderers for the Plant Doctor interface (pure functions, easy to test)."""
from __future__ import annotations

import html
import json

import config
from core import disease_info
from ui.styles import LOGO_SVG


def esc(s) -> str:
    return html.escape(str(s))


def masthead(model_ok: bool, model_note: str) -> str:
    dot = "dot" if model_ok else "dot err"
    return f"""
<div class="pd-masthead">
  <div class="pd-brand">{LOGO_SVG}<div><div class="name">{config.APP_NAME}</div><div class="tag">{esc(config.TAGLINE)}</div></div></div>
  <div class="pd-status"><span class="{dot}"></span>{esc(model_note)}</div>
</div>"""


def hero(num_classes: int, crops: list[str]) -> str:
    chips = "".join(f'<span class="chip">{esc(c)}</span>' for c in crops)
    return f"""
<div class="pd-hero">
  <div class="eyebrow">AI-powered plant health screening</div>
  <h1>Understand your plants.<br>Protect your harvest.</h1>
  <p>Upload a photograph of a leaf and Plant Doctor screens it with a neural network trained on {num_classes} healthy and diseased leaf categories across {len(crops)} crops, then shows you exactly which parts of the image drove the result.</p>
  <div class="chips">{chips}</div>
</div>
<div class="pd-steps">
  <div class="pd-step"><div class="num">1</div><h3>Upload a leaf photo</h3><p>JPG, PNG or WEBP. Fill the frame with a single leaf against a plain background for best results.</p></div>
  <div class="pd-step"><div class="num">2</div><h3>Crop and analyse</h3><p>Optionally crop to the suspicious area. The model only sees the pixels you select.</p></div>
  <div class="pd-step"><div class="num">3</div><h3>Review the evidence</h3><p>See the most likely condition, alternatives, a Grad-CAM heatmap and sourced guidance on prevention.</p></div>
</div>"""


def crops_grid() -> str:
    cards = "".join(
        f'<div class="pd-crop"><b>{esc(crop)}</b><span>{esc(", ".join(conds))}</span></div>'
        for crop, conds in disease_info.crops_summary().items()
    )
    return f'<div class="pd-crops">{cards}</div>'


def disclaimer_box() -> str:
    return (
        '<div class="pd-disclaimer"><b>Educational use only.</b> Plant Doctor is a screening aid trained on laboratory-style '
        'leaf photographs. Its output is not a diagnosis and not a treatment recommendation. Confirm suspected diseases with a '
        'plant diagnostic laboratory or your local agricultural extension service before acting.</div>'
    )


def _bar(name: str, sub: str, pct: float, alt: bool) -> str:
    return (
        f'<div class="pd-bar{" alt" if alt else ""}" title="{esc(name)}: {pct:.1f}%"><div class="name"><b>{esc(name)}</b><span>{esc(sub)}</span></div>'
        f'<div class="track"><div class="fill" style="width:{max(1.5, pct):.1f}%"></div></div><div class="pct">{pct:.1f}%</div></div>'
    )


def predictions_chart(top: dict, alternatives: list[dict]) -> str:
    rows = [_bar(top["condition"], top["crop"], top["probability"] * 100, False)]
    rows += [_bar(a["condition"], a["crop"], a["probability"] * 100, True) for a in alternatives]
    return (
        '<div class="pd-card"><div class="pd-result"><div class="main"><div class="eyebrow">Model predictions</div>'
        f'<div class="pd-bars">{"".join(rows)}</div>'
        '<div class="pd-note">Softmax probabilities show the model\'s relative preference among its 38 training classes. '
        'They are not calibrated probabilities that the result is correct for your plant.</div></div></div></div>'
    )


def _ring(pct: float, good: bool) -> str:
    color = "#1f4d2e" if good else "#b7791f"
    return (
        f'<div class="pd-ring" style="background: conic-gradient({color} {pct:.1f}%, #d9e5db 0)"><div class="inner">'
        f'<span class="val" style="font-size:24px;font-weight:700;color:{color}">{pct:.0f}%</span></div></div>'
    )


def result_card(top: dict, quality: dict, region_note: str, inference_ms: float) -> str:
    healthy = top["is_healthy"]
    badge = ('<span class="pd-badge ok">● Healthy leaf pattern</span>' if healthy
             else '<span class="pd-badge warn">▲ Possible disease pattern</span>')
    warns = quality.get("warnings", [])
    warn_html = ("<ul class='pd-warnlist'>" + "".join(f"<li>{esc(w)}</li>" for w in warns) + "</ul>") if warns else ""
    pct = top["probability"] * 100
    return f"""
<div class="pd-card"><div class="pd-result">
  <div class="main">
    <div class="eyebrow">Most likely condition</div>
    <h2>{esc(top['condition'])}</h2>
    <div class="crop">{esc(top['crop'])} &nbsp;·&nbsp; {esc(region_note)} &nbsp;·&nbsp; {inference_ms:.0f} ms</div>
    {badge}{warn_html}
    <div class="pd-note">This is a screening suggestion from an image-classification model, not a confirmed diagnosis.</div>
  </div>
  <div class="pd-conf">{_ring(pct, pct >= config.MIN_CONFIDENCE * 100)}<div class="lbl">model confidence<br>(softmax probability)</div></div>
</div></div>"""


def abstain_card(quality: dict, top: dict | None, region_note: str) -> str:
    reasons = "".join(f"<li>{esc(r)}</li>" for r in quality.get("reasons", []))
    best = ""
    if top:
        best = (f"<p style='margin-top:10px'>For transparency, the model's strongest (but unreliable) guess was "
                f"<b>{esc(top['crop'])} — {esc(top['condition'])}</b> at {top['probability']*100:.0f}%.</p>")
    return f"""
<div class="pd-abstain">
  <h2>No confident result</h2>
  <p>{esc(config.ABSTAIN_MESSAGE)}</p>
  <ul>{reasons}</ul>{best}
  <p style="margin-top:12px;font-size:13px">{esc(region_note)}. Tips: fill the frame with one leaf, use even daylight, hold the camera steady, and make sure the crop is one of the supported types.</p>
</div>"""


def quality_panel(quality: dict) -> str:
    nov = quality.get("novelty_score")
    nov_txt = "n/a" if nov is None else f"{nov:.2f}"
    return f"""
<div class="pd-kv">
  <div><b>{quality.get('sharpness', 0):.0f}</b><span>sharpness score (min {config.BLUR_THRESHOLD:.0f})</span></div>
  <div><b>{quality.get('vegetation_fraction', 0)*100:.0f}%</b><span>plant-coloured area</span></div>
  <div><b>{nov_txt}</b><span>novelty score (limit {config.OOD_SCORE_LIMIT:.2f})</span></div>
  <div><b>{quality.get('top_confidence', 0)*100:.0f}%</b><span>top confidence (min {config.MIN_CONFIDENCE*100:.0f}%)</span></div>
</div>
<div class="pd-note">Image checks run before a condition is named. Sharpness is the variance of the Laplacian; the novelty score compares the network's internal features with those of training leaves (1.00 = more unusual than 99% of validation images). These checks are heuristics and can be wrong in both directions.</div>"""


def info_panel(class_name: str) -> str:
    try:
        info = disease_info.get_info(class_name)
    except KeyError:
        return '<div class="pd-card pd-info"><p>No information is available for this class.</p></div>'
    sci = f'<div class="sci">{esc(info["scientific_name"])} · {esc(info.get("pathogen_type", ""))}</div>' if info.get("scientific_name") else ""
    conf = info.get("information_confidence", "")
    conf_note = ' <span class="pd-badge neutral" style="margin:0 0 0 8px;padding:3px 9px;font-size:11px">limited published information</span>' if conf == "limited" else ""

    def ul(key):
        return "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in info.get(key, [])) + "</ul>"

    refs = " · ".join(f'<a href="{esc(r["url"])}" target="_blank" rel="noopener">{esc(r["name"])}</a>' for r in info.get("references", []))
    return f"""
<div class="pd-card pd-info">
  <div class="eyebrow" style="text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:600;color:#5b6b5e">About this condition</div>
  <h2 style="font-family:'Fraunces',Georgia,serif;font-size:24px;margin:4px 0 2px;color:#1f4d2e">{esc(info['common_name'])}{conf_note}</h2>
  {sci}
  <p style="margin-top:10px">{esc(info['description'])}</p>
  <h3>Typical symptoms</h3>{ul('symptoms')}
  <h3>Conditions that favour it</h3>{ul('environmental_factors')}
  <h3>General prevention</h3>{ul('prevention')}
  <h3>Suggested next steps</h3>{ul('next_steps')}
  <div class="refs"><b>References:</b> {refs}<br>{esc(disease_info.disclaimer())}</div>
</div>"""


def compare_panel(full: dict, crop: dict) -> str:
    same = full["top"]["class_name"] == crop["top"]["class_name"]
    if full.get("abstained") or crop.get("abstained"):
        verdict = "One of the two analyses did not pass the confidence checks, so the comparison is inconclusive."
    elif same:
        verdict = (f"Both views agree on <b>{esc(crop['top']['condition'])}</b>. Agreement between the whole leaf and a close-up "
                   "increases (but does not guarantee) the reliability of the result.")
    else:
        verdict = ("The predictions differ. This usually happens when the selected region isolates a lesion that is small relative to "
                   "the whole leaf, when the crop removes context the model relies on (leaf shape, colour), or when the region is "
                   "ambiguous. Treat differing results as a reason for extra caution and consider a sharper, closer photograph.")

    def col(title, r):
        t = r["top"]
        status = "No confident result" if r.get("abstained") else f"{t['crop']} — {t['condition']}"
        return (f'<div class="col"><h4>{esc(title)}</h4><div class="big">{esc(status)}</div>'
                f'<div><span class="pct">{t["probability"]*100:.1f}%</span> <span style="color:#5b6b5e;font-size:12.5px">confidence · {r["input_size"][0]}×{r["input_size"][1]} px analysed</span></div></div>')

    return f'<div class="pd-card"><div class="eyebrow" style="font-size:11px;text-transform:uppercase;letter-spacing:.12em;font-weight:600;color:#5b6b5e;margin-bottom:10px">Full image vs selected region</div><div class="pd-compare">{col("Full image", full)}{col("Selected region", crop)}</div><p class="pd-note">{verdict}</p></div>'


def history_list(entries) -> str:
    if not entries:
        return '<div class="pd-empty">No analyses yet in this session. Results appear here as you analyse images; nothing is stored after you close the tab.</div>'
    items = []
    for e in entries:
        conf = "—" if e.confidence is None else f"{e.confidence*100:.0f}%"
        cls = "c abst" if e.abstained else "c"
        items.append(
            f'<div class="pd-hist"><img src="data:image/jpeg;base64,{e.thumbnail_b64}" alt=""><div><div class="t">{esc(e.prediction)}</div>'
            f'<div class="m">{esc(e.file_name)} · {esc(e.timestamp)} · id {esc(e.id)}</div></div><div class="{cls}">{conf}</div></div>'
        )
    return '<div class="pd-history">' + "".join(items) + "</div>"


def error_box(message: str) -> str:
    return f'<div class="pd-error"><b>We couldn\'t use that image.</b> {esc(message)}</div>'


def educational() -> str:
    return """
<div class="pd-card pd-about">
  <h2 style="margin-top:0">How did the AI decide?</h2>
  <p><b>Image classification</b> means the computer looks at the whole picture and chooses one label from a fixed list — here, 38 combinations of crop and condition. It does not understand plants; it has learned statistical patterns of colour, texture and shape that separated those 38 categories in its training photographs.</p>
  <p><b>EfficientNet-B0</b> is the neural network doing the looking. It is a stack of small filters that first detect edges and colour blobs, then combine them into textures (speckles, rings, powdery patches) and finally into evidence for each category. It was pre-trained on millions of everyday photographs and then fine-tuned on leaf images, which is why it can learn from a comparatively small plant dataset.</p>
  <p><b>Softmax</b> turns the network's 38 raw scores into numbers between 0 and 100% that add up to 100%. A high value means the network strongly prefers that category <i>over the other 37</i>; it does not mean the category is correct, and it cannot say "none of these" — which is why Plant Doctor adds separate sharpness, novelty and confidence checks and sometimes declines to answer.</p>
  <p><b>Grad-CAM</b> asks: if the score for the chosen category went up a little, which parts of the last feature map would be responsible? Those regions are coloured warm on the heatmap. It shows where the model looked, not what the disease is: a heatmap over a lesion is reassuring, a heatmap over the background or the leaf edge is a warning sign that the prediction may be based on the wrong cues.</p>
</div>"""


def about_page(meta: dict, metrics: dict | None, training: dict | None) -> str:
    val = meta.get("val_metrics", {}) if meta else {}
    off = (metrics or {}).get("official_test", {})
    strict = (metrics or {}).get("strict_test_no_leaf_overlap", {})
    hist = (training or {}).get("history", [])

    def pct(v):
        return "—" if v in (None, "") else f"{v*100:.2f}%"

    rows = "".join(
        f"<tr><td>{h['epoch']}</td><td>{h['train_loss']:.4f}</td><td>{pct(h['train_accuracy'])}</td><td>{h['val_loss']:.4f}</td><td>{pct(h['val_accuracy'])}</td><td>{pct(h['val_macro_f1'])}</td></tr>"
        for h in hist
    )
    test_table = ""
    if off:
        test_table = f"""
<table><tr><th>Metric</th><th>Official test split ({off.get('images')} images)</th><th>Strict subset, no leaf overlap ({strict.get('images')} images)</th></tr>
<tr><td>Accuracy</td><td>{pct(off.get('accuracy'))}</td><td>{pct(strict.get('accuracy'))}</td></tr>
<tr><td>Top-3 accuracy</td><td>{pct(off.get('top3_accuracy'))}</td><td>{pct(strict.get('top3_accuracy'))}</td></tr>
<tr><td>Macro precision</td><td>{pct(off.get('macro_precision'))}</td><td>{pct(strict.get('macro_precision'))}</td></tr>
<tr><td>Macro recall</td><td>{pct(off.get('macro_recall'))}</td><td>{pct(strict.get('macro_recall'))}</td></tr>
<tr><td>Macro F1</td><td>{pct(off.get('macro_f1'))}</td><td>{pct(strict.get('macro_f1'))}</td></tr></table>"""
    else:
        test_table = "<p><i>The held-out test evaluation has not been run yet (python -m training.evaluate).</i></p>"
    crops = disease_info.crops_summary()
    return f"""
<div class="pd-card pd-about">
  <h2 style="margin-top:0">How Plant Doctor works</h2>
  <p>You upload a leaf photograph (optionally cropped). The image is converted to RGB, resized to 224×224 pixels and normalised with the same statistics used in training. A convolutional neural network produces 38 scores, softmax converts them to percentages, and separate checks on sharpness, feature-space novelty and confidence decide whether a result is shown at all. Grad-CAM then back-propagates the chosen class score to the last convolutional layer to show which regions influenced the prediction.</p>

  <h2>Model</h2>
  <table>
    <tr><th>Architecture</th><td>EfficientNet-B0 (torchvision), ImageNet-pretrained, all layers fine-tuned; classifier replaced by Dropout(0.2) → Linear(1280, 38)</td></tr>
    <tr><th>Input</th><td>224×224 RGB, ImageNet mean/std normalisation</td></tr>
    <tr><th>Classes</th><td>38 (14 crops; 26 diseases/pests + 12 healthy classes)</td></tr>
    <tr><th>Training</th><td>AdamW (lr 3e-4, weight decay 0.01), label-smoothed cross-entropy (0.1), linear warm-up + cosine decay, batch 64, augmentation (random resized crop, flips, rotation, colour jitter) on training images only, early stopping on validation macro-F1, fixed seed 42</td></tr>
    <tr><th>Checkpoint</th><td>epoch {meta.get('epoch', '—')} · validation accuracy {pct(val.get('accuracy'))} · validation macro-F1 {pct(val.get('macro_f1'))}</td></tr>
    <tr><th>Inference</th><td>CPU by default (about 50–150 ms per image on a modern laptop), GPU used automatically if available; the model is loaded once and cached</td></tr>
  </table>

  <h2>Dataset and attribution</h2>
  <p>Trained on the <b>PlantVillage</b> dataset (Hughes &amp; Salathé, 2015; Mohanty, Hughes &amp; Salathé, 2016, <i>Frontiers in Plant Science</i> 7:1419), colour images, 54,305 photographs of single leaves on uniform backgrounds. Obtained from the Hugging Face dataset <code>mohanty/PlantVillage</code>, whose dataset card declares the licence <b>CC BY-SA 3.0</b>. Attribution is required and derivatives must be shared under the same licence; verify the terms yourself before any commercial use, since the original repository does not state a licence file. Splits: the official train/test lists were used; a validation set of 10% of training <i>leaves</i> per class was held out using the dataset's leaf-identity map so that photographs of the same physical leaf never appear on both sides. The test set was never used for tuning.</p>

  <h2>Evaluation</h2>
  {test_table}
  <table><tr><th>Epoch</th><th>Train loss</th><th>Train acc</th><th>Val loss</th><th>Val acc</th><th>Val macro-F1</th></tr>{rows}</table>
  <p>The full report with per-class F1 and the confusion matrix is in <code>artifacts/evaluation_report.md</code>.</p>

  <h2>Limitations — please read</h2>
  <ul>
    <li><b>Benchmark ≠ field.</b> High accuracy on PlantVillage test images means the model separates these 38 classes <i>under laboratory photo conditions</i>. Published work shows accuracy drops sharply on field photographs with complex backgrounds, multiple leaves, shadows or different cameras. Plant Doctor has not been validated on field images.</li>
    <li><b>Closed world.</b> Only {len(crops)} crops and 38 conditions are known. Other crops, nutrient disorders, herbicide injury, most pests and co-infections cannot be recognised; the novelty check catches only images that are far from the training data.</li>
    <li><b>Confidence is not probability of correctness.</b> Softmax values are not calibrated.</li>
    <li><b>Grad-CAM is a visualisation, not proof.</b> Highlighted regions show where the model looked, not where the disease is.</li>
    <li><b>No treatment advice.</b> The disease information is general educational material from extension sources; pesticide decisions require local, current guidance.</li>
  </ul>

  <h2>About Grad-CAM</h2>
  <p>Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017) computes the gradient of the selected class score with respect to the 7×7×1280 feature map of the last convolutional block, averages the gradients per channel to obtain importance weights, forms a weighted sum of the channels, keeps the positive part and upsamples it to the analysed image. It is computed live from the real network on every analysis. Because the map is only 7×7 before upsampling, it is coarse: it can point at a region, not at individual spots.</p>

  <h2>Privacy</h2>
  <p>Uploaded images are processed in memory for the current session only and are not saved by the application. The session history disappears when the page is closed.</p>
</div>"""
