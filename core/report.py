"""PDF results report (ReportLab)."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

import config

FOREST = colors.HexColor("#1f4d2e")
SAGE = colors.HexColor("#8aa88f")
MUTED = colors.HexColor("#5b6b5e")
LIGHT = colors.HexColor("#eef3ee")


def _img_flowable(image: Image.Image, width_mm: float) -> RLImage:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    buf.seek(0)
    w, h = image.size
    width = width_mm * mm
    return RLImage(buf, width=width, height=width * h / w)


def build_report(
    out_path: Path,
    *,
    analysed_image: Image.Image,
    heatmap: Image.Image | None,
    overlay: Image.Image | None,
    file_name: str,
    top: dict,
    alternatives: list[dict],
    quality: dict,
    info: dict | None,
    abstained: bool,
    region_note: str,
    model_meta: dict,
) -> Path:
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=FOREST, fontSize=20, alignment=TA_LEFT, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=MUTED, fontSize=9.5, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=FOREST, fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, textColor=MUTED, leading=11)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=0)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"{config.APP_NAME} report", author=config.APP_NAME)
    story = []
    story.append(Paragraph(f"{config.APP_NAME} — Leaf analysis report", h1))
    story.append(Paragraph(f"{config.TAGLINE} &nbsp;•&nbsp; Generated {datetime.now():%Y-%m-%d %H:%M} &nbsp;•&nbsp; File: {file_name} &nbsp;•&nbsp; {region_note}", sub))

    # Primary result block
    if abstained:
        headline = "No confident result"
        detail = config.ABSTAIN_MESSAGE
    else:
        headline = f"{top['crop']} — {top['condition']}"
        detail = f"Model confidence {top['probability']*100:.1f}% (softmax). Status: {'Healthy' if top['is_healthy'] else 'Possible disease detected'}."
    t = Table([[Paragraph(f"<b>{headline}</b>", ParagraphStyle('r', parent=body, fontSize=13, textColor=FOREST)),],
               [Paragraph(detail, body)]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.6, SAGE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)

    # Alternatives
    story.append(Paragraph("Model predictions", h2))
    rows = [["Rank", "Crop", "Condition", "Confidence"]]
    for i, p in enumerate([top] + alternatives, start=1):
        rows.append([str(i), p["crop"], p["condition"], f"{p['probability']*100:.1f}%"])
    at = Table(rows, colWidths=[14 * mm, 40 * mm, 90 * mm, 30 * mm])
    at.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), FOREST), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.3, SAGE),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT])]))
    story.append(at)
    story.append(Paragraph("Softmax confidence describes the model's relative preference among its 38 training classes; it is not a calibrated probability that the diagnosis is correct in the real world.", small))

    # Quality
    q_lines = [f"Sharpness score {quality.get('sharpness')}", f"Plant-coloured area {quality.get('vegetation_fraction', 0)*100:.0f}%"]
    if quality.get("novelty_score") is not None:
        q_lines.append(f"Novelty score {quality['novelty_score']:.2f} (1.00 = calibrated limit)")
    story.append(Paragraph("Image checks: " + " • ".join(q_lines), small))
    for r in quality.get("reasons", []):
        story.append(Paragraph(f"• {r}", small))
    for w in quality.get("warnings", []):
        story.append(Paragraph(f"• Warning: {w}", small))

    # Images
    story.append(Paragraph("Visual explanation (Grad-CAM)", h2))
    imgs = [_img_flowable(analysed_image, 54)]
    captions = ["Analysed image"]
    if heatmap is not None and overlay is not None:
        imgs += [_img_flowable(heatmap, 54), _img_flowable(overlay, 54)]
        captions += ["Grad-CAM heatmap", "Overlay"]
    it = Table([imgs, [Paragraph(c, small) for c in captions]], colWidths=[58 * mm] * len(imgs))
    it.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(it)
    story.append(Paragraph("The highlighted regions show where the model focused when producing its prediction. This visualization does not prove that the highlighted region contains the disease.", small))

    # Disease info
    if info and not abstained:
        story.append(Paragraph(f"About {info['common_name']}", h2))
        if info.get("scientific_name"):
            story.append(Paragraph(f"<i>{info['scientific_name']}</i> — {info.get('pathogen_type', '')}", small))
        story.append(Paragraph(info["description"], body))
        for title, key in (("Typical symptoms", "symptoms"), ("Conditions that favour it", "environmental_factors"),
                           ("General prevention", "prevention"), ("Suggested next steps", "next_steps")):
            block = [Paragraph(f"<b>{title}</b>", body)] + [Paragraph(f"• {x}", bullet) for x in info.get(key, [])]
            story.append(KeepTogether(block))
            story.append(Spacer(1, 3))
        refs = ", ".join(r["name"] for r in info.get("references", []))
        if refs:
            story.append(Paragraph(f"References: {refs}", small))

    # Model + disclaimer
    story.append(Paragraph("Model", h2))
    story.append(Paragraph(f"EfficientNet-B0 fine-tuned on PlantVillage (38 classes, 14 crops). Checkpoint epoch {model_meta.get('epoch', '?')}; "
                           f"validation macro-F1 {model_meta.get('val_metrics', {}).get('macro_f1', 0)*100:.1f}% on leaf-grouped validation data. "
                           "Benchmark accuracy on laboratory-style photographs does not transfer directly to field conditions.", small))
    story.append(Paragraph("Disclaimer", h2))
    story.append(Paragraph("Plant Doctor provides educational screening information generated by a machine-learning model. It is not a diagnosis "
                           "and not a substitute for advice from a qualified agronomist, plant pathologist or extension service. "
                           "Do not apply pesticides based solely on this report.", small))
    doc.build(story)
    return out_path
