"""Evaluate the trained checkpoint on the held-out test split and fit the novelty detector.

    python -m training.evaluate

Outputs (all under artifacts/):
* evaluation_metrics.json  – accuracy, macro/weighted precision/recall/F1, per-class metrics,
                             confusion matrix, strict-subset metrics
* evaluation_report.md     – human-readable report
* confusion_matrix.png, per_class_f1.png, training_curves.png
* ood_statistics.npz       – Mahalanobis novelty detector fitted on train features,
                             thresholds calibrated on validation features
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, top_k_accuracy_score)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from models.inference import Predictor, split_class_name  # noqa: E402
from models.quality import NoveltyDetector  # noqa: E402
from training.preprocessing import ManifestDataset, get_eval_transforms, load_manifest  # noqa: E402


def run_split(predictor: Predictor, rows: list[dict], batch_size: int, workers: int, desc: str):
    ds = ManifestDataset(rows, get_eval_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    logits, feats, labels = [], [], []
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        lg, ft = predictor.extract_features(x)
        logits.append(lg); feats.append(ft); labels.append(y.numpy())
        if (i + 1) % 20 == 0:
            print(f"[{desc}] {min((i+1)*batch_size, len(rows))}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return np.concatenate(logits), np.concatenate(feats), np.concatenate(labels)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def plot_outputs(cm: np.ndarray, class_names: list[str], per_class_f1: list[float], history: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = [c.replace("___", ": ").replace("_", " ")[:34] for c in class_names]
    fig, ax = plt.subplots(figsize=(16, 14))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(short))); ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=90, fontsize=7); ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalised confusion matrix (official PlantVillage test split)")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 11))
    order = np.argsort(per_class_f1)
    ax.barh([short[i] for i in order], [per_class_f1[i] for i in order], color="#2f6b3f")
    ax.set_xlim(0, 1); ax.set_xlabel("F1-score"); ax.set_title("Per-class F1 (test split)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(out / "per_class_f1.png", dpi=130); plt.close(fig)

    if history:
        ep = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="train loss", color="#8aa88f")
        axes[0].plot(ep, [h["val_loss"] for h in history], marker="o", label="val loss", color="#2f6b3f")
        axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].set_title("Loss")
        axes[1].plot(ep, [h["train_accuracy"] for h in history], marker="o", label="train acc", color="#8aa88f")
        axes[1].plot(ep, [h["val_accuracy"] for h in history], marker="o", label="val acc", color="#2f6b3f")
        axes[1].plot(ep, [h["val_macro_f1"] for h in history], marker="s", label="val macro F1", color="#b7791f")
        axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].set_title("Accuracy / F1")
        fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=130); plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", type=Path, default=config.ARTIFACTS_DIR / "splits")
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--model", type=Path, default=config.MODEL_PATH)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--ood-train-per-class", type=int, default=200, help="train images per class used to fit the novelty detector")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    out = args.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    predictor = Predictor(args.model)
    class_names = predictor.class_names
    test_rows = load_manifest(args.manifest_dir / "test.csv")
    val_rows = load_manifest(args.manifest_dir / "val.csv")
    train_rows = load_manifest(args.manifest_dir / "train.csv")

    # ---------------- Test split ----------------
    logits, feats_test, y = run_split(predictor, test_rows, args.batch_size, args.workers, "test")
    probs = softmax(logits)
    pred = probs.argmax(1)
    strict_mask = np.array([int(r.get("overlaps_train", 0)) == 0 for r in test_rows])

    def metrics(mask: np.ndarray) -> dict:
        yy, pp, pr = y[mask], pred[mask], probs[mask]
        return {
            "images": int(mask.sum()),
            "accuracy": float(accuracy_score(yy, pp)),
            "top3_accuracy": float(top_k_accuracy_score(yy, pr, k=3, labels=list(range(len(class_names))))),
            "macro_precision": float(precision_score(yy, pp, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yy, pp, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(yy, pp, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yy, pp, average="weighted", zero_division=0)),
        }

    official = metrics(np.ones_like(strict_mask, dtype=bool))
    strict = metrics(strict_mask)
    report = classification_report(y, pred, target_names=class_names, output_dict=True, zero_division=0)
    per_class = {c: {"precision": report[c]["precision"], "recall": report[c]["recall"], "f1": report[c]["f1-score"],
                     "support": int(report[c]["support"])} for c in class_names}
    cm = confusion_matrix(y, pred, labels=list(range(len(class_names))))

    # Confidence behaviour (how often the abstention threshold would fire on in-distribution data)
    top_conf = probs.max(1)
    conf_stats = {
        "mean_top_confidence": float(top_conf.mean()),
        "fraction_below_min_confidence": float((top_conf < config.MIN_CONFIDENCE).mean()),
        "accuracy_when_confident": float((pred[top_conf >= config.MIN_CONFIDENCE] == y[top_conf >= config.MIN_CONFIDENCE]).mean()),
        "accuracy_when_not_confident": float((pred[top_conf < config.MIN_CONFIDENCE] == y[top_conf < config.MIN_CONFIDENCE]).mean()) if (top_conf < config.MIN_CONFIDENCE).any() else None,
        "min_confidence_threshold": config.MIN_CONFIDENCE,
    }

    # ---------------- Novelty detector ----------------
    rng = random.Random(args.seed)
    by_class: dict[int, list[dict]] = {}
    for r in train_rows:
        by_class.setdefault(r["label"], []).append(r)
    fit_rows = []
    for c, rows in by_class.items():
        rng.shuffle(rows)
        fit_rows.extend(rows[: args.ood_train_per_class])
    _, feats_fit, y_fit = run_split(predictor, fit_rows, args.batch_size, args.workers, "ood-fit")
    _, feats_val, _ = run_split(predictor, val_rows, args.batch_size, args.workers, "ood-calib")
    detector = NoveltyDetector.fit(feats_fit, y_fit, feats_val)
    detector.save(out / config.OOD_STATS_PATH.name)
    test_nov = detector.raw_scores(feats_test) / detector.threshold
    ood_stats = {
        "fit_images": len(fit_rows), "calibration_images": len(val_rows),
        "calibration_percentiles_raw": detector.percentiles, "threshold_raw_p99": detector.threshold,
        "test_fraction_flagged_as_novel": float((test_nov > config.OOD_SCORE_LIMIT).mean()),
    }

    # ---------------- Write everything ----------------
    history = []
    tl = out / "training_log.json"
    if tl.exists():
        history = json.loads(tl.read_text()).get("history", [])
    plot_outputs(cm, class_names, [per_class[c]["f1"] for c in class_names], history, out)

    results = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.model), "checkpoint_epoch": predictor.meta.get("epoch"),
        "official_test": official, "strict_test_no_leaf_overlap": strict,
        "confidence": conf_stats, "per_class": per_class,
        "confusion_matrix": cm.tolist(), "class_names": class_names, "novelty_detector": ood_stats,
        "training_history": history,
    }
    (out / "evaluation_metrics.json").write_text(json.dumps(results, indent=2))

    worst = sorted(class_names, key=lambda c: per_class[c]["f1"])[:5]
    pairs = []
    cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
    for _ in range(5):
        i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
        if cm_off[i, j] == 0:
            break
        pairs.append((class_names[i], class_names[j], int(cm_off[i, j]))); cm_off[i, j] = 0

    def pct(v): return f"{v*100:.2f}%"
    md = [
        "# Plant Doctor — Model Evaluation Report", "",
        f"Generated: {results['evaluated_utc']}  ", f"Checkpoint: `{Path(args.model).name}` (best epoch {predictor.meta.get('epoch')})  ",
        "Model: EfficientNet-B0 (ImageNet-pretrained, fully fine-tuned), 38 classes, 224×224 input", "",
        "## 1. Benchmark performance (PlantVillage official test split)", "",
        "| Metric | Official test split | Strict subset (no leaf-ID overlap with train) |", "|---|---|---|",
        f"| Images | {official['images']} | {strict['images']} |",
        f"| Accuracy | {pct(official['accuracy'])} | {pct(strict['accuracy'])} |",
        f"| Top-3 accuracy | {pct(official['top3_accuracy'])} | {pct(strict['top3_accuracy'])} |",
        f"| Macro precision | {pct(official['macro_precision'])} | {pct(strict['macro_precision'])} |",
        f"| Macro recall | {pct(official['macro_recall'])} | {pct(strict['macro_recall'])} |",
        f"| Macro F1 | {pct(official['macro_f1'])} | {pct(strict['macro_f1'])} |",
        f"| Weighted F1 | {pct(official['weighted_f1'])} | {pct(strict['weighted_f1'])} |", "",
        "The *strict subset* removes test images whose leaf identifier (from the dataset's leaf map) also occurs in the "
        "training/validation lists. The official split is kept for comparability with published results.", "",
        "## 2. Training curves", "",
        "| Epoch | Train loss | Train acc | Val loss | Val acc | Val macro F1 | Minutes |", "|---|---|---|---|---|---|---|",
        *[f"| {h['epoch']} | {h['train_loss']:.4f} | {pct(h['train_accuracy'])} | {h['val_loss']:.4f} | {pct(h['val_accuracy'])} | {pct(h['val_macro_f1'])} | {h['epoch_minutes']:.1f} |" for h in history],
        "", "![curves](training_curves.png)", "",
        "## 3. Per-class F1 (test split)", "",
        "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
        *[f"| {c} | {per_class[c]['precision']:.3f} | {per_class[c]['recall']:.3f} | {per_class[c]['f1']:.3f} | {per_class[c]['support']} |" for c in class_names],
        "", "![per-class F1](per_class_f1.png)", "",
        "**Weakest classes:** " + ", ".join(f"{c} (F1 {per_class[c]['f1']:.3f})" for c in worst), "",
        "**Most frequent confusions (true → predicted, count):** " + ("; ".join(f"{a} → {b} ({n})" for a, b, n in pairs) or "none"), "",
        "## 4. Confusion matrix", "", "![confusion matrix](confusion_matrix.png)", "",
        "## 5. Confidence and abstention on in-distribution data", "",
        f"* Mean top-1 softmax probability: {pct(conf_stats['mean_top_confidence'])}",
        f"* Test images below the {pct(config.MIN_CONFIDENCE)} confidence threshold (would trigger abstention): {pct(conf_stats['fraction_below_min_confidence'])}",
        f"* Accuracy among confident predictions: {pct(conf_stats['accuracy_when_confident'])}",
        f"* Accuracy among low-confidence predictions: {pct(conf_stats['accuracy_when_not_confident']) if conf_stats['accuracy_when_not_confident'] is not None else 'n/a'}", "",
        "Softmax probabilities are **not** calibrated estimates of real-world correctness; they describe the model's "
        "relative preference among the 38 training classes only.", "",
        "## 6. Novelty (out-of-distribution) detector", "",
        f"* Fitted on {ood_stats['fit_images']} training images ({args.ood_train_per_class} per class), calibrated on {ood_stats['calibration_images']} validation images.",
        f"* Raw Mahalanobis distance percentiles on validation: " + ", ".join(f"{k}={v:.1f}" for k, v in detector.percentiles.items()),
        f"* Test images flagged as novel at the default limit: {pct(ood_stats['test_fraction_flagged_as_novel'])} (expected ≈1% by construction).", "",
        "## 7. Benchmark vs. real-world performance (please read)", "",
        "PlantVillage photographs were taken in controlled conditions: a single detached leaf on a uniform background, "
        "consistent lighting, no occlusion. The scores above therefore measure how well the model separates these 38 "
        "classes **under those conditions**. Published work (e.g. Mohanty et al. 2016) reports that accuracy of PlantVillage-"
        "trained models drops substantially on field photographs. Plant Doctor has **not** been evaluated on field images, "
        "and its outputs must be treated as screening suggestions, not diagnoses.", "",
        "Known limitations: only 14 crops / 38 conditions are covered; co-infections, early asymptomatic stages, nutrient "
        "disorders and pest damage outside the class list cannot be recognised; the novelty detector only catches images "
        "that are far from the training distribution in feature space.", "",
    ]
    (out / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"official_test": official, "strict_test": strict, "ood": ood_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
