"""Reproducible transfer-learning pipeline for Plant Doctor.

Example (full run on CPU, resumable):
    python -m training.train --data-root /path/to/PlantVillage --epochs 4

The script
* builds leaf-grouped train/val manifests (test is the official list),
* fine-tunes EfficientNet-B0 (ImageNet weights) with AdamW, label-smoothed
  cross-entropy, linear warm-up + cosine decay,
* evaluates every epoch on the validation split (macro F1 drives model
  selection and early stopping),
* writes the best checkpoint plus class mapping / configs / metadata to
  ``artifacts/`` and a JSON training log.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from models.model import build_model, save_checkpoint, write_class_mapping  # noqa: E402
from training.preprocessing import (  # noqa: E402
    ManifestDataset,
    build_manifests,
    get_eval_transforms,
    get_train_transforms,
    load_manifest,
    preprocessing_config,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:  # pragma: no cover - runs in workers
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Plant Doctor (EfficientNet-B0 transfer learning)")
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("PLANTVILLAGE_ROOT", "../datasets/plantvillage")))
    p.add_argument("--splits-dir", type=Path, default=None, help="folder with color_train.txt / color_test.txt")
    p.add_argument("--leaf-map", type=Path, default=None)
    p.add_argument("--manifest-dir", type=Path, default=config.ARTIFACTS_DIR / "splits")
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--patience", type=int, default=2, help="early-stopping patience (epochs)")
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--subset", type=int, default=None, help="use only N train/val images (smoke test)")
    p.add_argument("--max-train-minutes", type=float, default=None, help="stop after the epoch that crosses this budget")
    p.add_argument("--resume", action="store_true", help="resume from artifacts/last_checkpoint.pt")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item() * y.size(0))
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return {
        "loss": float(sum(losses) / len(targets)),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"[train] device={device} threads={torch.get_num_threads()}", flush=True)

    data_root = args.data_root.resolve()
    splits_dir = args.splits_dir or data_root / "raw" / "splits"
    leaf_map = args.leaf_map or data_root / "raw" / "leaf_grouping" / "leaf-map.json"
    manifest_dir = args.manifest_dir
    artifacts = args.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    if not (manifest_dir / "train.csv").exists():
        print("[train] building leaf-grouped manifests ...", flush=True)
        build_manifests(
            data_root=data_root,
            train_list=splits_dir / "color_train.txt",
            test_list=splits_dir / "color_test.txt",
            leaf_map_path=leaf_map,
            out_dir=manifest_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    class_names = json.loads((manifest_dir / "class_names.json").read_text())
    train_rows = load_manifest(manifest_dir / "train.csv")
    val_rows = load_manifest(manifest_dir / "val.csv")
    if args.subset:
        random.Random(args.seed).shuffle(train_rows)
        random.Random(args.seed + 1).shuffle(val_rows)
        train_rows, val_rows = train_rows[: args.subset], val_rows[: max(8, args.subset // 5)]
    print(f"[train] train={len(train_rows)} val={len(val_rows)} classes={len(class_names)}", flush=True)

    train_ds = ManifestDataset(train_rows, get_train_transforms())
    val_ds = ManifestDataset(val_rows, get_eval_transforms())
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        drop_last=True, worker_init_fn=seed_worker, generator=g, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(num_classes=len(class_names), pretrained=True).to(device)
    model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = min(args.warmup_steps, max(1, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Static artifacts written up-front so the app can validate them.
    write_class_mapping(class_names, artifacts / "class_mapping.json")
    (artifacts / "preprocessing_config.json").write_text(json.dumps(preprocessing_config(), indent=2))
    (artifacts / "model_config.json").write_text(
        json.dumps(
            {
                "architecture": config.ARCHITECTURE,
                "pretrained_weights": "torchvision EfficientNet_B0_Weights.IMAGENET1K_V1",
                "num_classes": len(class_names),
                "classifier": "Dropout(0.2) -> Linear(1280, num_classes)",
                "input_size": config.IMAGE_SIZE,
                "gradcam_target_layer": "features[-1] (last conv block, 7x7x1280)",
            },
            indent=2,
        )
    )

    start_epoch, best_f1, bad_epochs, history, global_step = 0, -1.0, 0, [], 0
    last_path = artifacts / "last_checkpoint.pt"
    best_path = config.MODEL_PATH if artifacts == config.ARTIFACTS_DIR else artifacts / config.MODEL_PATH.name
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_f1, bad_epochs, history, global_step = (
            state["epoch"] + 1, state["best_f1"], state["bad_epochs"], state["history"], state["global_step"])
        print(f"[train] resumed from epoch {state['epoch']} (best val macro-F1 {best_f1:.4f})", flush=True)

    t_start = time.time()
    run_info = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device),
        "torch": torch.__version__,
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "num_classes": len(class_names),
        "steps_per_epoch": len(train_loader),
    }

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_epoch = time.time()
        running, seen, correct = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, memory_format=torch.channels_last)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            running += loss.item() * y.size(0)
            seen += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            if (step + 1) % args.log_every == 0 or step + 1 == len(train_loader):
                el = time.time() - t_epoch
                eta = el / (step + 1) * (len(train_loader) - step - 1)
                print(
                    f"[epoch {epoch+1}/{args.epochs}] step {step+1}/{len(train_loader)} "
                    f"loss {running/seen:.4f} acc {correct/seen:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                    f"elapsed {el/60:.1f}m eta {eta/60:.1f}m",
                    flush=True,
                )
        train_metrics = {"loss": running / max(1, seen), "accuracy": correct / max(1, seen)}
        val_metrics = evaluate(model, val_loader, criterion, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_minutes": (time.time() - t_epoch) / 60,
            "lr_end": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(f"[epoch {epoch+1}] " + json.dumps(record), flush=True)

        improved = val_metrics["macro_f1"] > best_f1
        if improved:
            best_f1, bad_epochs = val_metrics["macro_f1"], 0
            save_checkpoint(
                model, best_path, class_names,
                extra={"epoch": epoch + 1, "val_metrics": val_metrics, "train_metrics": train_metrics,
                       "training_args": run_info["args"]},
            )
            print(f"[train] saved best checkpoint -> {best_path}", flush=True)
        else:
            bad_epochs += 1

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": epoch, "best_f1": best_f1, "bad_epochs": bad_epochs, "history": history,
             "global_step": global_step},
            last_path,
        )
        (artifacts / "training_log.json").write_text(json.dumps({**run_info, "history": history}, indent=2))

        if bad_epochs >= args.patience:
            print(f"[train] early stopping: no val improvement for {bad_epochs} epochs", flush=True)
            break
        if args.max_train_minutes and (time.time() - t_start) / 60 > args.max_train_minutes:
            print("[train] time budget reached; stopping", flush=True)
            break

    best = max(history, key=lambda r: r["val_macro_f1"]) if history else None
    metadata = {
        **run_info,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_minutes": (time.time() - t_start) / 60,
        "epochs_completed": len(history),
        "best_epoch": best["epoch"] if best else None,
        "best_val_macro_f1": best["val_macro_f1"] if best else None,
        "best_val_accuracy": best["val_accuracy"] if best else None,
        "checkpoint": str(best_path),
        "history": history,
        "dataset": {
            "name": "PlantVillage (color)",
            "source": "https://huggingface.co/datasets/mohanty/PlantVillage",
            "license": "CC BY-SA 3.0 (as declared on the dataset card)",
            "split_strategy": "official train/test lists; validation = 10% of training leaves per class, grouped by leaf_id",
        },
    }
    (artifacts / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("[train] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
