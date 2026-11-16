"""Dataset manifests, leaf-grouped splitting and image transforms.

Why leaf-grouped splitting matters
----------------------------------
PlantVillage contains several photographs of the same physical leaf. If those
end up on both sides of a train/validation split, validation accuracy becomes
optimistic. The dataset ships a ``leaf-map.json`` that identifies the leaf
behind each file name; we reproduce the official loader's lookup logic and
split *leaves*, never individual images.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config

MANIFEST_FIELDS = ["path", "label", "class_name", "leaf_id", "overlaps_train"]


# --------------------------------------------------------------------------- #
# Transforms (the eval transform is what the application uses at inference)
# --------------------------------------------------------------------------- #
def get_train_transforms(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Deterministic preprocessing shared by evaluation and the live app."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def preprocessing_config() -> dict:
    return {
        "input_size": [config.IMAGE_SIZE, config.IMAGE_SIZE],
        "color_space": "RGB",
        "resize": "direct resize to input_size (bilinear)",
        "normalize_mean": config.IMAGENET_MEAN,
        "normalize_std": config.IMAGENET_STD,
        "train_augmentation": [
            "RandomResizedCrop(scale 0.65-1.0, ratio 0.85-1.18)",
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "RandomRotation(20 deg)",
            "ColorJitter(brightness 0.25, contrast 0.25, saturation 0.2, hue 0.02)",
        ],
    }


# --------------------------------------------------------------------------- #
# Leaf identity (mirrors the official Hugging Face loader logic)
# --------------------------------------------------------------------------- #
def derive_leaf_id(file_name: str, class_name: str, leaf_map: dict[str, list[str]]) -> str:
    identifier = file_name.replace("_final_masked", "")
    if "___" in identifier:
        identifier = identifier.split("___")[-1]
    identifier = identifier.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"):
        identifier = identifier.replace(ext, "")
    identifier = identifier.strip()
    key = identifier.lower().strip()
    suggestions = leaf_map.get(key)
    if suggestions:
        if len(suggestions) == 1:
            return suggestions[0]
        for s in suggestions:
            if class_name in s:
                return s
    return f"fallback_{identifier}"


def read_split_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_manifests(
    data_root: Path,
    train_list: Path,
    test_list: Path,
    leaf_map_path: Path,
    out_dir: Path,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> dict[str, Path]:
    """Create train/val/test CSV manifests.

    * ``test`` is exactly the official PlantVillage test list (never touched
      during training or tuning).
    * ``val`` is carved out of the official train list by holding out ~10 % of
      the *leaves* of every class.
    """
    leaf_map = json.loads(leaf_map_path.read_text(encoding="utf-8"))
    class_names = sorted({p.split("/")[2] for p in read_split_list(train_list)})
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    def records(paths: list[str]) -> list[dict]:
        rows = []
        for rel in paths:
            parts = rel.split("/")
            class_name, file_name = parts[2], parts[3]
            full = data_root / rel
            if not full.exists():
                raise FileNotFoundError(full)
            rows.append(
                {
                    "path": str(full),
                    "label": class_to_idx[class_name],
                    "class_name": class_name,
                    "leaf_id": derive_leaf_id(file_name, class_name, leaf_map),
                    "overlaps_train": 0,
                }
            )
        return rows

    train_rows = records(read_split_list(train_list))
    test_rows = records(read_split_list(test_list))

    # Leaf-grouped, class-stratified hold-out.
    rng = random.Random(seed)
    by_class_leaf: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in train_rows:
        by_class_leaf[r["class_name"]][r["leaf_id"]].append(r)
    train_final, val_final = [], []
    for class_name in class_names:
        leaves = sorted(by_class_leaf[class_name].keys())
        rng.shuffle(leaves)
        n_val = max(1, round(len(leaves) * val_fraction))
        val_leaves = set(leaves[:n_val])
        for leaf, rows in by_class_leaf[class_name].items():
            (val_final if leaf in val_leaves else train_final).extend(rows)

    # Sanity: no leaf shared between any two splits.
    def leaves(rows: list[dict]) -> set[str]:
        return {r["leaf_id"] for r in rows}

    assert not (leaves(train_final) & leaves(val_final)), "leaf leakage between train and val"
    # The *official* test list shares a small number of unmapped ("fallback")
    # leaf identifiers with the official train list. We keep the official list
    # untouched for benchmark comparability but flag those rows so that the
    # evaluation can additionally report a strict, overlap-free subset.
    overlapping = leaves(train_final) | leaves(val_final)
    n_overlap = 0
    for r in test_rows:
        if r["leaf_id"] in overlapping:
            r["overlaps_train"] = 1
            n_overlap += 1
    print(f"[manifests] test rows whose leaf id also appears in train/val: {n_overlap} of {len(test_rows)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, rows in (("train", train_final), ("val", val_final), ("test", test_rows)):
        p = out_dir / f"{name}.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(rows)
        outputs[name] = p
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    return outputs


def load_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict], transform, limit: int | None = None):
        self.rows = rows[:limit] if limit else rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(row["path"]) as im:
            image = im.convert("RGB")
        return self.transform(image), row["label"]


def class_counts(rows: list[dict], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for r in rows:
        counts[r["label"]] += 1
    return counts
