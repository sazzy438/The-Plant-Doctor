"""Checks on the data pipeline: leaf-grouped splits, transforms, manifest integrity."""
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import config
from training.preprocessing import (derive_leaf_id, get_eval_transforms, get_train_transforms, load_manifest,
                                    preprocessing_config)

SPLITS = config.ARTIFACTS_DIR / "splits"


def test_eval_transform_matches_config(diseased_leaf):
    x = get_eval_transforms()(diseased_leaf)
    assert x.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    # Normalisation uses the ImageNet statistics recorded in the preprocessing config
    cfg = preprocessing_config()
    assert cfg["normalize_mean"] == config.IMAGENET_MEAN and cfg["normalize_std"] == config.IMAGENET_STD
    white = Image.new("RGB", (300, 200), (255, 255, 255))
    y = get_eval_transforms()(white)
    for c in range(3):
        assert abs(y[c].mean().item() - (1 - config.IMAGENET_MEAN[c]) / config.IMAGENET_STD[c]) < 1e-4


def test_train_transform_is_stochastic(diseased_leaf):
    t = get_train_transforms()
    torch.manual_seed(1); a = t(diseased_leaf)
    torch.manual_seed(2); b = t(diseased_leaf)
    assert a.shape == b.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE) and not torch.equal(a, b)


def test_derive_leaf_id_mirrors_dataset_loader():
    leaf_map = {"rs_erly.b 8389": ["Tomato___Early_blight:::12.0"], "x 1": ["A___b:::1.0", "C___d:::2.0"]}
    assert derive_leaf_id("0012b9d2___RS_Erly.B 8389.JPG", "Tomato___Early_blight", leaf_map) == "Tomato___Early_blight:::12.0"
    assert derive_leaf_id("abc___X 1 copy 2.jpg", "C___d", leaf_map) == "C___d:::2.0"
    assert derive_leaf_id("abc___UNKNOWN 7.JPG", "C___d", leaf_map) == "fallback_UNKNOWN 7"


@pytest.mark.skipif(not (SPLITS / "train.csv").exists(), reason="manifests not built")
def test_manifests_have_no_leaf_leakage_between_train_and_val():
    train, val, test = (load_manifest(SPLITS / f"{n}.csv") for n in ("train", "val", "test"))
    names = json.loads((SPLITS / "class_names.json").read_text())
    assert len(names) == config.NUM_CLASSES
    assert {r["leaf_id"] for r in train}.isdisjoint({r["leaf_id"] for r in val})
    assert {r["path"] for r in train}.isdisjoint({r["path"] for r in test})
    assert {r["path"] for r in val}.isdisjoint({r["path"] for r in test})
    assert len(train) + len(val) == 43596 and len(test) == 10709  # official PlantVillage counts
    assert {r["label"] for r in val} == set(range(config.NUM_CLASSES))  # every class represented in validation
    flagged = sum(int(r["overlaps_train"]) for r in test)
    assert 0 < flagged < len(test) * 0.1
