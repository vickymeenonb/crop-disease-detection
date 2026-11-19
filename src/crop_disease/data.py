"""Dataset loading, transforms, and stratified splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataBundle:
    train: Subset
    val: Subset
    test: Subset
    class_names: list[str]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        steps = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        steps = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def stratified_indices(
    labels: np.ndarray, val_split: float, test_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, val, and test indices with class proportions preserved."""
    idx = np.arange(len(labels))
    holdout = val_split + test_split
    train_idx, hold_idx = train_test_split(
        idx, test_size=holdout, stratify=labels, random_state=seed
    )
    rel_test = test_split / holdout
    val_idx, test_idx = train_test_split(
        hold_idx, test_size=rel_test, stratify=labels[hold_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


def load_data(root: str | Path, image_size: int, val_split: float,
              test_split: float, seed: int) -> DataBundle:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    base = datasets.ImageFolder(root, transform=build_transforms(image_size, train=False))
    train_ds_aug = datasets.ImageFolder(root, transform=build_transforms(image_size, train=True))
    labels = np.array(base.targets)

    tr, va, te = stratified_indices(labels, val_split, test_split, seed)
    return DataBundle(
        train=Subset(train_ds_aug, tr.tolist()),
        val=Subset(base, va.tolist()),
        test=Subset(base, te.tolist()),
        class_names=base.classes,
    )
