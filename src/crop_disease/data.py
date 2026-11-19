"""Dataset loading and transforms.

The official dataset split is preserved on disk:
  <root>/train/<class>/...  -> split into train and validation (stratified, seeded)
  <root>/test/<class>/...   -> held-out test set, loaded unchanged and never used for training or tuning
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataBundle:
    train: Dataset
    val: Dataset
    test: Dataset
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


def stratified_split(labels: np.ndarray, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train and validation with class proportions preserved."""
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_split, stratify=labels, random_state=seed
    )
    return train_idx, val_idx


def load_data(root: str | Path, image_size: int, val_split: float, seed: int) -> DataBundle:
    root = Path(root)
    train_dir = root / "train"
    test_dir = root / "test"
    for d in (train_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Expected split directory not found: {d}")

    train_eval = datasets.ImageFolder(train_dir, transform=build_transforms(image_size, train=False))
    train_aug = datasets.ImageFolder(train_dir, transform=build_transforms(image_size, train=True))
    # The test set uses the eval transform and is never passed to a stratified split.
    test_ds = datasets.ImageFolder(test_dir, transform=build_transforms(image_size, train=False))

    if train_eval.classes != test_ds.classes:
        raise ValueError(
            f"Class lists differ between {train_dir} and {test_dir}: "
            f"{train_eval.classes} vs {test_ds.classes}")

    labels = np.array(train_eval.targets)
    tr, va = stratified_split(labels, val_split, seed)
    return DataBundle(
        train=Subset(train_aug, tr.tolist()),
        val=Subset(train_eval, va.tolist()),
        test=test_ds,
        class_names=train_eval.classes,
    )
