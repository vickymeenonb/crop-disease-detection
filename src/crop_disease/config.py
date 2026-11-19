"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    # Must contain train/ and test/ subdirectories (see scripts/download_dataset.py).
    root: str = "data/raw"
    image_size: int = 224
    # Fraction of the official train split held out for validation. The official test split is fixed.
    val_split: float = 0.15
    num_workers: int = 4
    seed: int = 42


class ModelConfig(BaseModel):
    name: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False


class TrainingConfig(BaseModel):
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    device: str = "auto"
    output_dir: str = "artifacts"


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return AppConfig(**raw)
