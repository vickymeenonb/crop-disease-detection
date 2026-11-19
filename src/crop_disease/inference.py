"""Checkpoint loading and single-image prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .model import build_model


@dataclass
class Predictor:
    model: torch.nn.Module
    class_names: list[str]
    image_size: int
    device: torch.device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "Predictor":
        dev = torch.device(device)
        ckpt = torch.load(path, map_location=dev)
        model = build_model(ckpt["model_name"], len(ckpt["class_names"]), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(dev).eval()
        return cls(model, ckpt["class_names"], ckpt["image_size"], dev)

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        tfm = build_transforms(self.image_size, train=False)
        x = tfm(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        k = min(top_k, probs.numel())
        values, indices = torch.topk(probs, k)
        return [
            {"class_name": self.class_names[i], "probability": float(v)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]
