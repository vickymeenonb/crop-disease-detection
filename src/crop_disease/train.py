"""Training loop with validation, early stopping, and metric export."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import AppConfig, load_config
from .data import load_data
from .metrics import compute_metrics
from .model import build_model


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, n = 0.0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    return total_loss / max(n, 1), np.concatenate(targets), np.concatenate(preds)


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def run(cfg: AppConfig) -> dict:
    torch.manual_seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)
    device = resolve_device(cfg.training.device)
    out = Path(cfg.training.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_data(cfg.data.root, cfg.data.image_size, cfg.data.val_split,
                     cfg.data.test_split, cfg.data.seed)
    nw = cfg.data.num_workers
    train_loader = DataLoader(data.train, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=nw)
    val_loader = DataLoader(data.val, batch_size=cfg.training.batch_size, num_workers=nw)
    test_loader = DataLoader(data.test, batch_size=cfg.training.batch_size, num_workers=nw)

    model = build_model(cfg.model.name, len(data.class_names), cfg.model.pretrained,
                        cfg.model.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay,
    )

    best_f1, best_epoch, bad_epochs = -1.0, -1, 0
    ckpt_path = out / "best.pt"
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, y_true, y_pred = evaluate(model, val_loader, device, criterion)
        val_f1 = compute_metrics(y_true, y_pred, data.class_names)["macro"]["f1"]
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": data.class_names,
                "model_name": cfg.model.name,
                "image_size": cfg.data.image_size,
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.training.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation on the held-out test set using the best checkpoint.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, y_true, y_pred = evaluate(model, test_loader, device, criterion)
    metrics = compute_metrics(y_true, y_pred, data.class_names)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_macro_f1"] = best_f1

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + data.class_names)
        for name, row in zip(data.class_names, metrics["confusion_matrix"]):
            writer.writerow([name] + row)
    print(f"Test accuracy={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro']['f1']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the crop disease classifier.")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
