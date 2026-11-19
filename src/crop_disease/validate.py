"""Validate an ImageFolder dataset: decodability, labels, duplicates, and class balance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_dataset(root: str | Path, min_per_class: int = 50,
                     max_imbalance_ratio: float = 10.0) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    errors: list[str] = []
    warnings: list[str] = []
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    stray_files = sorted(p.name for p in root.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    if len(class_dirs) < 2:
        errors.append(f"Expected at least 2 class directories, found {len(class_dirs)}.")
    if stray_files:
        warnings.append(f"Files at dataset root (ignored by ImageFolder): {stray_files[:10]}")

    counts: dict[str, int] = {}
    corrupt: list[str] = []
    non_image: list[str] = []
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    hash_to_classes: dict[str, set[str]] = defaultdict(set)

    for cdir in class_dirs:
        n = 0
        for f in sorted(cdir.iterdir()):
            if not f.is_file():
                continue
            rel = str(f.relative_to(root))
            if f.suffix.lower() not in IMAGE_SUFFIXES:
                non_image.append(rel)
                continue
            data = f.read_bytes()
            try:
                with Image.open(f) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError, SyntaxError):
                corrupt.append(rel)
                continue
            n += 1
            digest = hashlib.sha256(data).hexdigest()
            hash_to_paths[digest].append(rel)
            hash_to_classes[digest].add(cdir.name)
        counts[cdir.name] = n

    if non_image:
        warnings.append(f"{len(non_image)} non-image files skipped, e.g. {non_image[:5]}")
    if corrupt:
        errors.append(f"{len(corrupt)} corrupt or undecodable images, e.g. {corrupt[:5]}")

    empty = [c for c, n in counts.items() if n == 0]
    if empty:
        errors.append(f"Classes with no valid images: {empty}")

    cross_class = [
        {"sha256": h, "classes": sorted(hash_to_classes[h]), "paths": hash_to_paths[h]}
        for h in hash_to_paths
        if len(hash_to_classes[h]) > 1
    ]
    if cross_class:
        errors.append(f"{len(cross_class)} identical images appear under different labels (label noise).")
    within_dupes = sum(len(p) - 1 for p in hash_to_paths.values() if len(p) > 1)
    if within_dupes:
        warnings.append(f"{within_dupes} exact duplicate images within a class (can leak across splits).")

    nonzero = {c: n for c, n in counts.items() if n > 0}
    balance: dict[str, Any] = {}
    if nonzero:
        largest = max(nonzero, key=nonzero.get)
        smallest = min(nonzero, key=nonzero.get)
        ratio = nonzero[largest] / nonzero[smallest]
        balance = {
            "largest_class": largest, "largest_count": nonzero[largest],
            "smallest_class": smallest, "smallest_count": nonzero[smallest],
            "imbalance_ratio": round(ratio, 2),
        }
        if ratio > max_imbalance_ratio:
            warnings.append(f"Class imbalance ratio {ratio:.1f} exceeds {max_imbalance_ratio}. "
                            "Consider weighted loss or resampling.")
    small = [c for c, n in nonzero.items() if n < min_per_class]
    if small:
        warnings.append(f"Classes below {min_per_class} images: {small}")

    return {
        "root": str(root),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "num_classes": len(class_dirs),
        "total_valid_images": sum(counts.values()),
        "class_counts": counts,
        "class_balance": balance,
        "corrupt_files": corrupt,
        "cross_class_duplicates": cross_class,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an ImageFolder dataset.")
    parser.add_argument("root", nargs="?", default="data/raw")
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--max-imbalance", type=float, default=10.0)
    parser.add_argument("--report", default="artifacts/dataset_report.json")
    args = parser.parse_args(argv)

    report = validate_dataset(args.root, args.min_per_class, args.max_imbalance)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"classes={report['num_classes']} images={report['total_valid_images']} ok={report['ok']}")
    for e in report["errors"]:
        print(f"ERROR: {e}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    print(f"report written to {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
