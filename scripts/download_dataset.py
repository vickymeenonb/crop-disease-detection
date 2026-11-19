"""Download PlantVillage parquet shards from Hugging Face, verify SHA-256, and export ImageFolder trees.

Source: BrandonFors/Plant-Diseases-PlantVillage-Dataset (license: CC0-1.0), pinned to an immutable commit.
The official train and test splits are preserved and written to separate directories:
  <out>/train/<class>/...
  <out>/test/<class>/...
Usage: python scripts/download_dataset.py --out data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
# Immutable dataset commit (Hugging Face repo sha as reported by the API). Do not change to a branch name.
REVISION = "96bdc27bd804297b76f373cba1c935e52076e107"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
SPLITS = ("train", "test")

# Expected SHA-256 (LFS oid) and byte size for each shard, as reported by the Hugging Face API.
EXPECTED = {
    "data/test-00000-of-00001.parquet": (
        "8d7ac40386ff4692870eebf09f513a19abac8ba762b20d0449ab6f9da9dc7a55", 169812718),
    "data/train-00000-of-00002.parquet": (
        "502eebdae6b946855fb957f71ebb1714abd1dc5468f22457185323fafd19acc2", 320625654),
    "data/train-00001-of-00002.parquet": (
        "acba43432e43d20d5e9ea7d0d924a6d6a142c9122f32881d1ad1b6e00720f7c6", 361639059),
}


def split_of(rel: str) -> str:
    """Map a shard path such as 'data/test-00000-of-00001.parquet' to its split name."""
    split = Path(rel).name.split("-")[0]
    if split not in SPLITS:
        raise ValueError(f"Unrecognized split for shard: {rel}")
    return split


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify(dest: Path) -> dict[str, list[Path]]:
    """Download any missing shards, verify each against EXPECTED, and group paths by split."""
    dest.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[Path]] = {s: [] for s in SPLITS}
    for rel, (digest, size) in EXPECTED.items():
        target = dest / Path(rel).name
        if target.exists() and target.stat().st_size == size and sha256_file(target) == digest:
            print(f"ok (cached): {target.name}")
            by_split[split_of(rel)].append(target)
            continue
        print(f"downloading: {rel}")
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(BASE_URL + rel, timeout=60) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        # Record the size and hash while the temp file still exists, so the error message can report them.
        actual_size = tmp.stat().st_size
        actual_digest = sha256_file(tmp)
        if actual_size != size or actual_digest != digest:
            tmp.unlink()
            raise RuntimeError(
                f"Verification failed for {rel}: "
                f"size={actual_size} (expected {size}), "
                f"sha256={actual_digest} (expected {digest})")
        tmp.rename(target)
        print(f"verified: {target.name}")
        by_split[split_of(rel)].append(target)
    return by_split


def check_output_dir(out: Path) -> None:
    """Refuse to mix the split layout with a leftover flat ImageFolder layout from older runs."""
    if not out.exists():
        return
    stale = sorted(p.name for p in out.iterdir() if p.name not in set(SPLITS) | {"MANIFEST.json"})
    if stale:
        raise RuntimeError(
            f"{out} contains entries from an older flat layout: {stale[:10]}. "
            "Delete them (or choose a different --out) and rerun.")


def export_split(parquet_paths: list[Path], out: Path) -> dict[str, int]:
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=[str(p) for p in parquet_paths], split="train")
    if "label" not in ds.features or not hasattr(ds.features["label"], "names"):
        raise RuntimeError("Expected a ClassLabel 'label' column with class names.")
    names = ds.features["label"].names
    # Keep raw encoded bytes so images are written without re-encoding.
    ds = ds.cast_column("image", HFImage(decode=False))

    counts: dict[str, int] = {n: 0 for n in names}
    out.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(ds):
        cls = names[row["label"]]
        img = row["image"]
        suffix = Path(img.get("path") or "").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".jpg"
        cdir = out / cls
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / f"{i:06d}{suffix}").write_bytes(img["bytes"])
        counts[cls] += 1
        if i % 5000 == 0:
            print(f"exported {i} images to {out.name}/")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw", help="Root output directory; writes train/ and test/ below it")
    parser.add_argument("--cache", default="data/downloads", help="Where parquet shards are stored")
    args = parser.parse_args()

    out = Path(args.out)
    check_output_dir(out)
    by_split = download_and_verify(Path(args.cache))

    split_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_counts[split] = export_split(by_split[split], out / split)

    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "license": "CC0-1.0",
        "shards": {p.name: sha256_file(p) for paths in by_split.values() for p in paths},
        "splits": {
            split: {
                "class_counts": counts,
                "total_images": sum(counts.values()),
            }
            for split, counts in split_counts.items()
        },
        "note": "The test split is the official held-out split and must not be used for training, "
                "validation, or tuning.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for split in SPLITS:
        print(f"done: {split} has {manifest['splits'][split]['total_images']} images "
              f"across {len(split_counts[split])} classes -> {out / split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
