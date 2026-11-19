"""Download PlantVillage parquet shards from Hugging Face, verify SHA-256, and export an ImageFolder tree.

Source: BrandonFors/Plant-Diseases-PlantVillage-Dataset (license: CC0-1.0).
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
REVISION = "main"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"

# Expected SHA-256 (LFS oid) and byte size for each shard, as reported by the Hugging Face API.
EXPECTED = {
    "data/test-00000-of-00001.parquet": (
        "8d7ac40386ff4692870eebf09f513a19abac8ba762b20d0449ab6f9da9dc7a55", 169812718),
    "data/train-00000-of-00002.parquet": (
        "502eebdae6b946855fb957f71ebb1714abd1dc5468f22457185323fafd19acc2", 320625654),
    "data/train-00001-of-00002.parquet": (
        "acba43432e43d20d5e9ea7d0d924a6d6a142c9122f32881d1ad1b6e00720f7c6", 361639059),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for rel, (digest, size) in EXPECTED.items():
        target = dest / Path(rel).name
        if target.exists() and target.stat().st_size == size and sha256_file(target) == digest:
            print(f"ok (cached): {target.name}")
            paths.append(target)
            continue
        print(f"downloading: {rel}")
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(BASE_URL + rel, timeout=60) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        actual = sha256_file(tmp)
        if tmp.stat().st_size != size or actual != digest:
            tmp.unlink()
            raise RuntimeError(
                f"Verification failed for {rel}: size={tmp.stat().st_size} sha256={actual}")
        tmp.rename(target)
        print(f"verified: {target.name}")
        paths.append(target)
    return paths


def export_imagefolder(parquet_paths: list[Path], out: Path) -> dict:
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
            print(f"exported {i} images")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw", help="ImageFolder output directory")
    parser.add_argument("--cache", default="data/downloads", help="Where parquet shards are stored")
    args = parser.parse_args()

    out = Path(args.out)
    shards = download_and_verify(Path(args.cache))
    counts = export_imagefolder(shards, out)

    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "license": "CC0-1.0",
        "shards": {p.name: sha256_file(p) for p in shards},
        "class_counts": counts,
        "total_images": sum(counts.values()),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {manifest['total_images']} images across {len(counts)} classes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
