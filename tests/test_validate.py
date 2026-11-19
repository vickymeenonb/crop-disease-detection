from PIL import Image

from crop_disease.validate import validate_dataset


def _img(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, format="PNG")


def test_clean_dataset_passes(tmp_path):
    _img(tmp_path / "a" / "1.png", (255, 0, 0))
    _img(tmp_path / "a" / "2.png", (250, 0, 0))
    _img(tmp_path / "b" / "1.png", (0, 255, 0))
    report = validate_dataset(tmp_path, min_per_class=1)
    assert report["ok"] is True
    assert report["class_counts"] == {"a": 2, "b": 1}
    assert report["class_balance"]["imbalance_ratio"] == 2.0


def test_corrupt_file_is_flagged(tmp_path):
    _img(tmp_path / "a" / "1.png", (255, 0, 0))
    _img(tmp_path / "b" / "1.png", (0, 255, 0))
    (tmp_path / "b" / "bad.jpg").write_bytes(b"not an image")
    report = validate_dataset(tmp_path, min_per_class=1)
    assert report["ok"] is False
    assert any("corrupt" in e for e in report["errors"])


def test_cross_class_duplicate_is_flagged(tmp_path):
    _img(tmp_path / "a" / "1.png", (10, 10, 10))
    _img(tmp_path / "b" / "1.png", (10, 10, 10))
    report = validate_dataset(tmp_path, min_per_class=1)
    assert report["ok"] is False
    assert len(report["cross_class_duplicates"]) == 1
