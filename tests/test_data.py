from pathlib import Path

from PIL import Image

from crop_disease.data import load_data


def _make_split(root: Path, split: str, classes: list[str], n: int) -> None:
    for c in classes:
        for i in range(n):
            p = root / split / c / f"{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(p, format="PNG")


def test_test_split_is_isolated(tmp_path):
    _make_split(tmp_path, "train", ["a", "b"], 10)
    _make_split(tmp_path, "test", ["a", "b"], 4)
    bundle = load_data(tmp_path, image_size=32, val_split=0.2, seed=0)

    train_val_paths = (
        {Path(bundle.train.dataset.samples[i][0]) for i in bundle.train.indices}
        | {Path(bundle.val.dataset.samples[i][0]) for i in bundle.val.indices}
    )
    test_paths = {Path(s[0]) for s in bundle.test.samples}

    assert not train_val_paths & test_paths
    assert all(p.is_relative_to(tmp_path / "test") for p in test_paths)
    assert len(test_paths) == 8
    assert bundle.class_names == ["a", "b"]


def test_missing_test_dir_fails(tmp_path):
    _make_split(tmp_path, "train", ["a", "b"], 4)
    try:
        load_data(tmp_path, image_size=32, val_split=0.2, seed=0)
    except FileNotFoundError as e:
        assert "test" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
