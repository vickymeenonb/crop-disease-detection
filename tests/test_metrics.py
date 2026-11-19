import numpy as np

from crop_disease.metrics import compute_metrics


def test_perfect_predictions():
    y = np.array([0, 1, 2, 1])
    m = compute_metrics(y, y, ["a", "b", "c"])
    assert m["accuracy"] == 1.0
    assert m["macro"]["f1"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]


def test_confusion_matrix_shape_and_support():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = compute_metrics(y_true, y_pred, ["a", "b"])
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["per_class"]["a"]["support"] == 2
    assert m["per_class"]["b"]["recall"] == 1.0
