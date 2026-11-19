"""Classification metrics: precision, recall, F1, and confusion matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f1)},
        "weighted": {"precision": float(weighted_p), "recall": float(weighted_r),
                     "f1": float(weighted_f1)},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
