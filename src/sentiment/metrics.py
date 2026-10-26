"""Binary classification metrics, with positive sentiment (label 1) as the positive class."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def classification_metrics(y_true, y_pred) -> Metrics:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape")
    tp = np.sum(y_true & y_pred)
    precision = tp / y_pred.sum() if y_pred.any() else 0.0
    recall = tp / y_true.sum() if y_true.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(
        accuracy=float(np.mean(y_true == y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )
