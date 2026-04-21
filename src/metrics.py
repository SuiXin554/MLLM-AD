from __future__ import annotations

from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class TaskMetrics:
    accuracy: float
    f1_macro: float


def cls_metrics(y_true: list[str], y_pred: list[str]) -> TaskMetrics:
    if not y_true:
        return TaskMetrics(accuracy=0.0, f1_macro=0.0)
    return TaskMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )
