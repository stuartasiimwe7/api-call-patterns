from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import N_BOOTSTRAP, SEED


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, max_fpr: float = 0.05) -> float:
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    best: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if fp + tn else 0.0
        if fpr > max_fpr:
            continue
        recall = tp / (tp + fn) if tp + fn else 0.0
        macro_f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
        candidate = (recall, macro_f1, threshold)
        if best is None or candidate > best:
            best = candidate
            best_threshold = float(threshold)
    return best_threshold


def metric_record(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=[0, 1], zero_division=0
    )
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, average="macro", zero_division=0),
        "benign_precision": precision[0],
        "benign_recall_specificity": recall[0],
        "benign_f1": f1[0],
        "malware_precision": precision[1],
        "malware_recall": recall[1],
        "malware_f1": f1[1],
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "roc_auc": roc_auc_score(y_true, probability),
        "malware_pr_auc": average_precision_score(y_true, probability),
        "benign_pr_auc": average_precision_score(1 - y_true, 1 - probability),
        "brier_score": brier_score_loss(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "test_benign": int(support[0]),
        "test_malware": int(support[1]),
    }


def bootstrap_intervals(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    benign = np.flatnonzero(y_true == 0)
    malware = np.flatnonzero(y_true == 1)
    values = defaultdict(list)
    for _ in range(N_BOOTSTRAP):
        indices = np.r_[
            rng.choice(benign, size=len(benign), replace=True),
            rng.choice(malware, size=len(malware), replace=True),
        ]
        y_sample = y_true[indices]
        probability_sample = probability[indices]
        prediction = (probability_sample >= threshold).astype(np.int8)
        values["balanced_accuracy"].append(
            balanced_accuracy_score(y_sample, prediction)
        )
        values["macro_f1"].append(
            f1_score(y_sample, prediction, average="macro", zero_division=0)
        )
        values["roc_auc"].append(roc_auc_score(y_sample, probability_sample))
    intervals = {}
    for name, samples in values.items():
        low, high = np.quantile(samples, [0.025, 0.975])
        intervals[f"{name}_ci_low"] = float(low)
        intervals[f"{name}_ci_high"] = float(high)
    return intervals
