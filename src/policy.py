from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PREFIXES
from .evaluation import choose_threshold


def alert_times(
    probabilities: np.ndarray, thresholds: np.ndarray, consecutive: int
) -> np.ndarray:
    positive = probabilities >= thresholds
    confirmed = np.zeros_like(positive, dtype=bool)
    for checkpoint in range(consecutive - 1, positive.shape[1]):
        confirmed[:, checkpoint] = positive[
            :, checkpoint - consecutive + 1 : checkpoint + 1
        ].all(axis=1)
    first_detection = np.full(len(probabilities), np.nan)
    for row in range(len(probabilities)):
        locations = np.flatnonzero(confirmed[row])
        if len(locations):
            first_detection[row] = PREFIXES[int(locations[0])]
    return first_detection


def policy_metrics(y: np.ndarray, first_detection: np.ndarray) -> dict:
    malware_detected = np.isfinite(first_detection[y == 1])
    benign_alerted = np.isfinite(first_detection[y == 0])
    detected_calls = first_detection[(y == 1) & np.isfinite(first_detection)]
    return {
        "malware_coverage": float(malware_detected.mean()),
        "benign_false_alarm_rate": float(benign_alerted.mean()),
        "median_calls": float(np.median(detected_calls)) if len(detected_calls) else None,
    }


def sequential_policy(
    y_validation: np.ndarray,
    y_test: np.ndarray,
    predictions: dict,
    model_name: str = "Order-aware 1-2 gram",
) -> tuple[pd.DataFrame, dict]:
    validation_probabilities = np.column_stack(
        [predictions[f"{model_name}|{prefix}"]["validation"] for prefix in PREFIXES]
    )
    test_probabilities = np.column_stack(
        [predictions[f"{model_name}|{prefix}"]["test"] for prefix in PREFIXES]
    )

    candidates = []
    for per_checkpoint_fpr in [0.0, 0.01, 0.02, 0.05]:
        thresholds = np.array(
            [
                choose_threshold(
                    y_validation,
                    validation_probabilities[:, index],
                    max_fpr=per_checkpoint_fpr,
                )
                for index in range(len(PREFIXES))
            ]
        )
        for consecutive in [1, 2, 3]:
            validation_times = alert_times(
                validation_probabilities, thresholds, consecutive
            )
            validation_metrics = policy_metrics(y_validation, validation_times)
            if validation_metrics["benign_false_alarm_rate"] <= 0.05:
                candidates.append(
                    {
                        "per_checkpoint_fpr": per_checkpoint_fpr,
                        "consecutive": consecutive,
                        "thresholds": thresholds,
                        **validation_metrics,
                    }
                )
    if not candidates:
        raise RuntimeError("No sequential policy satisfied the validation false-alarm limit")
    selected = max(
        candidates,
        key=lambda item: (
            item["malware_coverage"],
            -item["benign_false_alarm_rate"],
            -(item["median_calls"] or 10_000),
        ),
    )
    first_detection = alert_times(
        test_probabilities, selected["thresholds"], selected["consecutive"]
    )

    rows = []
    for budget in PREFIXES:
        alerted = np.isfinite(first_detection) & (first_detection <= budget)
        malware_mask = y_test == 1
        benign_mask = y_test == 0
        rows.append(
            {
                "call_budget": budget,
                "malware_detection_coverage": alerted[malware_mask].mean(),
                "benign_false_alarm_rate": alerted[benign_mask].mean(),
                "malware_detected": int(alerted[malware_mask].sum()),
                "malware_total": int(malware_mask.sum()),
                "benign_false_alarms": int(alerted[benign_mask].sum()),
                "benign_total": int(benign_mask.sum()),
            }
        )
    detected_malware = first_detection[(y_test == 1) & np.isfinite(first_detection)]
    summary = {
        "policy": f"{selected['consecutive']} consecutive positive checkpoint(s)",
        "checkpoints": PREFIXES,
        "validation_selected_per_checkpoint_fpr_limit": selected[
            "per_checkpoint_fpr"
        ],
        "validation_malware_coverage": selected["malware_coverage"],
        "validation_benign_false_alarm_rate": selected[
            "benign_false_alarm_rate"
        ],
        "validation_median_calls_to_detection": selected["median_calls"],
        "thresholds": {
            str(prefix): float(threshold)
            for prefix, threshold in zip(PREFIXES, selected["thresholds"])
        },
        "malware_detection_coverage_by_100": float(
            np.isfinite(first_detection[y_test == 1]).mean()
        ),
        "benign_false_alarm_rate_by_100": float(
            np.isfinite(first_detection[y_test == 0]).mean()
        ),
        "median_calls_to_detection_among_detected_malware": float(
            np.median(detected_malware)
        ),
    }
    return pd.DataFrame(rows), summary
