from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .baselines import naive_row_split_comparison
from .config import FIGURES, MODELS, PREFIXES, TABLES
from .data import fixed_split, load_and_audit
from .evaluation import paired_bootstrap_differences
from .lstm import run_lstm_experiments
from .models import run_prefix_experiments
from .plotting import save_figures
from .policy import sequential_policy


def run_experiment() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    x, y, raw_df, audit = load_and_audit()
    splits = fixed_split(y)
    split_manifest = {
        name: {
            "rows": int(len(indices)),
            "benign": int((y[indices] == 0).sum()),
            "malware": int((y[indices] == 1).sum()),
            "indices": indices.tolist(),
        }
        for name, indices in splits.items()
    }
    (TABLES / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
    (TABLES / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))

    results, predictions = run_prefix_experiments(x, y, splits)
    lstm_results, lstm_predictions, lstm_history = run_lstm_experiments(
        x,
        y,
        splits,
    )
    results = pd.concat([results, lstm_results], ignore_index=True)
    predictions.update(lstm_predictions)
    results.to_csv(TABLES / "prefix_metrics.csv", index=False)
    lstm_history.to_csv(TABLES / "lstm_training_history.csv", index=False)

    paired_comparisons = []
    for prefix in PREFIXES:
        lstm_item = predictions[f"LSTM|{prefix}"]
        bigram_item = predictions[f"Order-aware 1-2 gram|{prefix}"]
        paired_comparisons.append(
            {
                "prefix_calls": prefix,
                "candidate": "LSTM",
                "reference": "Order-aware 1-2 gram",
                **paired_bootstrap_differences(
                    y[splits["test"]],
                    lstm_item["test"],
                    lstm_item["threshold"],
                    bigram_item["test"],
                    bigram_item["threshold"],
                    seed=20241022 + prefix,
                ),
            }
        )
    pd.DataFrame(paired_comparisons).to_csv(
        TABLES / "lstm_vs_bigram_paired_bootstrap.csv",
        index=False,
    )

    policy, policy_summary = sequential_policy(
        y[splits["validation"]], y[splits["test"]], predictions
    )
    policy.to_csv(TABLES / "early_detection_policy.csv", index=False)
    (TABLES / "early_detection_summary.json").write_text(
        json.dumps(policy_summary, indent=2)
    )
    lstm_policy, lstm_policy_summary = sequential_policy(
        y[splits["validation"]],
        y[splits["test"]],
        predictions,
        model_name="LSTM",
    )
    lstm_policy.to_csv(TABLES / "early_detection_policy_lstm.csv", index=False)
    (TABLES / "early_detection_summary_lstm.json").write_text(
        json.dumps(lstm_policy_summary, indent=2)
    )

    naive = naive_row_split_comparison(raw_df)
    (TABLES / "naive_row_split_comparison.json").write_text(
        json.dumps(naive, indent=2)
    )
    np.savez_compressed(
        TABLES / "test_predictions.npz",
        y_test=y[splits["test"]],
        **{
            f"order_{prefix}": predictions[f"Order-aware 1-2 gram|{prefix}"]["test"]
            for prefix in PREFIXES
        },
        **{
            f"lstm_{prefix}": predictions[f"LSTM|{prefix}"]["test"]
            for prefix in PREFIXES
        },
    )

    save_figures(results, policy, predictions, y[splits["test"]], audit)
    print(
        json.dumps(
            {
                "audit": audit,
                "order_aware_policy": policy_summary,
                "lstm_policy": lstm_policy_summary,
                "naive": naive,
            },
            indent=2,
        )
    )
