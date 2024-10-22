from __future__ import annotations

import json

import numpy as np

from .baselines import naive_row_split_comparison
from .config import FIGURES, MODELS, PREFIXES, TABLES
from .data import fixed_split, load_and_audit
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
    results.to_csv(TABLES / "prefix_metrics.csv", index=False)

    policy, policy_summary = sequential_policy(
        y[splits["validation"]], y[splits["test"]], predictions
    )
    policy.to_csv(TABLES / "early_detection_policy.csv", index=False)
    (TABLES / "early_detection_summary.json").write_text(
        json.dumps(policy_summary, indent=2)
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
    )

    save_figures(results, policy, predictions, y[splits["test"]], audit)
    print(json.dumps({"audit": audit, "policy": policy_summary, "naive": naive}, indent=2))
