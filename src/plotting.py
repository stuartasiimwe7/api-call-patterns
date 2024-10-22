from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import FIGURES

MODEL_COLOURS = {
    "Frequency": "#d62728",
    "LSTM": "#2ca02c",
    "Order-aware 1-2 gram": "#1f77b4",
}
CURVE_COLOURS = {
    10: "#d62728",
    20: "#2ca02c",
    40: "#1f77b4",
    100: "#000000",
}


def save_prefix_performance(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for model_name, group in results.groupby("model"):
        axes[0].errorbar(
            group["prefix_calls"],
            group["balanced_accuracy"],
            yerr=np.vstack(
                [
                    group["balanced_accuracy"]
                    - group["balanced_accuracy_ci_low"],
                    group["balanced_accuracy_ci_high"]
                    - group["balanced_accuracy"],
                ]
            ),
            marker="o",
            capsize=3,
            label=model_name,
            color=MODEL_COLOURS[model_name],
        )
        axes[1].errorbar(
            group["prefix_calls"],
            group["macro_f1"],
            yerr=np.vstack(
                [
                    group["macro_f1"] - group["macro_f1_ci_low"],
                    group["macro_f1_ci_high"] - group["macro_f1"],
                ]
            ),
            marker="o",
            capsize=3,
            label=model_name,
            color=MODEL_COLOURS[model_name],
        )
    axes[0].set(
        title="Balanced accuracy by observed API calls",
        xlabel="Observed API calls",
        ylabel="Balanced accuracy",
        ylim=(0.0, 1.01),
    )
    axes[1].set(
        title="Macro F1 by observed API calls",
        xlabel="Observed API calls",
        ylabel="Macro F1",
        ylim=(0.0, 1.01),
    )
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.savefig(FIGURES / "prefix_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "prefix_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def save_roc_pr_curves(predictions: dict, y_test: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for prefix in [10, 20, 40, 100]:
        probability = predictions[f"Order-aware 1-2 gram|{prefix}"]["test"]
        colour = CURVE_COLOURS[prefix]
        fpr, tpr, _ = roc_curve(y_test, probability)
        auc = roc_auc_score(y_test, probability)
        axes[0].plot(
            fpr,
            tpr,
            color=colour,
            label=f"{prefix} calls (AUC {auc:.3f})",
        )
        precision, recall, _ = precision_recall_curve(1 - y_test, 1 - probability)
        average_precision = average_precision_score(1 - y_test, 1 - probability)
        axes[1].plot(
            recall,
            precision,
            color=colour,
            label=f"{prefix} calls (AP {average_precision:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
    axes[0].set(
        title="ROC curves",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axes[1].set(
        title="Benign-class precision-recall",
        xlabel="Benign recall",
        ylabel="Benign precision",
    )
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "roc_pr_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "roc_pr_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def save_figures(
    results: pd.DataFrame,
    policy: pd.DataFrame,
    predictions: dict,
    y_test: np.ndarray,
    audit: dict,
):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    save_prefix_performance(results)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.plot(
        policy["call_budget"],
        policy["malware_detection_coverage"],
        marker="o",
        label="Malware detected",
        color="#0f8ebd",
    )
    ax.plot(
        policy["call_budget"],
        policy["benign_false_alarm_rate"],
        marker="o",
        label="Benign false alarms",
        color="#dc2626",
    )
    ax.set(
        title="Sequential early-detection policy",
        xlabel="API-call budget",
        ylabel="Cumulative proportion",
        ylim=(-0.02, 1.02),
    )
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "early_detection_policy.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "early_detection_policy.pdf", bbox_inches="tight")
    plt.close(fig)

    save_roc_pr_curves(predictions, y_test)

    order_aware_results = results.query("model == 'Order-aware 1-2 gram'")
    reaches_target = (order_aware_results["balanced_accuracy"] >= 0.85).any()
    chosen_prefix = int(
        results.loc[results["model"] == "Order-aware 1-2 gram"]
        .sort_values("prefix_calls")
        .query("balanced_accuracy >= 0.85")
        ["prefix_calls"]
        .min()
    ) if reaches_target else 100
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)
    for ax, prefix in zip(axes, [chosen_prefix, 100]):
        item = predictions[f"Order-aware 1-2 gram|{prefix}"]
        prediction = (item["test"] >= item["threshold"]).astype(np.int8)
        matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(
            title=f"{prefix} observed calls",
            xlabel="Predicted class",
            ylabel="True class",
            xticklabels=["Benign", "Malware"],
            yticklabels=["Benign", "Malware"],
        )
    fig.savefig(FIGURES / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    categories = ["Raw benign", "Raw malware", "Unique benign", "Unique malware"]
    values = [
        audit["raw_benign"],
        audit["raw_malware"],
        audit["clean_benign"],
        audit["clean_malware"],
    ]
    colors = ["#94a3b8", "#0f8ebd", "#64748b", "#08749d"]
    ax.bar(categories, values, color=colors)
    ax.set(title="Dataset composition before and after sequence deduplication", ylabel="Samples")
    ax.tick_params(axis="x", rotation=15)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    fig.savefig(FIGURES / "dataset_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "dataset_audit.pdf", bbox_inches="tight")
    plt.close(fig)
