from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/dynamic_api_call_sequence_per_malware_100_0_306.csv"
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
MODELS = ROOT / "results/models"
SEED = 20241022
PREFIXES = [5, 10, 20, 40, 60, 80, 100]
CS = [0.1, 1.0, 10.0]
N_BOOTSTRAP = 1000


def sequence_fingerprint(row: np.ndarray) -> str:
    return hashlib.sha256(row.astype(np.int16).tobytes()).hexdigest()


def load_and_audit() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    df = pd.read_csv(DATA)
    sequence_cols = [f"t_{i}" for i in range(100)]
    x = df[sequence_cols].to_numpy(dtype=np.int16)
    y = df["malware"].to_numpy(dtype=np.int8)
    fingerprints = np.array([sequence_fingerprint(row) for row in x])

    groups: dict[str, list[int]] = defaultdict(list)
    for index, fingerprint in enumerate(fingerprints):
        groups[fingerprint].append(index)

    clean_indices: list[int] = []
    conflicting_groups: list[dict] = []
    duplicate_groups = 0
    duplicate_extra_rows = 0
    for fingerprint, indices in groups.items():
        labels = sorted(set(int(y[index]) for index in indices))
        if len(labels) > 1:
            conflicting_groups.append(
                {
                    "fingerprint": fingerprint,
                    "rows": len(indices),
                    "benign_rows": sum(y[index] == 0 for index in indices),
                    "malware_rows": sum(y[index] == 1 for index in indices),
                }
            )
            continue
        clean_indices.append(indices[0])
        if len(indices) > 1:
            duplicate_groups += 1
            duplicate_extra_rows += len(indices) - 1

    clean_x = x[clean_indices]
    clean_y = y[clean_indices]
    audit = {
        "raw_rows": int(len(df)),
        "raw_benign": int((y == 0).sum()),
        "raw_malware": int((y == 1).sum()),
        "unique_sequence_groups": int(len(groups)),
        "duplicate_sequence_groups": int(sum(len(v) > 1 for v in groups.values())),
        "duplicate_extra_rows": int(sum(len(v) - 1 for v in groups.values())),
        "conflicting_sequence_groups": int(len(conflicting_groups)),
        "conflicting_rows": int(sum(item["rows"] for item in conflicting_groups)),
        "clean_unique_rows": int(len(clean_x)),
        "clean_benign": int((clean_y == 0).sum()),
        "clean_malware": int((clean_y == 1).sum()),
    }
    pd.DataFrame(conflicting_groups).to_csv(TABLES / "conflicting_sequences.csv", index=False)
    return clean_x, clean_y, df, audit


def fixed_split(y: np.ndarray) -> dict[str, np.ndarray]:
    all_indices = np.arange(len(y))
    train_indices, remainder = train_test_split(
        all_indices, test_size=0.30, random_state=SEED, stratify=y
    )
    validation_indices, test_indices = train_test_split(
        remainder,
        test_size=0.50,
        random_state=SEED,
        stratify=y[remainder],
    )
    return {
        "train": np.sort(train_indices),
        "validation": np.sort(validation_indices),
        "test": np.sort(test_indices),
    }


def frequency_features(x: np.ndarray, prefix: int, vocabulary_size: int = 307) -> csr_matrix:
    rows = np.repeat(np.arange(len(x)), prefix)
    cols = x[:, :prefix].reshape(-1)
    values = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix((values, (rows, cols)), shape=(len(x), vocabulary_size))
    return matrix.multiply(1.0 / prefix).tocsr()


def sequence_documents(x: np.ndarray, prefix: int) -> list[str]:
    return [" ".join(map(str, row[:prefix])) for row in x]


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


def fit_candidate_models(
    model_name: str,
    train_features,
    validation_features,
    test_features,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray, dict]:
    best = None
    for c in CS:
        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=SEED,
        )
        started = time.perf_counter()
        model.fit(train_features, y_train)
        training_seconds = time.perf_counter() - started
        validation_probability = model.predict_proba(validation_features)[:, 1]
        threshold = choose_threshold(y_validation, validation_probability)
        validation_record = metric_record(
            y_validation, validation_probability, threshold
        )
        score = (
            validation_record["macro_f1"],
            validation_record["balanced_accuracy"],
            -c,
        )
        if best is None or score > best[0]:
            best = (
                score,
                model,
                validation_probability,
                threshold,
                training_seconds,
                c,
            )

    _, model, validation_probability, threshold, training_seconds, c = best
    started = time.perf_counter()
    test_probability = model.predict_proba(test_features)[:, 1]
    inference_seconds = time.perf_counter() - started
    details = {
        "model": model_name,
        "C": c,
        "threshold": threshold,
        "training_seconds": training_seconds,
        "inference_ms_per_sample": inference_seconds / len(y_test) * 1000,
    }
    return model, validation_probability, test_probability, details


def run_prefix_experiments(x: np.ndarray, y: np.ndarray, splits: dict):
    vectorizer = HashingVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        ngram_range=(1, 2),
        n_features=2**15,
        alternate_sign=False,
        norm="l2",
    )
    y_train = y[splits["train"]]
    y_validation = y[splits["validation"]]
    y_test = y[splits["test"]]
    records = []
    predictions = {}

    for prefix in PREFIXES:
        feature_sets = {}
        frequency = frequency_features(x, prefix)
        feature_sets["Frequency"] = (
            frequency[splits["train"]],
            frequency[splits["validation"]],
            frequency[splits["test"]],
        )
        documents = sequence_documents(x, prefix)
        order_features = vectorizer.transform(documents)
        feature_sets["Order-aware 1-2 gram"] = (
            order_features[splits["train"]],
            order_features[splits["validation"]],
            order_features[splits["test"]],
        )

        for model_name, (train_features, validation_features, test_features) in feature_sets.items():
            model, validation_probability, test_probability, details = fit_candidate_models(
                model_name,
                train_features,
                validation_features,
                test_features,
                y_train,
                y_validation,
                y_test,
            )
            record = {"prefix_calls": prefix, **details}
            record.update(metric_record(y_test, test_probability, details["threshold"]))
            record.update(
                bootstrap_intervals(y_test, test_probability, details["threshold"])
            )
            records.append(record)
            key = f"{model_name}|{prefix}"
            predictions[key] = {
                "validation": validation_probability,
                "test": test_probability,
                "threshold": details["threshold"],
            }
            joblib.dump(model, MODELS / f"{model_name.lower().replace(' ', '_').replace('-', '_')}_{prefix}.joblib")
            print(
                f"{model_name:20s} prefix={prefix:3d} "
                f"bal_acc={record['balanced_accuracy']:.3f} "
                f"macro_f1={record['macro_f1']:.3f} auc={record['roc_auc']:.3f}"
            )
    return pd.DataFrame(records), predictions


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


def naive_row_split_comparison(raw_df: pd.DataFrame) -> dict:
    x = raw_df[[f"t_{i}" for i in range(100)]].to_numpy(dtype=np.int16)
    y = raw_df["malware"].to_numpy(dtype=np.int8)
    train_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=SEED, stratify=y
    )
    train_fingerprints = {sequence_fingerprint(row) for row in x[train_indices]}
    leakage = np.array(
        [sequence_fingerprint(row) in train_fingerprints for row in x[test_indices]]
    )
    vectorizer = HashingVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        ngram_range=(1, 2),
        n_features=2**15,
        alternate_sign=False,
        norm="l2",
    )
    documents = sequence_documents(x, 100)
    features = vectorizer.transform(documents)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(features[train_indices], y[train_indices])
    probability = model.predict_proba(features[test_indices])[:, 1]
    metrics = metric_record(y[test_indices], probability, 0.5)
    return {
        "split": "naive random row split",
        "test_rows": int(len(test_indices)),
        "test_rows_with_sequence_seen_in_training": int(leakage.sum()),
        "test_sequence_leakage_rate": float(leakage.mean()),
        **metrics,
    }


def save_figures(
    results: pd.DataFrame,
    policy: pd.DataFrame,
    predictions: dict,
    y_test: np.ndarray,
    audit: dict,
):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    palette = {"Frequency": "#64748b", "Order-aware 1-2 gram": "#0f8ebd"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for model_name, group in results.groupby("model"):
        axes[0].errorbar(
            group["prefix_calls"],
            group["balanced_accuracy"],
            yerr=np.vstack(
                [
                    group["balanced_accuracy"] - group["balanced_accuracy_ci_low"],
                    group["balanced_accuracy_ci_high"] - group["balanced_accuracy"],
                ]
            ),
            marker="o",
            capsize=3,
            label=model_name,
            color=palette[model_name],
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
            color=palette[model_name],
        )
    axes[0].set(title="Balanced accuracy by observed API calls", xlabel="Observed API calls", ylabel="Balanced accuracy", ylim=(0.0, 1.01))
    axes[1].set(title="Macro F1 by observed API calls", xlabel="Observed API calls", ylabel="Macro F1", ylim=(0.0, 1.01))
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.savefig(FIGURES / "prefix_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "prefix_performance.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.plot(policy["call_budget"], policy["malware_detection_coverage"], marker="o", label="Malware detected", color="#0f8ebd")
    ax.plot(policy["call_budget"], policy["benign_false_alarm_rate"], marker="o", label="Benign false alarms", color="#dc2626")
    ax.set(title="Sequential early-detection policy", xlabel="API-call budget", ylabel="Cumulative proportion", ylim=(-0.02, 1.02))
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "early_detection_policy.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "early_detection_policy.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for prefix in [10, 20, 40, 100]:
        probability = predictions[f"Order-aware 1-2 gram|{prefix}"]["test"]
        fpr, tpr, _ = roc_curve(y_test, probability)
        axes[0].plot(fpr, tpr, label=f"{prefix} calls (AUC {roc_auc_score(y_test, probability):.3f})")
        precision, recall, _ = precision_recall_curve(1 - y_test, 1 - probability)
        axes[1].plot(recall, precision, label=f"{prefix} calls (AP {average_precision_score(1-y_test, 1-probability):.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
    axes[0].set(title="ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Benign-class precision-recall", xlabel="Benign recall", ylabel="Benign precision")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "roc_pr_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "roc_pr_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    chosen_prefix = int(
        results.loc[results["model"] == "Order-aware 1-2 gram"]
        .sort_values("prefix_calls")
        .query("balanced_accuracy >= 0.85")
        ["prefix_calls"]
        .min()
    ) if (results.query("model == 'Order-aware 1-2 gram'")["balanced_accuracy"] >= 0.85).any() else 100
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)
    for ax, prefix in zip(axes, [chosen_prefix, 100]):
        item = predictions[f"Order-aware 1-2 gram|{prefix}"]
        prediction = (item["test"] >= item["threshold"]).astype(np.int8)
        matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(title=f"{prefix} observed calls", xlabel="Predicted class", ylabel="True class", xticklabels=["Benign", "Malware"], yticklabels=["Benign", "Malware"])
    fig.savefig(FIGURES / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    categories = ["Raw benign", "Raw malware", "Unique benign", "Unique malware"]
    values = [audit["raw_benign"], audit["raw_malware"], audit["clean_benign"], audit["clean_malware"]]
    colors = ["#94a3b8", "#0f8ebd", "#64748b", "#08749d"]
    ax.bar(categories, values, color=colors)
    ax.set(title="Dataset composition before and after sequence deduplication", ylabel="Samples")
    ax.tick_params(axis="x", rotation=15)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    fig.savefig(FIGURES / "dataset_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "dataset_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
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


if __name__ == "__main__":
    main()
