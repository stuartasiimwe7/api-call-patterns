from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from .config import CS, MODELS, PREFIXES, SEED
from .evaluation import bootstrap_intervals, choose_threshold, metric_record
from .features import frequency_features, sequence_documents


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

        for model_name, feature_set in feature_sets.items():
            train_features, validation_features, test_features = feature_set
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
            model_slug = model_name.lower().replace(" ", "_").replace("-", "_")
            joblib.dump(model, MODELS / f"{model_slug}_{prefix}.joblib")
            print(
                f"{model_name:20s} prefix={prefix:3d} "
                f"bal_acc={record['balanced_accuracy']:.3f} "
                f"macro_f1={record['macro_f1']:.3f} auc={record['roc_auc']:.3f}"
            )
    return pd.DataFrame(records), predictions
