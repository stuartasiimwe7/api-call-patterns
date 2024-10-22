from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import SEED
from .data import sequence_fingerprint
from .evaluation import metric_record
from .features import sequence_documents


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
