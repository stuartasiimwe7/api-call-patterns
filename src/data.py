from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA, SEED, TABLES


def sequence_fingerprint(row: np.ndarray) -> str:
    encoded = row.astype("<i2", copy=False).tobytes()
    return hashlib.sha256(encoded).hexdigest()


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
