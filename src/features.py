from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def frequency_features(x: np.ndarray, prefix: int, vocabulary_size: int = 307) -> csr_matrix:
    rows = np.repeat(np.arange(len(x)), prefix)
    cols = x[:, :prefix].reshape(-1)
    values = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix((values, (rows, cols)), shape=(len(x), vocabulary_size))
    return matrix.multiply(1.0 / prefix).tocsr()


def sequence_documents(x: np.ndarray, prefix: int) -> list[str]:
    return [" ".join(map(str, row[:prefix])) for row in x]
