from __future__ import annotations

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    LSTM_BATCH_SIZE,
    LSTM_EMBEDDING_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_LEARNING_RATE,
    LSTM_MAX_EPOCHS,
    LSTM_PATIENCE,
    MODELS,
    PREFIXES,
    SEED,
)
from .evaluation import bootstrap_intervals, choose_threshold, metric_record


class ApiCallLstm(nn.Module):
    def __init__(self, vocabulary_size: int = 307) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, LSTM_EMBEDDING_DIM)
        self.lstm = nn.LSTM(
            input_size=LSTM_EMBEDDING_DIM,
            hidden_size=LSTM_HIDDEN_DIM,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(LSTM_HIDDEN_DIM, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequence)
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(self.dropout(hidden[-1])).squeeze(1)


def set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.long),
        torch.as_tensor(y, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.inference_mode()
def predict_probabilities(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    batches = []
    for sequences, _ in _loader(
        x,
        np.zeros(len(x), dtype=np.float32),
        shuffle=False,
        seed=SEED,
    ):
        batches.append(torch.sigmoid(model(sequences)).cpu().numpy())
    return np.concatenate(batches)


def fit_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int,
    vocabulary_size: int,
) -> tuple[ApiCallLstm, np.ndarray, float, dict, list[dict]]:
    set_deterministic_seed(seed)
    model = ApiCallLstm(vocabulary_size=vocabulary_size)
    optimiser = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * class_counts)
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    train_loader = _loader(x_train, y_train, shuffle=True, seed=seed)

    best_state = None
    best_probability = None
    best_threshold = 0.5
    best_score = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, LSTM_MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        observations = 0
        for sequences, labels in train_loader:
            optimiser.zero_grad(set_to_none=True)
            logits = model(sequences)
            sample_weights = weight_tensor[labels.long()]
            loss = (criterion(logits, labels) * sample_weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.detach().item() * len(labels)
            observations += len(labels)

        validation_probability = predict_probabilities(model, x_validation)
        threshold = choose_threshold(y_validation, validation_probability)
        validation_record = metric_record(
            y_validation,
            validation_probability,
            threshold,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss / observations,
                "validation_threshold": threshold,
                "validation_balanced_accuracy": validation_record[
                    "balanced_accuracy"
                ],
                "validation_macro_f1": validation_record["macro_f1"],
                "validation_roc_auc": validation_record["roc_auc"],
                "validation_false_positive_rate": validation_record[
                    "false_positive_rate"
                ],
                "validation_malware_recall": validation_record[
                    "malware_recall"
                ],
            }
        )
        score = (
            validation_record["macro_f1"],
            validation_record["balanced_accuracy"],
            -epoch,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_probability = validation_probability.copy()
            best_threshold = threshold
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= LSTM_PATIENCE:
                break

    model.load_state_dict(best_state)
    details = {
        "model": "LSTM",
        "C": np.nan,
        "threshold": best_threshold,
        "training_seconds": time.perf_counter() - started,
        "selected_epoch": best_epoch,
        "embedding_dim": LSTM_EMBEDDING_DIM,
        "hidden_dim": LSTM_HIDDEN_DIM,
    }
    return model, best_probability, best_threshold, details, history


def run_lstm_experiments(
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    y_train = y[splits["train"]]
    y_validation = y[splits["validation"]]
    y_test = y[splits["test"]]
    records = []
    predictions = {}
    training_history = []
    vocabulary_size = int(x.max()) + 1

    for prefix in PREFIXES:
        model, validation_probability, threshold, details, history = fit_lstm(
            x[splits["train"], :prefix],
            y_train,
            x[splits["validation"], :prefix],
            y_validation,
            seed=SEED + prefix,
            vocabulary_size=vocabulary_size,
        )
        training_history.extend(
            {"prefix_calls": prefix, **epoch_record}
            for epoch_record in history
        )
        started = time.perf_counter()
        test_probability = predict_probabilities(
            model,
            x[splits["test"], :prefix],
        )
        details["inference_ms_per_sample"] = (
            (time.perf_counter() - started) / len(y_test) * 1000
        )
        record = {"prefix_calls": prefix, **details}
        record.update(metric_record(y_test, test_probability, threshold))
        record.update(bootstrap_intervals(y_test, test_probability, threshold))
        records.append(record)
        predictions[f"LSTM|{prefix}"] = {
            "validation": validation_probability,
            "test": test_probability,
            "threshold": threshold,
        }
        torch.save(
            {
                "state_dict": model.state_dict(),
                "prefix_calls": prefix,
                "vocabulary_size": vocabulary_size,
                "embedding_dim": LSTM_EMBEDDING_DIM,
                "hidden_dim": LSTM_HIDDEN_DIM,
                "selected_epoch": details["selected_epoch"],
                "threshold": threshold,
            },
            MODELS / f"lstm_{prefix}.pt",
        )
        print(
            f"{'LSTM':20s} prefix={prefix:3d} "
            f"epoch={details['selected_epoch']:2d} "
            f"bal_acc={record['balanced_accuracy']:.3f} "
            f"macro_f1={record['macro_f1']:.3f} "
            f"auc={record['roc_auc']:.3f}"
        )

    return pd.DataFrame(records), predictions, pd.DataFrame(training_history)
