# API Call Patterns

This project re-evaluates the 2024 author manuscript *Temporal Analysis of API Call Patterns for Early Detection of Malicious Behavior* with a reproducible early-detection protocol.

## Research question

How accurately can malicious Windows behavior be identified after observing only the first 5, 10, 20, 40, 60, 80, or 100 non-repeated API calls?

## Safeguards against optimistic results

- The source CSV is retained unchanged and checksummed.
- Exact 100-call behavior sequences are fingerprinted before splitting.
- Duplicate sequences are represented once in the primary experiment.
- Sequence groups with contradictory labels are excluded and reported.
- Train, validation, and test sets are stratified and fixed before model fitting.
- Class balancing is applied through training weights only.
- Thresholds and hyperparameters are selected using validation data only.
- The final test set remains untouched until evaluation.
- Performance is reported for both malware and benign samples because the source dataset is strongly imbalanced.

## Models

1. Frequency baseline: normalized API-call counts with class-weighted logistic regression.
2. Order-aware model: hashed API unigram and bigram features with class-weighted logistic regression.

The validation-selected sequential policy raises an alert after three consecutive checkpoints exceed their selected thresholds, while constraining cumulative validation false alarms to at most 5%.

## Run

```bash
.venv/bin/python src/run_experiment.py
```

Generated tables, model artifacts, and publication-ready figures are written under `results/`.

The source dataset is not committed. Place the CSV described in `data/README.md` under `data/raw/` before running the experiment.
