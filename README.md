# API Call Patterns

The study tests how early malicious Windows behaviour can be detected from the first 5, 10, 20, 40, 60, 80, or 100 API calls. It asks:

1. How does detection quality change as the observed prefix grows?
2. Does local call order improve on call-frequency features?
3. How early can a sequential policy alert while controlling benign false alarms?

Call-frequency, local-order, and recurrent representations are compared to answer these questions.

## Results

- 13,191 unique labelled traces after removing exact duplicates and conflicting labels
- 0.818 balanced accuracy and 0.919 ROC-AUC after 20 calls
- 0.904 balanced accuracy after 80 calls
- 83.6% malware detection by 100 calls with a 5.4% benign false-alarm rate
- Median alert at 60 calls among detected malware
- Longer-range sequence modelling did not improve performance consistently and produced a higher sequential false-alarm rate

Hyperparameters and alert thresholds are selected on the validation set. The test set is reserved for final evaluation.

## Dataset

The experiment uses **Malware Analysis Datasets: API Call Sequences**, created by Angelo Oliveira and published through IEEE DataPort in 2019.

The local CSV was acquired from the [malware-api-classification-dl](https://github.com/RenatoMignone/malware-api-classification-dl) repository. The source file is retained unchanged and is not committed to this repository.

Expected path:

```text
data/raw/dynamic_api_call_sequence_per_malware_100_0_306.csv
```

SHA-256:

```text
11005ff6f5007bfee7d60bd0dc2e787f4e77b46f3b0a3d09424421c5339a8406
```

See [`data/README.md`](data/README.md) for details.

## Run

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Generated figures, tables, predictions, and fitted models are written to `results/`.

## Structure

```text
data/               Dataset instructions and local source data
paper/              Manuscript PDF, LaTeX source, and watermark assets
results/figures/    Evaluation figures
results/models/     Fitted models for each call budget
results/tables/     Metrics, predictions, and audit records
main.py             Experiment entry point
src/                Data, features, models, evaluation, policy, and plotting
```

## Citation

```bibtex
@misc{asiimwe2024temporal,
  author = {Asiimwe, Stuart},
  title = {Temporal Analysis of API Call Patterns for Early Detection of Malicious Behaviour},
  year = {2024},
  month = {10},
  note = {Author manuscript and reproducible experiment}
}
```

When using the dataset, also cite its original source:

```bibtex
@dataset{oliveira2019malware,
  author = {Oliveira, Angelo},
  title = {Malware Analysis Datasets: API Call Sequences},
  publisher = {IEEE DataPort},
  year = {2019}
}
```
