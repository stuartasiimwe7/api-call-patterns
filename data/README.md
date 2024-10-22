# Dataset

`raw/dynamic_api_call_sequence_per_malware_100_0_306.csv` contains the first 100 non-repeated consecutive Windows API calls associated with the parent process in Cuckoo Sandbox reports.

Source dataset: Angelo Oliveira, *Malware Analysis Datasets: API Call Sequences* (2019), originally distributed through IEEE DataPort and later Kaggle.

Acquisition source for this reproducibility study:
https://github.com/RenatoMignone/malware-api-classification-dl

The CSV is retained unchanged. Processing code fingerprints all 100-call sequences, removes exact duplicate behavior traces, and excludes sequence groups with contradictory labels before splitting.
