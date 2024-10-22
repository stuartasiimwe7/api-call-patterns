from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/dynamic_api_call_sequence_per_malware_100_0_306.csv"
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
MODELS = ROOT / "results/models"
SEED = 20241022
PREFIXES = [5, 10, 20, 40, 60, 80, 100]
CS = [0.1, 1.0, 10.0]
N_BOOTSTRAP = 1000
