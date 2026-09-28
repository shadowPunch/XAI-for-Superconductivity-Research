"""
config.py
---------
Central configuration for the superconductor screening funnel.
Edit the values here; all other modules import from this file.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data" / "raw"
PROC_DIR   = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"

for d in [DATA_DIR, PROC_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Raw file paths
SUPERCON_CSV      = DATA_DIR / "supercon.csv"          # download from NIMS MDR
FULL_DATASET_PKL  = PROC_DIR / "full_dataset.pkl"
FEAT_DATASET_PKL  = PROC_DIR / "featurized_dataset.pkl"

# ── Materials Project ─────────────────────────────────────────────────────────
# Register at https://next-gen.materialsproject.org/ and paste your key below.
MP_API_KEY = os.environ.get("MP_API_KEY", "YOUR_MP_API_KEY_HERE")

# Stability threshold: energy above hull (eV/atom)
EHULL_MAX = 0.05

# ── Data splits ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.10

# ── Featurization + feature selection ─────────────────────────────────────────
MAGPIE_PRESET          = "magpie"    # matminer preset name
PEARSON_CORR_THRESHOLD = 0.95        # drop features above this pairwise correlation
RFE_N_FEATURES         = 40          # number of features to keep after RFE

# ── Screening funnel (XGBoost classifier + regressor) ─────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":   500,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "eval_metric":    "logloss",
    "random_state":   RANDOM_SEED,
    "n_jobs":         -1,
}
