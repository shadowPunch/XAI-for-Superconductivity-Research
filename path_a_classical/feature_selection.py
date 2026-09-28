"""
path_a_classical/feature_selection.py
--------------------------------------
Two-stage feature selection as described in the report:

  Stage 1 – Pearson correlation pruning
      Drop one feature from any pair with |r| > threshold (default 0.95).

  Stage 2 – Recursive Feature Elimination (RFE)
      Use a LightGBM estimator to iteratively remove the least important
      features until only the top-40 remain.

Returns the list of selected feature names.
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEAT_DATASET_PKL, PROC_DIR,
    PEARSON_CORR_THRESHOLD, RFE_N_FEATURES, RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SELECTED_FEATURES_FILE = PROC_DIR / "selected_features.txt"


# ── Stage 1: Pearson correlation pruning ─────────────────────────────────────

def remove_correlated_features(
    X: pd.DataFrame,
    threshold: float = PEARSON_CORR_THRESHOLD,
) -> List[str]:
    """
    Compute the absolute Pearson correlation matrix and greedily remove
    one feature from each highly-correlated pair (|r| > threshold).

    Returns the list of surviving feature names.
    """
    log.info(f"Stage 1: Removing features with |Pearson r| > {threshold} …")
    corr_matrix = X.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Columns to drop: any column with at least one correlation above threshold
    to_drop = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > threshold)
    ]
    surviving = [c for c in X.columns if c not in to_drop]

    log.info(
        f"  Dropped {len(to_drop):,} correlated features. "
        f"{len(surviving):,} remain."
    )
    return surviving


# ── Stage 2: RFE with LightGBM ───────────────────────────────────────────────

def rfe_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = RFE_N_FEATURES,
) -> List[str]:
    """
    Use sklearn's RFECV-style wrapper around a LightGBM classifier to select
    the top `n_features` most predictive features.
    """
    try:
        from lightgbm import LGBMClassifier
        from sklearn.feature_selection import RFE
    except ImportError as e:
        log.error(f"Missing dependency: {e}\nRun: pip install lightgbm scikit-learn")
        sys.exit(1)

    log.info(f"Stage 2: RFE with LightGBM → selecting top {n_features} features …")

    estimator = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=5)
    rfe.fit(X, y)

    selected = X.columns[rfe.support_].tolist()
    log.info(f"  RFE selected {len(selected)} features.")
    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full two-stage feature selection pipeline.

    Returns
    -------
    X_selected : DataFrame with only the selected feature columns
    selected_features : list of selected column names
    """
    # Stage 1
    stage1_features = remove_correlated_features(X)
    X_stage1 = X[stage1_features]

    # Stage 2
    selected_features = rfe_feature_selection(X_stage1, y)
    X_selected = X[selected_features]

    return X_selected, selected_features


def main():
    if not FEAT_DATASET_PKL.exists():
        log.error(f"Featurized dataset not found at {FEAT_DATASET_PKL}.")
        log.error("Run:  python path_a_classical/featurize.py")
        sys.exit(1)

    df = pd.read_pickle(FEAT_DATASET_PKL)

    # Identify feature columns (everything that isn't metadata)
    meta_cols = ["formula", "Tc", "label", "source", "composition"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols]
    y = df["label"]

    _, selected = select_features(X, y)

    # Persist selection
    SELECTED_FEATURES_FILE.write_text("\n".join(selected))
    log.info(f"Saved selected feature list → {SELECTED_FEATURES_FILE}")
    log.info(f"Selected features: {selected}")


if __name__ == "__main__":
    main()
