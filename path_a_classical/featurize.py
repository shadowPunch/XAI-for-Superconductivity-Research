"""
path_a_classical/featurize.py
------------------------------
Generates 132-feature Magpie composition descriptors using matminer
for every formula in the merged dataset.

Output: data/processed/featurized_dataset.pkl
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FULL_DATASET_PKL, FEAT_DATASET_PKL, MAGPIE_PRESET

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def featurize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert chemical formulas → 132-dimensional Magpie feature vectors.

    Uses matminer's ElementProperty featurizer with the 'magpie' preset.
    Invalid/unparseable formulas are dropped.

    Parameters
    ----------
    df : DataFrame with at least columns ['formula', 'label', 'Tc']

    Returns
    -------
    DataFrame with original columns + 132 Magpie feature columns
    """
    try:
        from matminer.featurizers.composition import ElementProperty
        from matminer.featurizers.base import MultipleFeaturizer
        from pymatgen.core import Composition
    except ImportError as e:
        log.error(f"Missing dependency: {e}\nRun: pip install matminer pymatgen")
        sys.exit(1)

    log.info("Parsing chemical formulas with pymatgen …")
    compositions = []
    valid_idx = []

    for i, formula in enumerate(tqdm(df["formula"], desc="Parsing formulas")):
        try:
            comp = Composition(formula)
            if len(comp) == 0:
                continue
            compositions.append(comp)
            valid_idx.append(i)
        except Exception:
            pass  # drop unparseable formula silently

    df_valid = df.iloc[valid_idx].copy().reset_index(drop=True)
    df_valid["composition"] = compositions

    n_dropped = len(df) - len(df_valid)
    log.info(f"Parsed {len(df_valid):,} formulas ({n_dropped:,} dropped as invalid).")

    # ── Magpie featurization ──────────────────────────────────────────────────
    log.info(f"Generating Magpie features (preset='{MAGPIE_PRESET}') …")
    ep = ElementProperty.from_preset(MAGPIE_PRESET)
    ep.set_n_jobs(1)   # avoid multiprocessing issues on some platforms

    feat_df = ep.featurize_dataframe(
        df_valid,
        col_id="composition",
        ignore_errors=True,
        pbar=True,
    )

    # Drop rows where featurization failed (NaN in feature columns)
    feature_cols = ep.feature_labels()
    before = len(feat_df)
    feat_df = feat_df.dropna(subset=feature_cols).reset_index(drop=True)
    log.info(
        f"Featurization complete. "
        f"{len(feat_df):,} rows retained ({before - len(feat_df):,} dropped)."
    )

    return feat_df, feature_cols


def main():
    if not FULL_DATASET_PKL.exists():
        log.error(f"Full dataset not found at {FULL_DATASET_PKL}.")
        log.error("Run:  python data/preprocess.py")
        sys.exit(1)

    if FEAT_DATASET_PKL.exists():
        log.info(f"Featurized dataset already exists at {FEAT_DATASET_PKL}. Skipping.")
        return

    df = pd.read_pickle(FULL_DATASET_PKL)
    log.info(f"Loaded {len(df):,} materials from {FULL_DATASET_PKL}.")

    feat_df, feature_cols = featurize_dataset(df)

    # Save
    feat_df.to_pickle(FEAT_DATASET_PKL)
    log.info(f"Saved featurized dataset → {FEAT_DATASET_PKL}")
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]} … (first 5 shown)")


if __name__ == "__main__":
    main()
