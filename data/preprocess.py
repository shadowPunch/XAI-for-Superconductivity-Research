"""
data/preprocess.py
------------------
Merges the SuperCon positive dataset with the Materials Project negative dataset
to produce the unified ~78,500-material dataset described in the report.

Outputs
-------
data/processed/full_dataset.pkl   – DataFrame with columns:
    formula   : chemical formula string
    Tc        : critical temperature (K); 0 for non-superconductors
    label     : 1 = superconductor, 0 = non-superconductor
    source    : "supercon" | "mp"
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SUPERCON_CSV, DATA_DIR, FULL_DATASET_PKL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MP_NEGATIVE_CSV = DATA_DIR / "mp_stable_non_sc.csv"


# ── SuperCon parsing ──────────────────────────────────────────────────────────

def load_supercon(path: Path) -> pd.DataFrame:
    """
    Load and clean the SuperCon CSV.

    The raw file typically has columns like 'name' (formula) and 'Tc'.
    We keep the highest reported Tc for each unique formula.
    Column names are normalised to handle different SuperCon file versions.
    """
    log.info(f"Loading SuperCon from {path} …")
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    # Normalise column names (lowercase, strip whitespace)
    df.columns = [c.strip().lower() for c in df.columns]

    # Map common column name variants. Rename only the FIRST match per target:
    # the file can carry several candidates at once (e.g. both "formula" and
    # "composition"), and renaming all of them yields duplicate column names.
    for candidates, target in [
        (("formula", "name", "material", "composition"), "formula"),
        (("tc", "critical_temperature", "tc(k)", "tc_k"), "Tc"),
    ]:
        for col in candidates:
            if col in df.columns:
                df = df.rename(columns={col: target})
                break

    if "formula" not in df.columns or "Tc" not in df.columns:
        raise ValueError(
            f"Could not find 'formula' and 'Tc' columns in SuperCon file.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Keep only valid rows
    df = df[["formula", "Tc"]].copy()
    df["formula"] = df["formula"].astype(str).str.strip()
    df["Tc"] = pd.to_numeric(df["Tc"], errors="coerce")
    df = df.dropna(subset=["formula", "Tc"])
    df = df[df["Tc"] > 0]  # must be a superconductor

    # Keep highest Tc per formula
    df = df.groupby("formula", as_index=False)["Tc"].max()

    df["label"]  = 1
    df["source"] = "supercon"
    log.info(f"  → {len(df):,} unique superconducting formulas after cleaning.")
    return df


# ── MP negative set parsing ───────────────────────────────────────────────────

def load_mp_negatives(path: Path, supercon_formulas: set) -> pd.DataFrame:
    """
    Load the pre-downloaded MP stable materials CSV and exclude any formula
    that appears in the SuperCon positive set.
    """
    log.info(f"Loading MP negative set from {path} …")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise formula column name
    for col in df.columns:
        if col in ("formula", "formula_pretty", "composition"):
            df = df.rename(columns={col: "formula"})
            break

    df["formula"] = df["formula"].astype(str).str.strip()

    # Remove any that are actually in SuperCon
    df = df[~df["formula"].isin(supercon_formulas)]

    df["Tc"]     = 0.0
    df["label"]  = 0
    df["source"] = "mp"
    df = df[["formula", "Tc", "label", "source"]]

    log.info(f"  → {len(df):,} non-superconducting formulas from MP.")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    if not SUPERCON_CSV.exists():
        raise FileNotFoundError(
            f"SuperCon file not found at {SUPERCON_CSV}.\n"
            "Please download it from https://mdr.nims.go.jp/ and save it there."
        )
    if not MP_NEGATIVE_CSV.exists():
        raise FileNotFoundError(
            f"MP negative set not found at {MP_NEGATIVE_CSV}.\n"
            "Please run:  python data/download_data.py"
        )

    supercon_df = load_supercon(SUPERCON_CSV)
    supercon_formulas = set(supercon_df["formula"])

    mp_df = load_mp_negatives(MP_NEGATIVE_CSV, supercon_formulas)

    full_df = pd.concat([supercon_df, mp_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["formula"]).reset_index(drop=True)

    log.info(
        f"\nDataset summary:\n"
        f"  Superconductors (label=1) : {(full_df.label == 1).sum():>7,}\n"
        f"  Non-superconductors (label=0) : {(full_df.label == 0).sum():>7,}\n"
        f"  Total                     : {len(full_df):>7,}"
    )

    full_df.to_pickle(FULL_DATASET_PKL)
    log.info(f"Saved → {FULL_DATASET_PKL}")
    return full_df


if __name__ == "__main__":
    build_dataset()
