"""
data/download_data.py
---------------------
Downloads the negative (non-superconductor) dataset from the Materials Project.

The POSITIVE dataset (SuperCon) must be obtained manually:
  1. Go to https://mdr.nims.go.jp/collections/8s45q879z
  2. Download the SuperCon CSV and save it to:  data/raw/supercon.csv

Run this script once before anything else:
    python data/download_data.py
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MP_API_KEY, DATA_DIR, EHULL_MAX

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MP_NEGATIVE_CSV = DATA_DIR / "mp_stable_non_sc.csv"


def fetch_mp_stable_materials(api_key: str, ehull_max: float = EHULL_MAX) -> pd.DataFrame:
    """
    Query the Materials Project for all computationally stable materials
    (energy above hull ≤ ehull_max eV/atom).

    Returns a DataFrame with columns: material_id, formula, energy_above_hull.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        log.error("mp-api not installed. Run: pip install mp-api")
        sys.exit(1)

    if api_key == "YOUR_MP_API_KEY_HERE":
        log.error(
            "Please set a valid MP_API_KEY in config.py or the MP_API_KEY env variable."
        )
        sys.exit(1)

    log.info("Connecting to Materials Project API …")
    records = []
    with MPRester(api_key) as mpr:
        # Fetch in chunks to avoid timeouts
        results = mpr.materials.summary.search(
            energy_above_hull=(0, ehull_max),
            fields=["material_id", "formula_pretty", "energy_above_hull"],
        )
        for doc in tqdm(results, desc="Fetching MP entries"):
            records.append(
                {
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "energy_above_hull": doc.energy_above_hull,
                }
            )

    df = pd.DataFrame(records)
    log.info(f"Fetched {len(df):,} stable MP materials (Ehull ≤ {ehull_max} eV/atom).")
    return df


def main():
    if MP_NEGATIVE_CSV.exists():
        log.info(f"MP negative dataset already exists at {MP_NEGATIVE_CSV}. Skipping download.")
        return

    df = fetch_mp_stable_materials(MP_API_KEY)
    df.to_csv(MP_NEGATIVE_CSV, index=False)
    log.info(f"Saved → {MP_NEGATIVE_CSV}")


if __name__ == "__main__":
    main()
