"""
run_pipeline.py
----------------
End-to-end runner for the superconductor screening funnel.

Steps
-----
  0. (One-time setup, see README) data/preprocess.py, then
     path_a_classical/featurize.py and, optionally, feature_selection.py.
     The raw CSVs in data/raw/ are committed; MP_API_KEY is only needed to
     re-run data/download_data.py.
  1. path_a_classical/build_screening_pipeline.py -- builds the labeled
     dataset (confirmed positives + confirmed negatives) and the unlabeled
     screening pool, trains the two-stage funnel, evaluates it honestly
     with leave-one-family-out splits.
  2. path_a_classical/run_screening.py -- scores the unlabeled pool,
     produces the ranked candidate shortlist.
  3. path_a_classical/annotate_shortlist.py -- flags rediscoveries and
     attaches per-candidate family + reliability context.

Usage
-----
    python run_pipeline.py
"""

import sys
import logging
import importlib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "path_a_classical"))


def section(title):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


def run_step(module_name):
    module = importlib.import_module(module_name)
    module.main()


def main():
    section("Step 1: Build + train + evaluate the screening funnel")
    run_step("build_screening_pipeline")

    section("Step 2: Score the unlabeled pool -> ranked shortlist")
    run_step("run_screening")

    section("Step 3: Annotate shortlist (rediscoveries, family, reliability)")
    run_step("annotate_shortlist")

    from config import OUTPUT_DIR
    section("Pipeline complete")
    log.info(f"All outputs are in: {OUTPUT_DIR}/")
    log.info("  screening_funnel_classifier.pkl / .json  -- trained classifier + honest LOFO metrics")
    log.info("  screening_funnel_regressor.pkl / .json   -- trained Tc regressor + honest LOFO metrics")
    log.info("  candidate_shortlist.csv                  -- full ranked pool (~62,000 candidates)")
    log.info("  candidate_shortlist_top500_annotated.csv -- final deliverable: top 500, with")
    log.info("                                              rediscovery flags + per-candidate reliability")


if __name__ == "__main__":
    main()
