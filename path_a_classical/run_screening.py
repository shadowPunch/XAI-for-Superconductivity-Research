"""
path_a_classical/run_screening.py
-------------------------------------
Applies the trained two-stage funnel (build_screening_pipeline.py) to the
unlabeled pool (~62,000 MP-stable materials with no confirmed SC status
either way) and produces a ranked candidate shortlist.

Funnel score = P(superconductor-like) x predicted Tc -- ranks candidates
that are both LIKELY to be superconducting AND predicted to have a
worthwhile critical temperature. This is a shortlisting tool for triage,
not a discovery claim: see the LOFO metrics in screening_funnel_classifier.json
for how much to trust a ranking in a given chemical family before acting
on it (Cuprates/Bismuthates/Other: ROC-AUC 0.93-0.96; Iron-based/Borocarbides:
0.64-0.66 -- still usable, just noisier).

Output
------
outputs/candidate_shortlist.csv        -- full pool, ranked
outputs/candidate_shortlist_top500.csv
"""

import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROC_DIR, OUTPUT_DIR
from tracking import start_run, file_md5, record_provenance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SELECTED_FEATURES_FILE = PROC_DIR / "selected_features.txt"
UNLABELED_POOL_PKL = PROC_DIR / "screening_unlabeled_pool.pkl"
CLF_MODEL_PKL = OUTPUT_DIR / "screening_funnel_classifier.pkl"
REG_MODEL_PKL = OUTPUT_DIR / "screening_funnel_regressor.pkl"

OUT_FULL = OUTPUT_DIR / "candidate_shortlist.csv"
OUT_TOP500 = OUTPUT_DIR / "candidate_shortlist_top500.csv"

# Famous known superconductors that a working funnel should rank near the top
SANITY_FORMULAS = ("H3S", "Ba2YCu3O7")


def main():
    if not CLF_MODEL_PKL.exists() or not REG_MODEL_PKL.exists():
        log.error("Trained funnel not found. Run build_screening_pipeline.py first.")
        sys.exit(1)

    feature_cols = SELECTED_FEATURES_FILE.read_text().splitlines()
    pool = pd.read_pickle(UNLABELED_POOL_PKL)
    log.info(f"Unlabeled pool to screen: {len(pool):,}")

    with open(CLF_MODEL_PKL, "rb") as f:
        clf = pickle.load(f)
    with open(REG_MODEL_PKL, "rb") as f:
        reg = pickle.load(f)

    config = {
        "features": feature_cols, "n_pool": len(pool), "sanity_formulas": SANITY_FORMULAS,
        "classifier_md5": file_md5(CLF_MODEL_PKL), "regressor_md5": file_md5(REG_MODEL_PKL),
        "pool_md5": file_md5(UNLABELED_POOL_PKL),
    }
    with start_run("score-pool", "inference", config, tags=["funnel", "shortlist"],
                   notes="Scores the unlabeled pool with the shipped funnel and writes the ranked shortlist.") as run:
        X_pool = pool[feature_cols].values
        p_sc = clf.predict_proba(X_pool)[:, 1]
        pred_tc = np.clip(np.sinh(reg.predict(X_pool)), 0, None)

        pool_out = pool[["formula", "source"]].copy()
        pool_out["p_superconductor"] = p_sc
        pool_out["predicted_tc_K"] = pred_tc
        pool_out["funnel_score"] = p_sc * pred_tc
        pool_out = pool_out.sort_values("funnel_score", ascending=False).reset_index(drop=True)
        pool_out.insert(0, "rank", np.arange(1, len(pool_out) + 1))

        pool_out.to_csv(OUT_FULL, index=False)
        pool_out.head(500).to_csv(OUT_TOP500, index=False)
        log.info(f"Saved full ranked pool -> {OUT_FULL}")
        log.info(f"Saved top-500 shortlist -> {OUT_TOP500}")
        log.info(f"\nTop 25 candidates:\n{pool_out.head(25).to_string(index=False)}")

        log_outputs(run, pool_out)
        record_provenance("run_screening", run, [OUT_FULL, OUT_TOP500])


def log_outputs(run, pool_out):
    """Output statistics + a known-superconductor recovery check."""
    import wandb
    run.summary["pool/mean_p_superconductor"] = float(pool_out.p_superconductor.mean())
    run.summary["pool/n_p_gt_0.9"] = int((pool_out.p_superconductor > 0.9).sum())
    run.summary["pool/predicted_tc_median_K"] = float(pool_out.predicted_tc_K.median())
    run.summary["pool/predicted_tc_max_K"] = float(pool_out.predicted_tc_K.max())
    run.summary["top500/mean_p_superconductor"] = float(pool_out.p_superconductor.head(500).mean())
    for formula in SANITY_FORMULAS:
        hit = pool_out.loc[pool_out.formula == formula, "rank"]
        run.summary[f"sanity/rank_{formula}"] = int(hit.iloc[0]) if len(hit) else None
    run.log({
        "top25": wandb.Table(dataframe=pool_out.head(25)),
        "funnel_score_hist": wandb.Histogram(pool_out.funnel_score.values),
    })


if __name__ == "__main__":
    main()
