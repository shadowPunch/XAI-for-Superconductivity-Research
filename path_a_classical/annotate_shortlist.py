"""
path_a_classical/annotate_shortlist.py
------------------------------------------
Final annotation pass on the top-500 shortlist (run_screening.py):

  1. Rediscovery flags -- some top-ranked candidates are already-known
     superconductors under a differently-formatted formula string (e.g.
     the unlabeled pool's "Ba2YCu3O7" is literally YBCO, just not an exact
     string match to how SuperCon records it). Flagged two ways: exact
     match by reduced formula, and the stricter same-element-set match
     (catches doping/stoichiometry variants of a known family, e.g. a
     different Hg-cuprate ratio).
  2. Family + reliability -- each candidate is tagged with the SuperCon
     family it composition-wise resembles (classify_supercon_category(),
     validated at 99.82% accuracy) and that family's actual measured
     leave-one-family-out reliability from screening_funnel_classifier.json,
     so the ranking comes with an honest sense of how much to trust it.

Output
------
outputs/candidate_shortlist_top500_annotated.csv
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, OUTPUT_DIR
from tracking import start_run, file_md5, record_provenance
from build_screening_pipeline import classify_supercon_category

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SHORTLIST_CSV = OUTPUT_DIR / "candidate_shortlist_top500.csv"
CLF_METRICS_JSON = OUTPUT_DIR / "screening_funnel_classifier.json"
SUPERCON_RAW = DATA_DIR / "supercon.csv"
OUT_CSV = OUTPUT_DIR / "candidate_shortlist_top500_annotated.csv"


def normalize(formula):
    from pymatgen.core import Composition
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def elemset(formula):
    from pymatgen.core import Composition
    try:
        return frozenset(str(e) for e in Composition(formula).elements)
    except Exception:
        return None


def main():
    shortlist = pd.read_csv(SHORTLIST_CSV)
    sc = pd.read_csv(SUPERCON_RAW, usecols=["formula"])

    config = {"input_md5": {p.name: file_md5(p) for p in (SHORTLIST_CSV, CLF_METRICS_JSON, SUPERCON_RAW)},
              "n_candidates": len(shortlist)}
    with start_run("annotate-shortlist", "inference", config, tags=["funnel", "shortlist"],
                   notes="Flags rediscoveries and attaches per-candidate family + reliability to the top-500.") as run:
        log.info("Flagging rediscoveries (candidates that are already-known superconductors) ...")
        sc_norm_set = set(sc["formula"].apply(normalize).dropna())
        sc_elemset_set = set(sc["formula"].apply(elemset).dropna())

        shortlist["already_known_supercon"] = shortlist["formula"].apply(normalize).isin(sc_norm_set)
        shortlist["same_element_set_as_known_SC"] = shortlist["formula"].apply(elemset).isin(sc_elemset_set)
        log.info(f"  exact rediscoveries (reduced formula): {shortlist['already_known_supercon'].sum()}")
        log.info(f"  same element combination as a known SC: {shortlist['same_element_set_as_known_SC'].sum()}")

        log.info("Tagging inferred family + LOFO reliability ...")
        lofo = json.loads(CLF_METRICS_JSON.read_text())["leave_one_family_out"]
        shortlist["inferred_family"] = shortlist["formula"].apply(classify_supercon_category)

        def reliability(fam):
            r = lofo.get(fam)
            return pd.Series({
                "family_lofo_roc_auc": r["roc_auc"] if r else None,
                "family_lofo_pr_auc": r["pr_auc"] if r else None,
            })

        shortlist = pd.concat([shortlist, shortlist["inferred_family"].apply(reliability)], axis=1)
        no_lofo = shortlist["family_lofo_roc_auc"].isna()
        if no_lofo.any():
            log.info(f"  {no_lofo.sum()} candidates have no measured reliability "
                     f"(family too small to evaluate, e.g. Hydrogen-rich) -- flagged, not silently omitted.")

        shortlist.to_csv(OUT_CSV, index=False)
        log.info(f"Saved -> {OUT_CSV}")
        log.info("\nFamily distribution across top-500:\n" + shortlist["inferred_family"].value_counts().to_string())
        log.info("\nTop 20:\n" + shortlist[["rank", "formula", "funnel_score", "inferred_family",
                                              "family_lofo_roc_auc", "already_known_supercon"]].head(20).to_string(index=False))

        log_outputs(run, shortlist)
        record_provenance("annotate_shortlist", run, [OUT_CSV])


def log_outputs(run, shortlist):
    import wandb
    run.summary["top500/exact_rediscoveries"] = int(shortlist["already_known_supercon"].sum())
    run.summary["top500/same_element_set_as_known_SC"] = int(shortlist["same_element_set_as_known_SC"].sum())
    run.summary["top500/no_reliability_available"] = int(shortlist["family_lofo_roc_auc"].isna().sum())
    for fam, n in shortlist["inferred_family"].value_counts().items():
        run.summary[f"top500/family/{fam.replace(' ', '_')}"] = int(n)
    bands = pd.cut(shortlist["family_lofo_roc_auc"], bins=[0, 0.6, 0.75, 0.9, 1.0],
                   labels=["low<=0.60", "moderate", "good", "high>0.90"])
    for band, n in bands.value_counts().items():
        run.summary[f"top500/reliability_band/{band}"] = int(n)
    run.log({"top20": wandb.Table(dataframe=shortlist.head(20))})


if __name__ == "__main__":
    main()
