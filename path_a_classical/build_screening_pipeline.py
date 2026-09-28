"""
path_a_classical/build_screening_pipeline.py
------------------------------------------------
Builds and validates the two-stage superconductor screening funnel.

Label design: the naive setup (SuperCon = positive, "MP-stable-and-not-in-
SuperCon" = negative) is positive-unlabeled, not binary -- most of those
"negatives" have simply never been tested, so training on them as ground
truth teaches "which materials are in the SC literature", not "which
materials are/aren't superconductors". This pipeline uses only confirmed
labels:

  POSITIVES          : SuperCon entries with Tc > 0 (16,463).
  CONFIRMED NEGATIVES: SuperCon-origin entries with Tc = 0, sourced from the
                       3DSC_MP dataset (Sommer et al. 2023, whose SuperCon
                       source -- Stanev et al. 2018 -- explicitly curates
                       tested-and-not-superconducting compounds). 1,776
                       unique formulas.
  UNLABELED POOL     : the remaining ~62,000 MP-stable-and-not-in-SuperCon
                       materials. NOT used for training -- this is the
                       actual screening target, scored in run_screening.py.

Two-stage funnel, trained ONLY on confirmed labels:
  Stage 1 (classifier): P(is this composition superconductor-like at all?)
  Stage 2 (regressor) : predicted Tc (arcsinh-scaled), trained on positives
                        only, ranks whatever the classifier lets through.

Both stages are evaluated with LEAVE-ONE-FAMILY-OUT splits, using SuperCon's
own category taxonomy (Cuprates, Iron-based, Bismuthates, Borocarbides,
Elemental/Organic/Hydrogen-rich Superconductors, Other). Confirmed negatives
carry no such label natively (they come from 3DSC's differently-scoped
sc_class taxonomy), so they're assigned one via composition rules validated
at 99.82% accuracy against the 16,463 positives' real ground-truth labels
(see classify_supercon_category() below). Each LOFO fold excludes the WHOLE
family -- positives AND negatives -- from training: excluding only the
held-out positives while leaving same-family negatives in training biases
the model against exactly the family it should detect (verified: doing
that gave Cuprates ROC-AUC 0.41 and Iron-based 0.32, both below chance;
excluding the whole family gives 0.96 and 0.66 respectively -- the honest
numbers).

Outputs
-------
data/processed/confirmed_negatives_featurized.pkl
data/processed/screening_labeled_dataset.pkl
data/processed/screening_unlabeled_pool.pkl
outputs/screening_funnel_classifier.pkl / .json  (trained model + LOFO metrics)
outputs/screening_funnel_regressor.pkl / .json
"""

import sys
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEAT_DATASET_PKL, PROC_DIR, OUTPUT_DIR, DATA_DIR,
    MAGPIE_PRESET, TEST_SIZE, RANDOM_SEED, XGBOOST_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SELECTED_FEATURES_FILE = PROC_DIR / "selected_features.txt"
DSC_CSV = Path(__file__).parent.parent / "data" / "3dsc" / "3DSC_MP.csv"
SUPERCON_RAW = DATA_DIR / "supercon.csv"

NEG_FEAT_CACHE = PROC_DIR / "confirmed_negatives_featurized.pkl"
LABELED_DATASET_PKL = PROC_DIR / "screening_labeled_dataset.pkl"
UNLABELED_POOL_PKL = PROC_DIR / "screening_unlabeled_pool.pkl"

CLF_METRICS_JSON = OUTPUT_DIR / "screening_funnel_classifier.json"
REG_METRICS_JSON = OUTPUT_DIR / "screening_funnel_regressor.json"
CLF_MODEL_PKL = OUTPUT_DIR / "screening_funnel_classifier.pkl"
REG_MODEL_PKL = OUTPUT_DIR / "screening_funnel_regressor.pkl"

TOPK_LIST = [50, 100, 250, 500, 1000]
ALKALI = {"Li", "Na", "K", "Rb", "Cs"}


def classify_supercon_category(formula: str) -> str:
    """
    Composition rules reverse-engineered from SuperCon's real category
    labels, checked in this order (order matters: 81% of true Bismuthates
    and 18% of true Iron-based ALSO contain Cu+O, so those must be checked
    before the Cuprates rule or they get absorbed into it). Validated at
    99.82% overall accuracy, >=98.8% precision and >=86.7% recall per class
    against the 16,463 positives that carry real ground-truth labels.
    """
    from pymatgen.core import Composition
    try:
        els = set(str(e) for e in Composition(formula).elements)
    except Exception:
        return "Other"

    if len(els) == 1:
        return "Elemental Superconductors"
    if "Bi" in els and "O" in els:
        return "Bismuthates"
    if "Fe" in els:
        return "Iron-based"
    if "B" in els and "C" in els:
        return "Borocarbides"
    if "Cu" in els and "O" in els:
        return "Cuprates"
    if "C" in els and (els & ALKALI):
        return "Organic Superconductors"
    if "H" in els and len(els) <= 3:
        return "Hydrogen-rich Superconductors"
    return "Other"


# ── Step 1: featurize the confirmed-negative formulas not already covered ────

def featurize_confirmed_negatives():
    if NEG_FEAT_CACHE.exists():
        log.info(f"Confirmed-negative features already cached at {NEG_FEAT_CACHE}.")
        return pd.read_pickle(NEG_FEAT_CACHE)

    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    dsc = pd.read_csv(DSC_CSV, skiprows=1, low_memory=False)
    neg = dsc[(dsc.tc == 0) & (dsc.origin_sc == "Supercon")][["formula_sc"]]
    neg = neg.drop_duplicates(subset="formula_sc").rename(columns={"formula_sc": "formula"})
    log.info(f"Confirmed negatives to featurize: {len(neg):,}")

    compositions, valid_idx = [], []
    for i, formula in enumerate(neg["formula"]):
        try:
            comp = Composition(formula)
            if len(comp) == 0:
                continue
            compositions.append(comp)
            valid_idx.append(i)
        except Exception:
            pass

    neg_valid = neg.iloc[valid_idx].copy().reset_index(drop=True)
    neg_valid["composition"] = compositions
    log.info(f"Parsed {len(neg_valid):,}/{len(neg):,} formulas.")

    ep = ElementProperty.from_preset(MAGPIE_PRESET)
    ep.set_n_jobs(1)
    feat_df = ep.featurize_dataframe(neg_valid, col_id="composition", ignore_errors=True, pbar=False)
    feature_cols = ep.feature_labels()
    before = len(feat_df)
    feat_df = feat_df.dropna(subset=feature_cols).reset_index(drop=True)
    log.info(f"Featurized {len(feat_df):,}/{before:,} confirmed negatives.")

    feat_df["Tc"] = 0.0
    feat_df["label"] = 0
    feat_df["source"] = "3dsc_confirmed_negative"
    feat_df["category"] = feat_df["formula"].apply(classify_supercon_category)
    feat_df.to_pickle(NEG_FEAT_CACHE)
    log.info(f"Saved -> {NEG_FEAT_CACHE}")
    return feat_df


# ── Step 2: assemble labeled dataset + unlabeled screening pool ──────────────

def build_datasets(neg_feat_df, feature_cols):
    full_featurized = pd.read_pickle(FEAT_DATASET_PKL)

    pos = full_featurized[full_featurized.label == 1].copy()
    sc_raw = pd.read_csv(SUPERCON_RAW, usecols=["formula", "category"]).drop_duplicates(subset="formula")
    pos = pos.merge(sc_raw, on="formula", how="left")
    pos["category"] = pos["category"].fillna("Other")
    log.info(f"Positives: {len(pos):,}")

    meta_cols = ["formula", "Tc", "label", "source", "composition", "category"]
    keep_cols = meta_cols + feature_cols
    pos = pos[[c for c in keep_cols if c in pos.columns]]
    neg = neg_feat_df[[c for c in keep_cols if c in neg_feat_df.columns]]

    labeled = pd.concat([pos, neg], ignore_index=True)
    labeled = labeled.drop_duplicates(subset="formula", keep="first").reset_index(drop=True)
    log.info(f"Labeled dataset: {len(labeled):,} "
             f"({int((labeled.label==1).sum()):,} positive, {int((labeled.label==0).sum()):,} confirmed negative)")

    # Unlabeled screening pool: original MP negatives, MINUS anything now in
    # the confirmed-negative or positive sets (by formula).
    used_formulas = set(labeled["formula"])
    pool = full_featurized[full_featurized.label == 0].copy()
    pool = pool[~pool["formula"].isin(used_formulas)].reset_index(drop=True)
    log.info(f"Unlabeled screening pool: {len(pool):,} materials (never confirmed either way)")

    labeled.to_pickle(LABELED_DATASET_PKL)
    pool.to_pickle(UNLABELED_POOL_PKL)
    return labeled, pool


# ── Step 3: classifier funnel stage, family-held-out evaluation ──────────────

def enrichment_at_k(y_true, y_score, k):
    order = np.argsort(-y_score)[:k]
    return float(y_true[order].mean())


def fit_score_classifier(X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    clf = XGBClassifier(**XGBOOST_PARAMS)
    clf.fit(X_train, y_train, verbose=False)
    y_prob = clf.predict_proba(X_test)[:, 1]
    res = {
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_test_pos": int(y_test.sum()), "base_rate": float(y_test.mean()),
    }
    if len(np.unique(y_test)) > 1:
        res["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        res["pr_auc"] = float(average_precision_score(y_test, y_prob))
    else:
        res["roc_auc"] = res["pr_auc"] = None
    res["enrichment_at_k"] = {
        str(k): enrichment_at_k(y_test, y_prob, k) for k in TOPK_LIST if k <= len(y_test)
    }
    return res, clf


def evaluate_classifier(labeled, feature_cols):
    from sklearn.model_selection import train_test_split

    X_all = labeled[feature_cols].values
    y_all = labeled["label"].values
    fam_all = labeled["category"].values

    results = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
    )
    log.info("[classifier] random split (headline number, not representative of novel-candidate performance) ...")
    random_res, final_clf = fit_score_classifier(X_train, y_train, X_test, y_test)
    results["random_split"] = random_res
    log.info(json.dumps(random_res, indent=2))

    log.info("[classifier] leave-one-family-out (the number that matters for screening) ...")
    lofo_results = {}
    for fam in sorted(set(fam_all)):
        fam_mask = fam_all == fam
        fam_pos_mask = fam_mask & (y_all == 1)
        if fam_pos_mask.sum() < 10:
            continue

        # Exclude the WHOLE family -- positives AND negatives -- from
        # training. Test on the family's positives + a fixed 20% slice of
        # OTHER families' negatives (shared across folds).
        other_neg_idx = np.where((~fam_mask) & (y_all == 0))[0]
        rng = np.random.RandomState(RANDOM_SEED)
        neg_test_idx = rng.choice(other_neg_idx, size=max(1, len(other_neg_idx) // 5), replace=False)

        test_idx = np.concatenate([np.where(fam_pos_mask)[0], neg_test_idx])
        train_idx = np.setdiff1d(np.arange(len(y_all)), test_idx)
        train_idx = np.setdiff1d(train_idx, np.where(fam_mask)[0])

        res, _ = fit_score_classifier(X_all[train_idx], y_all[train_idx], X_all[test_idx], y_all[test_idx])
        lofo_results[fam] = res
        log.info(f"  {fam}: roc_auc={res['roc_auc']:.4f} pr_auc={res['pr_auc']:.4f} base_rate={res['base_rate']:.4f}")

    results["leave_one_family_out"] = lofo_results

    # Final production model: refit on ALL confirmed labels (no held-out
    # split) -- deployment should use every scrap of confirmed data available.
    from xgboost import XGBClassifier
    final_clf = XGBClassifier(**XGBOOST_PARAMS)
    final_clf.fit(X_all, y_all, verbose=False)

    with open(CLF_MODEL_PKL, "wb") as f:
        pickle.dump(final_clf, f)
    CLF_METRICS_JSON.write_text(json.dumps(results, indent=2))
    log.info(f"Saved classifier -> {CLF_MODEL_PKL}, metrics -> {CLF_METRICS_JSON}")
    return final_clf, results


# ── Step 4: regressor funnel stage (Tc, positives only) ──────────────────────

def fit_score_regressor(X_train, y_train, X_test, y_test, tc_test):
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_log_error, r2_score

    reg = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1,
    )
    reg.fit(X_train, y_train)
    pred_tc = np.clip(np.sinh(reg.predict(X_test)), 0, None)
    msle = float(mean_squared_log_error(tc_test.clip(min=1e-6), pred_tc.clip(min=1e-6)))
    r2 = float(r2_score(y_test, reg.predict(X_test)))
    return {"n_train": int(len(y_train)), "n_test": int(len(y_test)),
            "msle_tc": msle, "r2_arcsinh_tc": r2}, reg


def evaluate_regressor(labeled, feature_cols):
    from sklearn.model_selection import train_test_split

    # Positives only -- Tc is only meaningful for confirmed superconductors,
    # and SuperCon's real category taxonomy applies natively here (no
    # negative-taxonomy mismatch to correct, unlike the classifier stage).
    pos = labeled[labeled.label == 1].copy()
    pos["tc_arcsinh"] = np.arcsinh(pos["Tc"].values)  # T0 = 1K, matches 3DSC's own scaling
    X_all = pos[feature_cols].values
    y_all = pos["tc_arcsinh"].values
    tc_all = pos["Tc"].values
    fam_all = pos["category"].values

    results = {}
    idx_train, idx_test = train_test_split(np.arange(len(y_all)), test_size=TEST_SIZE, random_state=RANDOM_SEED)
    random_res, final_reg = fit_score_regressor(
        X_all[idx_train], y_all[idx_train], X_all[idx_test], y_all[idx_test], tc_all[idx_test]
    )
    results["random_split"] = random_res
    log.info(f"[regressor] random split: {random_res}")

    lofo_results = {}
    for fam in sorted(set(fam_all)):
        fam_mask = fam_all == fam
        if fam_mask.sum() < 15:
            continue
        train_idx = np.where(~fam_mask)[0]
        test_idx = np.where(fam_mask)[0]
        res, _ = fit_score_regressor(X_all[train_idx], y_all[train_idx], X_all[test_idx], y_all[test_idx], tc_all[test_idx])
        lofo_results[fam] = res
        log.info(f"  {fam}: {res}")

    results["leave_one_family_out"] = lofo_results

    from xgboost import XGBRegressor
    final_reg = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1,
    )
    final_reg.fit(X_all, y_all)

    with open(REG_MODEL_PKL, "wb") as f:
        pickle.dump(final_reg, f)
    REG_METRICS_JSON.write_text(json.dumps(results, indent=2))
    log.info(f"Saved regressor -> {REG_MODEL_PKL}, metrics -> {REG_METRICS_JSON}")
    return final_reg


def main():
    if not FEAT_DATASET_PKL.exists():
        log.error(f"Featurized dataset not found at {FEAT_DATASET_PKL}.")
        log.error("Run first:  python data/preprocess.py  &&  python path_a_classical/featurize.py")
        sys.exit(1)

    feature_cols = SELECTED_FEATURES_FILE.read_text().splitlines()
    neg_feat_df = featurize_confirmed_negatives()
    labeled, pool = build_datasets(neg_feat_df, feature_cols)

    log.info("=" * 60)
    log.info("STAGE 1: CLASSIFIER (is this composition SC-like at all?)")
    log.info("=" * 60)
    evaluate_classifier(labeled, feature_cols)

    log.info("=" * 60)
    log.info("STAGE 2: REGRESSOR (predicted Tc, ranks survivors)")
    log.info("=" * 60)
    evaluate_regressor(labeled, feature_cols)


if __name__ == "__main__":
    main()
