"""
path_a_classical/reproduce_original_study.py
-----------------------------------------------
Reproduces the composition-based pipeline of the original study ("Theoretical
Modeling of Superconductivity with Explainable AI", Final Report) as designed:

  data      : ~78.5k materials -- SuperCon positives, and MP-stable materials
              not reported in SuperCon as negatives (Tc = 0)
  features  : 40 Magpie features (Pearson pruning at |r| > 0.95, then RFE)
  model     : XGBoost, 500 trees, depth 6, lr 0.05
  split     : stratified 80 / 10 / 10 train / validation / test
  outputs   : Table I (classification report), ROC-AUC, SHAP importance

This is the ORIGINAL methodology, kept so that half of the project stays
reproducible next to the corrected screening funnel (build_screening_pipeline.py).
It answers "can composition alone tell a literature superconductor from an
arbitrary stable material, on a random split?" -- not "does it work on a
chemical family it has never seen?" (that is what the funnel's leave-one-family-
out evaluation measures).

Differences from the printed report, to be read alongside the comparison this
script writes: the dataset here has 78,482 rows (the report's test-set support
implies ~78.5k as well, but not the identical file), and selected_features.txt
comes from a later re-run of the same RFE procedure, so it does not match the
report's Figure 1 feature set exactly (see `report_fig1_features_in_selected_40`).

Outputs
-------
outputs/original_study.json
outputs/original_study_shap.csv
outputs/original_study_shap.png
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEAT_DATASET_PKL, PROC_DIR, OUTPUT_DIR, TEST_SIZE, RANDOM_SEED, XGBOOST_PARAMS
from tracking import start_run, file_md5, record_provenance, line_chart

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SELECTED_FEATURES_FILE = PROC_DIR / "selected_features.txt"
OUT_JSON = OUTPUT_DIR / "original_study.json"
OUT_SHAP_CSV = OUTPUT_DIR / "original_study_shap.csv"
OUT_PLOT = OUTPUT_DIR / "original_study_shap.png"

# Numbers printed in the Final Report (Table I and abstract)
REPORTED = {
    "accuracy": 0.9525, "roc_auc": 0.9886,
    "Non-SC": {"precision": 0.98, "recall": 0.96, "f1": 0.97, "support": 6202},
    "Supercon": {"precision": 0.85, "recall": 0.93, "f1": 0.89, "support": 1652},
}

# Figure 1 of the Final Report: its top-20 features, mapped to Magpie labels
REPORT_FIG1 = {
    "Mean Covalent Radius": "MagpieData mean CovalentRadius",
    "Mean Electronegativity": "MagpieData mean Electronegativity",
    "Electronegativity Range": "MagpieData range Electronegativity",
    "Mean d-Valence Electrons": "MagpieData mean NdValence",
    "Mean p-Valence Electrons": "MagpieData mean NpValence",
    "Mean Unfilled f-Orbitals": "MagpieData mean NfUnfilled",
    "Electronegativity Deviation": "MagpieData avg_dev Electronegativity",
    "Periodic Group Deviation": "MagpieData avg_dev Column",
    "Mean Melting Temperature": "MagpieData mean MeltingT",
    "Maximum Electronegativity": "MagpieData maximum Electronegativity",
    "Mode Melting Temperature": "MagpieData mode MeltingT",
    "Maximum Ground-State Volume": "MagpieData maximum GSvolume_pa",
    "Magnetic Moment Deviation": "MagpieData avg_dev GSmagmom",
    "p-Valence Electron Deviation": "MagpieData avg_dev NpValence",
    "Melting Temperature Range": "MagpieData range MeltingT",
    "Mean Unfilled Orbitals": "MagpieData mean NUnfilled",
    "s-Valence Electron Deviation": "MagpieData avg_dev NsValence",
    "Mendeleev Number Range": "MagpieData range MendeleevNumber",
    "Band Gap Deviation": "MagpieData avg_dev GSbandgap",
    "Total Valence Electron Deviation": "MagpieData avg_dev NValence",
}
TOP_N = 20


def split_80_10_10(X, y):
    """The original split: stratified 10% test, then 10% of the total as validation."""
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    val_fraction = TEST_SIZE / (1 - TEST_SIZE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_fraction, random_state=RANDOM_SEED, stratify=y_tr
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def table_one(y_true, y_pred, y_prob):
    from sklearn.metrics import classification_report, roc_auc_score
    rep = classification_report(y_true, y_pred, target_names=["Non-SC", "Supercon"], output_dict=True)
    return {
        "accuracy": float(rep["accuracy"]),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        **{cls: {k: float(rep[cls][k]) for k in ("precision", "recall", "f1-score", "support")}
           for cls in ("Non-SC", "Supercon")},
    }


def shap_importance(model, X_test, feature_cols) -> pd.DataFrame:
    import shap
    values = shap.TreeExplainer(model).shap_values(X_test)
    values = values[1] if isinstance(values, list) else values
    corr = [np.corrcoef(X_test[:, j], values[:, j])[0, 1] if X_test[:, j].std() > 0 else np.nan
            for j in range(X_test.shape[1])]
    df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": np.abs(values).mean(axis=0),
                       "value_shap_corr": corr}).sort_values("mean_abs_shap", ascending=False)
    df = df.reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def plot_top(shap_df):
    top = shap_df.head(TOP_N)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([f.replace("MagpieData ", "") for f in top.feature], top.mean_abs_shap, color="#4c72b0")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Original methodology (random split, unlabeled-as-negative):\ntop-20 SHAP features")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)


def main():
    from xgboost import XGBClassifier

    if not FEAT_DATASET_PKL.exists():
        log.error(f"Featurized dataset not found at {FEAT_DATASET_PKL}. See README setup.")
        sys.exit(1)

    feature_cols = SELECTED_FEATURES_FILE.read_text().splitlines()
    df = pd.read_pickle(FEAT_DATASET_PKL)
    X, y = df[feature_cols].values, df["label"].values
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_80_10_10(X, y)
    log.info(f"train {len(y_tr):,} / val {len(y_val):,} / test {len(y_te):,}  (positive rate {y.mean():.3f})")

    config = {
        "seed": RANDOM_SEED, "test_size": TEST_SIZE, "xgboost_params": XGBOOST_PARAMS,
        "features": feature_cols, "n_rows": len(df), "n_positive": int(y.sum()),
        "split": {"train": len(y_tr), "val": len(y_val), "test": len(y_te)},
        "data_md5": file_md5(FEAT_DATASET_PKL), "reported_in_final_report": REPORTED,
    }
    notes = ("Reproduction of the ORIGINAL study's composition pipeline (random split, "
             "unlabeled-as-negative labels). Not the shipped funnel.")

    with start_run("original-study", "train", config, tags=["original-study", "baseline"], notes=notes) as run:
        import wandb
        clf = XGBClassifier(**XGBOOST_PARAMS)
        clf.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)

        y_prob = clf.predict_proba(X_te)[:, 1]
        table = table_one(y_te, (y_prob >= 0.5).astype(int), y_prob)
        log.info(json.dumps(table, indent=2))

        shap_df = shap_importance(clf, X_te, feature_cols)
        shap_df.to_csv(OUT_SHAP_CSV, index=False)
        plot_top(shap_df)

        reproduced_top = set(shap_df.head(TOP_N).feature)
        fig1 = set(REPORT_FIG1.values())
        results = {
            "table_one_reproduced": table,
            "table_one_reported": REPORTED,
            "split": config["split"],
            "shap_top10": shap_df.head(10)["feature"].tolist(),
            "report_fig1_features_in_selected_40": sorted(fig1 & set(feature_cols)),
            "report_fig1_features_missing_from_selected_40": sorted(fig1 - set(feature_cols)),
            "report_fig1_overlap_with_reproduced_top20": sorted(fig1 & reproduced_top),
        }
        OUT_JSON.write_text(json.dumps(results, indent=2))

        ev = clf.evals_result()
        run.log({
            "logloss_curves": line_chart("Original methodology: logloss",
                                         {"train": ev["validation_0"]["logloss"], "validation": ev["validation_1"]["logloss"]}),
            "shap_features": wandb.Table(dataframe=shap_df),
            "shap_plot": wandb.Image(str(OUT_PLOT)),
        })
        run.summary["test/accuracy"] = table["accuracy"]
        run.summary["test/roc_auc"] = table["roc_auc"]
        for cls in ("Non-SC", "Supercon"):
            for k in ("precision", "recall", "f1-score"):
                run.summary[f"test/{cls}/{k}"] = table[cls][k]
        run.summary["fig1_features_in_selected_40"] = len(results["report_fig1_features_in_selected_40"])
        run.summary["fig1_overlap_with_reproduced_top20"] = len(results["report_fig1_overlap_with_reproduced_top20"])

        record_provenance("reproduce_original_study", run, [OUT_JSON, OUT_SHAP_CSV, OUT_PLOT])
        log.info(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
