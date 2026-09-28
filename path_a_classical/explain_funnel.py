"""
path_a_classical/explain_funnel.py
------------------------------------
SHAP explanations for the shipped screening funnel, plus a confound check on
its most-cited feature.

The original project's headline design rule was "low mean covalent radius
predicts superconductivity, via BCS phonon stiffness". That was derived from
an earlier classifier trained on unlabeled-as-negative labels, and it was
never re-tested on the confirmed-label funnel that ships here. This script
does that, and checks whether the feature is doing general work or is scoped
to oxide chemistry (oxygen has a very small covalent radius, and cuprates,
which are not BCS superconductors, dominate the positives).

Four views of "does this feature matter outside oxides?":
  1. shipped model, SHAP restricted to oxygen-free / oxide-only rows
     (no retraining -- explains exactly the model that produces the shortlist)
  2. models retrained on the full / oxygen-free / oxide-only subsets of the
     confirmed-label data
  3. the same retrains on the ORIGINAL labels (unlabeled-as-negative), so the
     original study's oxide-scoped result is reproducible from this repo
  4. how well oxygen fraction alone separates the classes

Caveats: SHAP for the shipped models is computed on the data they were fit
on (global importance, not a generalization claim), and the shipped models
come from an untracked run -- this run explains them, it did not produce them.

Outputs
-------
outputs/explain_funnel.json
outputs/explain_funnel_classifier_shap.csv / explain_funnel_regressor_shap.csv
outputs/explain_funnel.png
"""

import sys
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEAT_DATASET_PKL, PROC_DIR, OUTPUT_DIR, RANDOM_SEED, TEST_SIZE, XGBOOST_PARAMS
from tracking import start_run, file_md5, record_provenance, line_chart

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SELECTED_FEATURES_FILE = PROC_DIR / "selected_features.txt"
LABELED_DATASET_PKL = PROC_DIR / "screening_labeled_dataset.pkl"
CLF_MODEL_PKL = OUTPUT_DIR / "screening_funnel_classifier.pkl"
REG_MODEL_PKL = OUTPUT_DIR / "screening_funnel_regressor.pkl"

OUT_JSON = OUTPUT_DIR / "explain_funnel.json"
OUT_CLF_CSV = OUTPUT_DIR / "explain_funnel_classifier_shap.csv"
OUT_REG_CSV = OUTPUT_DIR / "explain_funnel_regressor_shap.csv"
OUT_PLOT = OUTPUT_DIR / "explain_funnel.png"

TARGET_FEATURE = "MagpieData mean CovalentRadius"
TOP_N = 15
GROUP_COLORS = {"original labels": "#dd8452", "retrained": "#55a868", "shipped": "#4c72b0"}


def oxygen_fraction(formula: str) -> float:
    from pymatgen.core import Composition
    try:
        return float(Composition(formula).get_atomic_fraction("O"))
    except Exception:
        return np.nan


def shap_values(model, X):
    import shap
    values = shap.TreeExplainer(model).shap_values(X)
    return values[1] if isinstance(values, list) else values


def feature_stats(shap_vals, X, feature_cols) -> pd.DataFrame:
    """Per feature: mean |SHAP|, and corr(value, SHAP) -- positive means a
    HIGH feature value pushes the prediction up."""
    corr = [
        np.corrcoef(X[:, j], shap_vals[:, j])[0, 1]
        if X[:, j].std() > 0 and shap_vals[:, j].std() > 0 else np.nan
        for j in range(X.shape[1])
    ]
    stats = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        "value_shap_corr": corr,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    stats.insert(0, "rank", stats.index + 1)
    return stats


def target_row(stats: pd.DataFrame) -> dict:
    row = stats[stats.feature == TARGET_FEATURE].iloc[0]
    return {
        "rank": int(row["rank"]),
        "mean_abs_shap": float(row["mean_abs_shap"]),
        "value_shap_corr": float(row["value_shap_corr"]),
    }


def retrain_and_explain(X, y, feature_cols):
    """Fit a fresh classifier on one subset; SHAP on its held-out split."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    clf = XGBClassifier(**XGBOOST_PARAMS)
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    stats = feature_stats(shap_values(clf, X_te), X_te, feature_cols)
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])),
        "target": target_row(stats),
        "top_features": stats.head(10)["feature"].tolist(),
    }, clf.evals_result()["validation_0"]["logloss"]


def plot_summary(clf_stats, reg_stats, regimes):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, stats, title in [
        (axes[0], clf_stats, "Shipped classifier: top features"),
        (axes[1], reg_stats, "Shipped regressor: top features"),
    ]:
        top = stats.head(TOP_N)[::-1]
        colors = ["#c44e52" if f == TARGET_FEATURE else "#4c72b0" for f in top.feature]
        ax.barh([f.replace("MagpieData ", "") for f in top.feature], top.mean_abs_shap, color=colors)
        ax.set_xlabel("mean |SHAP|")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

    # Rank (not mean |SHAP|) is compared across models: |SHAP| scales differ between
    # models, so only the rank is meaningful across the regimes.
    names = list(regimes)[::-1]
    ranks = [regimes[n]["rank"] for n in names]
    ax = axes[2]
    ax.barh(range(len(names)), ranks, color=[GROUP_COLORS[n.split(" | ")[0]] for n in names])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace(" | ", ": ") for n in names], fontsize=8)
    for i, r in enumerate(ranks):
        ax.annotate(f"#{r}", (r, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("rank of mean covalent radius (1 = most important)")
    ax.set_title("Where mean covalent radius ranks, by model and subset", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)


def main():
    import wandb
    from sklearn.metrics import roc_auc_score

    feature_cols = SELECTED_FEATURES_FILE.read_text().splitlines()
    labeled = pd.read_pickle(LABELED_DATASET_PKL)
    with open(CLF_MODEL_PKL, "rb") as f:
        clf = pickle.load(f)
    with open(REG_MODEL_PKL, "rb") as f:
        reg = pickle.load(f)

    labeled["o_frac"] = labeled["formula"].apply(oxygen_fraction)
    labeled = labeled.dropna(subset=["o_frac"]).reset_index(drop=True)
    has_o = (labeled["o_frac"] > 0).values
    X = labeled[feature_cols].values
    y = labeled["label"].values

    config = {
        "seed": RANDOM_SEED, "test_size": TEST_SIZE, "xgboost_params": XGBOOST_PARAMS,
        "features": feature_cols, "target_feature": TARGET_FEATURE,
        "n_labeled": len(labeled), "n_positive": int(y.sum()), "n_confirmed_negative": int((y == 0).sum()),
        "data_md5": file_md5(LABELED_DATASET_PKL),
        "classifier_md5": file_md5(CLF_MODEL_PKL), "regressor_md5": file_md5(REG_MODEL_PKL),
    }
    notes = ("SHAP explanation of the shipped funnel models + oxygen-confound check. "
             "The shipped models were produced by an untracked run (md5s in config); "
             "this run explains them, it did not train them.")

    with start_run("explain-funnel", "explain", config, tags=["shap", "oxygen-confound"], notes=notes) as run:
        # 1. Shipped classifier and regressor: global SHAP
        log.info("SHAP: shipped classifier ...")
        clf_shap = shap_values(clf, X)
        clf_stats = feature_stats(clf_shap, X, feature_cols)

        log.info("SHAP: shipped regressor (confirmed positives) ...")
        pos = labeled["label"].values == 1
        reg_shap = shap_values(reg, X[pos])
        reg_stats = feature_stats(reg_shap, X[pos], feature_cols)

        clf_stats.to_csv(OUT_CLF_CSV, index=False)
        reg_stats.to_csv(OUT_REG_CSV, index=False)

        # 2. Same shipped model, SHAP sliced by chemistry (no retraining)
        regimes = {
            "shipped | all": target_row(clf_stats),
            "shipped | oxygen-free": target_row(feature_stats(clf_shap[~has_o], X[~has_o], feature_cols)),
            "shipped | oxide-only": target_row(feature_stats(clf_shap[has_o], X[has_o], feature_cols)),
        }

        # 3. Models retrained on each subset
        retrained, curves = {}, {}
        for tag, mask in [("full", np.ones(len(y), bool)), ("oxygen-free", ~has_o), ("oxide-only", has_o)]:
            log.info(f"Retraining on '{tag}' subset ({int(mask.sum()):,} rows) ...")
            retrained[tag], curves[tag] = retrain_and_explain(X[mask], y[mask], feature_cols)
            regimes[f"retrained | {tag}"] = retrained[tag]["target"]
            curves[f"funnel/{tag}"] = curves.pop(tag)

        original = pd.read_pickle(FEAT_DATASET_PKL)
        original["o_frac"] = original["formula"].apply(oxygen_fraction)
        original = original.dropna(subset=["o_frac"]).reset_index(drop=True)
        orig_has_o = (original["o_frac"] > 0).values
        X_orig, y_orig = original[feature_cols].values, original["label"].values
        original_retrained = {}
        for tag, mask in [("full", np.ones(len(y_orig), bool)), ("oxygen-free", ~orig_has_o), ("oxide-only", orig_has_o)]:
            log.info(f"Retraining on ORIGINAL labels, '{tag}' subset ({int(mask.sum()):,} rows) ...")
            original_retrained[tag], curves[f"original/{tag}"] = retrain_and_explain(X_orig[mask], y_orig[mask], feature_cols)
            regimes[f"original labels | {tag}"] = original_retrained[tag]["target"]

        # 4. Oxygen alone, and the feature's overlap with oxygen
        oxygen_only_auc = float(roc_auc_score(y, labeled["o_frac"]))
        corr_with_o = float(labeled[TARGET_FEATURE].corr(labeled["o_frac"]))

        results = {
            "target_feature": TARGET_FEATURE,
            "share_with_oxygen": {
                "positives": float(has_o[y == 1].mean()), "confirmed_negatives": float(has_o[y == 0].mean()),
            },
            "oxygen_fraction_alone_roc_auc": oxygen_only_auc,
            "corr_feature_vs_oxygen_fraction": corr_with_o,
            "shipped_classifier_top10": clf_stats.head(10)["feature"].tolist(),
            "shipped_regressor_top10": reg_stats.head(10)["feature"].tolist(),
            "target_by_regime": regimes,
            "retrained_subsets": retrained,
            "original_label_subsets": original_retrained,
        }
        OUT_JSON.write_text(json.dumps(results, indent=2))
        plot_summary(clf_stats, reg_stats, regimes)

        # W&B: metrics, tables, curves, plot
        for name, r in regimes.items():
            key = name.replace(" | ", "/").replace(" ", "_")
            run.summary[f"target/{key}/rank"] = r["rank"]
            run.summary[f"target/{key}/mean_abs_shap"] = r["mean_abs_shap"]
            run.summary[f"target/{key}/value_shap_corr"] = r["value_shap_corr"]
        for tag, r in retrained.items():
            run.summary[f"retrained/{tag}/roc_auc"] = r["roc_auc"]
        for tag, r in original_retrained.items():
            run.summary[f"original_labels/{tag}/roc_auc"] = r["roc_auc"]
        run.summary["oxygen_fraction_alone_roc_auc"] = oxygen_only_auc
        run.summary["corr_feature_vs_oxygen_fraction"] = corr_with_o
        run.log({
            "shap/classifier_features": wandb.Table(dataframe=clf_stats),
            "shap/regressor_features": wandb.Table(dataframe=reg_stats),
            "val_logloss_curves": line_chart("Held-out logloss per boosting round", curves),
            "explain_funnel_plot": wandb.Image(str(OUT_PLOT)),
        })
        record_provenance("explain_funnel", run, [OUT_JSON, OUT_CLF_CSV, OUT_REG_CSV, OUT_PLOT])

        log.info(json.dumps({"target_by_regime": regimes,
                             "oxygen_fraction_alone_roc_auc": oxygen_only_auc,
                             "corr_feature_vs_oxygen_fraction": corr_with_o}, indent=2))
        log.info(f"Saved -> {OUT_JSON}, {OUT_PLOT}")


if __name__ == "__main__":
    main()
