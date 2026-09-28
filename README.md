# Superconductor Screening Funnel

A two-stage composition-based funnel for shortlisting candidate superconductors,
built to minimize false negatives at a given screening budget rather than to
predict Tc precisely for any single material.

This is the **ablated, production version** of a larger investigation. Everything
here is the part that works and is used by the final shortlist. The larger
investigation (a structure-based GNN path, SHAP/GNNExplainer interrogation,
dataset-bug forensics) is not part of this repository; its method and results
are summarised in [`PROJECT_REPORT.md`](PROJECT_REPORT.md) (§5–7), and the code
here is deliberately just the part that ships.

## What it does

1. **Stage 1 (classifier)** — XGBoost, predicts whether a composition is
   superconductor-like at all.
2. **Stage 2 (regressor)** — XGBoost, predicts Tc (arcsinh-scaled) for
   whatever survives Stage 1, to rank the shortlist.
3. Both stages train on **confirmed labels only**: SuperCon positives
   (Tc > 0) and SuperCon-origin confirmed negatives (Tc = 0, sourced from
   the 3DSC dataset, since SuperCon's own raw file drops most of these).
   The ~62,000 "MP-stable-and-not-in-SuperCon" materials are *not* used as
   training negatives — most have simply never been tested — they are the
   actual screening pool, scored rather than assumed negative.
4. Both stages are evaluated with **leave-one-family-out** splits (not a
   random split), because that's what predicts performance on a genuinely
   novel candidate. Family assignment uses SuperCon's own category
   taxonomy, validated at 99.82% accuracy.

## Honest performance (leave-one-family-out ROC-AUC)

| Family | ROC-AUC | Notes |
|---|---|---|
| Cuprates | 0.96 | |
| Bismuthates | 0.96 | |
| Other | 0.93 | |
| Elemental | 0.93 | |
| Organic | 0.91 | only 75 positives, 0 confirmed negatives — noisy |
| Iron-based | 0.66 | usable for triage, wide error bars |
| Borocarbides | 0.64 | SuperCon's own label mixes true intermetallic borocarbides with B/C-doped cuprates — genuinely heterogeneous, not a bug |

A random 90/10 split gives ROC-AUC ~0.96 — close to the family-held-out
number for the well-behaved families, but that agreement is family-specific,
not general; always read the LOFO table before trusting a number for a
family not shown above (e.g. Hydrogen-rich has only 6 known examples,
too few to evaluate honestly at all).

Sanity check: the trained funnel recovers **H3S** and **Ba2YCu3O7 (YBCO)**
— two of the most famous known superconductors — from the "unlabeled" pool
without being told about them, ranked in the top 20 by funnel score.

## Usage

**First-time setup.** Large intermediates (`data/processed/*.pkl`) are not
committed; they are rebuilt from the raw CSVs in `data/raw/` (no API key
needed for this path):

```bash
pip install -r requirements.txt
python data/preprocess.py                       # merge SuperCon + MP negatives -> full_dataset.pkl
python path_a_classical/featurize.py            # Magpie features for ~78k formulas (~1-2 min)
python path_a_classical/feature_selection.py    # optional -- see note
```

Skip `feature_selection.py` to reproduce the shipped results: the committed
`data/processed/selected_features.txt` is the feature list the models were
trained with. Re-running it is not bit-reproducible (a re-run overlapped 38 of
the 40 committed features), so it would slightly change every downstream number.
Featurization itself is exact (verified: max abs difference 0 on a 300-row check).

`data/download_data.py` (needs `MP_API_KEY`) only re-fetches
`mp_stable_non_sc.csv`, which is already committed.

**Then run the funnel:**

```bash
python run_pipeline.py
```

Or run the three steps individually:

```bash
python path_a_classical/build_screening_pipeline.py   # train + evaluate
python path_a_classical/run_screening.py               # score the pool
python path_a_classical/annotate_shortlist.py           # flag + contextualize
```

## Explainability

`path_a_classical/explain_funnel.py` explains the shipped classifier and
regressor with SHAP and stress-tests the original project's headline "design
rule" (low mean covalent radius => superconductor):

```bash
python path_a_classical/explain_funnel.py
```

Outputs: `outputs/explain_funnel.json`, `explain_funnel_{classifier,regressor}_shap.csv`,
`explain_funnel.png`. Findings on the shipped model:

- The funnel relies mostly on **chemical contrast** -- range of electronegativity
  is the top feature of both stages, followed by the range of Mendeleev number,
  atomic number and melting temperature.
- Mean covalent radius is only **rank 6** (not the top feature), is **not**
  oxide-specific on this model, and a *higher* value pushes toward "superconductor" --
  the opposite sign to the original BCS-stiffness explanation.
- These are associations with a label that separates superconductors from
  literature-tested near-misses, not evidence of a mechanism. SHAP is computed on
  the data the models were fit on, so it describes what they use, not how well
  they generalize (see the leave-one-family-out table above for that).

Full discussion, including the earlier classifier where the feature *did* look
oxide-scoped: [`PROJECT_REPORT.md`](PROJECT_REPORT.md) section 7.

## Experiment tracking

Scripts that train, evaluate or explain a model log to Weights & Biases
(project `xai-superconductivity-screening`) through `tracking.py`: config,
feature list, seeds, data/model hashes and code version, plus metrics, tables,
curves and plots -- never raw data. `explain_funnel.py` is tracked today.

Run offline with `SCREENING_WANDB=0`. If W&B is enabled but unreachable or
unauthenticated the script stops instead of running untracked.

Known gap: `build_screening_pipeline.py`, `run_screening.py` and
`annotate_shortlist.py` are not yet tracked, and the shipped models were produced
by an untracked run (their md5s are recorded in the explain run's config).

## Key outputs

- `outputs/screening_funnel_classifier.json` / `regressor.json` — the
  honest leave-one-family-out numbers, per family.
- `outputs/candidate_shortlist.csv` — the full ~62,000-candidate pool,
  ranked by funnel score (P(superconductor) x predicted Tc).
- `outputs/candidate_shortlist_top500_annotated.csv` — **the deliverable**:
  top 500 candidates, each tagged with:
  - `inferred_family` / `family_lofo_roc_auc` — which family it resembles
    and how much to trust a ranking in that family
  - `already_known_supercon` / `same_element_set_as_known_SC` — whether
    it's a rediscovery under a different formula string, not a novel
    candidate (about a quarter of the raw top-500 are)

## Reading the shortlist responsibly

This is a triage tool, not a discovery claim. A high funnel score means
"worth a closer, cheaper follow-up look" (DFT stability check, then
electron-phonon calculation, then synthesis attempt) — not "this is a
superconductor." Filter out `already_known_supercon == True` rows before
treating anything as novel, and weight confidence by `family_lofo_roc_auc`,
not by rank alone: a rank-40 Cuprate-family candidate (reliability 0.96) is
more trustworthy than a rank-5 Iron-based one (reliability 0.66).
