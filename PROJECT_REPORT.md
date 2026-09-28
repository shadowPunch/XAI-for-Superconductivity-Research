# Superconductor Candidate Screening: Methodology, Findings, and an Abridged Structural-GNN Investigation

**Author:** Nithish Ravikkumar, IIT Roorkee
**Scope:** Revision and extension of the original "Theoretical Modeling of Superconductivity with Explainable AI" project (PHC-391)

---

## Abstract

The original project (PHC-391) built a two-path study of superconductivity from the SuperCon database and the Materials Project: a composition-based XGBoost classifier on Magpie descriptors (ROC-AUC 0.9886 on a random split), interpreted with SHAP, and a structure-based Graph Neural Network (ROC-AUC 0.9412), interpreted with GNNExplainer. The composition pipeline was carried out as designed and is reproduced in this repository (ROC-AUC 0.991 on the same methodology, §1.3). This report keeps that study and extends it in two directions. First, it asks what such a model is useful for — shortlisting candidates — which needs a different test from a random split: performance on a chemical family the model has never seen. Under leave-one-family-out validation, and with confirmed non-superconductors replacing "not in the literature" as the negative class, composition-based screening remains genuinely useful (ROC-AUC 0.64–0.96 depending on chemistry) and is the basis of the final deliverable: a two-stage funnel that classifies candidates and ranks them by predicted critical temperature, applied to ~62,000 unlabeled candidate materials to produce a reliability-annotated shortlist. SHAP on the shipped funnel identifies chemical contrast (electronegativity range) as its dominant signal; the original report's covalent-radius design rule does not hold up — it is not oxide-specific on the corrected model, and its sign is opposite to the claimed one. Second, it re-examines the GNN path, which does not survive the same scrutiny: its interpretability tooling was broken, its data pipeline silently dropped most doped structures, and once both were fixed, leave-one-family-out performance on cuprates and several other families is at or below chance. Its explanations, aggregated across 40 real cuprate superconductors rather than a single anecdote, show a chemically shallow shortcut — reading raw dopant-site occupancy — rather than the Cu–O-plane physics it was reported to have learned. The full GNN investigation is summarized here; the repository contains only the effective, deployed screening pipeline and the reproduction of the original composition study.

---

## 1. Background and the Original Study

### 1.1 Motivation

Discovering new superconductors has historically relied on chemical intuition and trial and error. BCS theory gave a roadmap for conventional, low-temperature superconductors, but the discovery of the high-temperature cuprates (Bednorz & Müller, 1986) and later the pnictides was largely unexpected, which makes the field a natural candidate for data-driven acceleration. The original project set out to build an explainable-AI system that predicts which materials are likely superconductors *and explains why*, to cut the time physicists spend verifying candidates and to yield human-interpretable design rules. It did so along two paths: a fast composition-based classifier (Path A) and a higher-fidelity structure-based Graph Neural Network (Path B), each paired with an explainability method (SHAP and GNNExplainer respectively).

### 1.2 Background

The study builds on materials informatics (Ramprasad et al., 2017) and on two data sources: the SuperCon database, a comprehensive but unstructured list of experimentally reported superconductors, and the Materials Project, a large computational database of material properties (Jain et al., 2013). Early machine-learning work focused on chemical composition because it is the most widely available data: Stanev et al. (2018) used classical models on Magpie descriptors — numerical statistics of elemental properties — to predict critical temperature from composition alone. Magpie features are generated with the matminer library (Dunn et al., 2020). More recent work uses 3D crystal structure, which carries more physical information; the Crystal Graph Convolutional Neural Network (Xie & Grossman, 2018) showed that GNNs can learn structure–property relationships directly from atomic structure.

### 1.3 The original study, as designed (pre-GNN)

**Data curation.** Positives were SuperCon entries, keeping the highest reported Tc for each unique formula (16,531 unique superconductors in the original report). Negatives were all computationally stable Materials Project materials (energy above hull between 0 and 0.05 eV/atom) that were not reported as superconductors, assigned Tc = 0 — about 62,000 materials, for a unified dataset of ~78,500. (The SuperCon file shipped here yields 16,463 positives and 62,019 negatives, 78,482 in total; the small difference from the report's 16,531 is a property of the file version, not a processing change.)

**Featurization and selection.** Each formula was converted to the 132-feature Magpie preset (matminer `ElementProperty`: statistics of elemental properties such as mean electronegativity or range of covalent radii). Selection was two-step: features with pairwise Pearson correlation above 0.95 were removed, then Recursive Feature Elimination with a LightGBM estimator kept the 40 most predictive.

**Model and split.** An XGBoost classifier (500 trees, depth 6, learning rate 0.05) trained on a stratified 80/10/10 train/validation/test split.

**Results.** `path_a_classical/reproduce_original_study.py` re-runs exactly this pipeline (W&B run `h466vile`), with these results on the held-out 10% test set:

| | Final Report | Reproduced here |
|---|---|---|
| ROC-AUC | 0.9886 | 0.991 |
| Accuracy | 0.9525 | 0.964 |
| Non-superconductor precision / recall / F1 | 0.98 / 0.96 / 0.97 | 0.97 / 0.98 / 0.98 |
| Superconductor precision / recall / F1 | 0.85 / 0.93 / 0.89 | 0.92 / 0.90 / 0.91 |
| Test support (non-SC / SC) | 6,202 / 1,652 | 6,203 / 1,646 |

The headline result reproduces closely (ROC-AUC within 0.003; test-set support within one row for the non-superconductors). The precision/recall balance for the superconductor class is shifted relative to the report (0.85/0.93 versus 0.92/0.90), and I have not isolated why; candidate causes are the six-material difference in the positive set and the fact that `selected_features.txt` comes from a later re-run of the same RFE procedure (18 of the report's 20 Figure 1 features are in it; maximum electronegativity and band-gap deviation are not).

**Interpretation as reported.** SHAP on this model gave three design rules: (1) *low mean covalent radius* was the top predictor, read as smaller atoms → stiffer lattices → higher phonon frequencies, consistent with BCS theory; (2) a *high range of electronegativity* was the next strongest, read as superconductors being chemically contrasting compounds rather than simple alloys; (3) *low mean d-valence electron count* was strongly predictive, read as high d-electron counts being associated with magnetism. The reproduction agrees on the broad picture: 13 of the report's 20 Figure 1 features appear in the reproduced top 20 (13–14 across repeated runs), with range of electronegativity, mean covalent radius, mean melting temperature, mean electronegativity and mean f-orbital vacancy among the top five. How well each *rule* holds up under further testing is the subject of §7.

**Original future work.** The report proposed (a) screening millions of hypothetical materials in OQMD with the XGBoost model, (b) a hybrid pipeline in which XGBoost filters formulas and a GNN then analyzes the 3D structure of the shortlist, and (c) pre-training the GNN on formation energy before fine-tuning. This revision realizes (a) in spirit by screening the ~62,000-material Materials Project pool (§4.4; OQMD was not used), tests (c) partially with frozen CHGNet embeddings (§5.2), and finds no support for the GNN stage in (b) (§5).

*Reproducibility notes.* A fresh rebuild from the raw CSVs featurizes four noble-gas formulas (Ar, He, HeSiO₂, Ne) that the original featurization run had dropped, giving 78,482 rows rather than 78,478; this adds four negatives to the screening pool and does not change the shortlist or any funnel metric. Models refit on this dataset use multithreaded XGBoost and are not bit-reproducible: ROC-AUC and accuracy vary in the third decimal between runs, and lower-ranked SHAP positions can swap between near-tied features. Quantities quoted from such refits below carry that tolerance; the shipped funnel models and their outputs are reproduced exactly.

### 1.4 What this revision adds

The original evaluation — a random split, with unlabeled materials treated as negatives — is standard practice and answers the question it was designed to answer: can composition alone tell a literature superconductor from an arbitrary stable material? It does not measure the property a *screening* tool needs, which is performance on a chemical family the model has not seen; and "not reported as a superconductor" is not the same as "tested and found not to be one". This revision therefore (i) evaluates by leave-one-family-out with confirmed negatives (§3–4), (ii) re-tests the explanations against that standard (§7), and (iii) subjects the GNN path to the same standard (§5), where it fails for reasons that include implementation defects. The independent 3DSC dataset paper (Sommer et al., 2023) reports the same structure-does-not-help conclusion using a similar grouped-evaluation methodology.

## 2. Data

**Positive examples (16,463):** SuperCon entries with Tc > 0. 85% of these have fractional/doped stoichiometry (e.g. `La1.85Sr0.15CuO4`) rather than integer formulas — a fact that turns out to be central to several of the problems described below.

**Negative examples — two designs, one unsuited to screening:**
- *Original design (§1.3):* "Materials Project structures that are computationally stable (energy above hull ≤ 0.05 eV/atom) and not reported in SuperCon" (~62,000 materials). This is a positive-unlabeled (PU) design, not a binary one — absence from the superconductivity literature does not mean a material was tested and found non-superconducting; most have simply never been tested. Quantified consequence: 1,169 of these 62,019 "negatives" (1.9%) are composition-near-duplicates of real superconductors, including elemental aluminium — a textbook BCS superconductor — labeled negative.
- *Corrected design:* 1,776 confirmed non-superconductors, sourced via the 3DSC dataset (Sommer, Willa, Schmalian & Friederich, *Scientific Data* 2023) from Stanev et al.'s (2018) curated SuperCon extract, which explicitly retains tested-and-not-superconducting compounds (our own raw SuperCon file retains only 58 such rows and drops them during preprocessing). 1,723 of these 1,776 had never been featurized in this project at all, because they were missed by exact-formula matching against MP structures — the same doped-formula matching failure described next.

**Unlabeled screening pool (61,978):** the remainder of the original "negative" pool once confirmed negatives are removed. This is not training data — it is the actual target of the screening funnel, scored rather than assumed negative.

**The doped-formula matching problem:** Materials Project structures are indexed by integer-reduced formula. Exact-string matching against SuperCon's fractional formulas therefore fails for most doped compounds — only 1,361 of 16,531 SuperCon entries (8.2%) matched a structure this way, and the matches that succeeded were disproportionately the simplest, least doped, least representative compounds. This single bug shaped both paths of the project and is discussed further in §5.

## 3. Methodology: Composition-Based Screening Funnel (deployed)

### 3.1 Featurization and feature selection
Each formula is converted to a 132-dimensional Magpie descriptor vector (matminer's `ElementProperty`, elemental-property statistics: electronegativity, covalent radius, valence-electron counts, etc.), then reduced to 40 features via Pearson-correlation pruning (|r| > 0.95) followed by Recursive Feature Elimination with a LightGBM estimator.

### 3.2 Two-stage funnel
Rather than a single binary classifier, the deployed pipeline is a funnel:
- **Stage 1 (classifier):** XGBoost, P(this composition is superconductor-like), trained on confirmed positives vs. confirmed negatives only (18,223 labeled examples total).
- **Stage 2 (regressor):** XGBoost, predicted Tc (arcsinh-scaled, following 3DSC's own convention — handles the long right tail of the Tc distribution without discarding low-Tc examples the way log1p does), trained on confirmed positives only.
- **Funnel score** = P(superconductor) × predicted Tc, used to rank the unlabeled pool.

### 3.3 Evaluation: leave-one-family-out, not random split
A random train/test split lets near-duplicate compounds (SuperCon is dense with doping-series variants of the same parent compound) leak across the split, inflating apparent accuracy. The only evaluation that predicts performance on a genuinely novel material family is to hold out an entire chemical family and test on it. This requires a family taxonomy for every example, including the 1,776 confirmed negatives — which do not carry SuperCon's own category label. A composition-rule classifier was built to assign one (checking Bismuthates and Iron-based before Cuprates, since 81% of true Bismuthates and 18% of true Iron-based compounds also contain Cu+O and would otherwise be misassigned), and **validated at 99.82% accuracy** against the 16,463 positives that do carry ground-truth labels before being trusted on the negatives.

A further correction was needed within this evaluation itself: an initial leave-one-family-out run excluded only the held-out family's *positive* examples from training, leaving same-family *negatives* in place. Since confirmed negatives are literature-tested near-misses concentrated in the same chemistry as nearby positives (17% of all confirmed negatives contain Cu+O; 5% are Fe-pnictogen compounds), this taught the model "this composition region = negative" with no positive examples left to counter it — producing scores *below random chance* for Cuprates (0.41) and Iron-based (0.32). Excluding the *entire* family — positives and negatives both — from each fold fixed this; see §4.1 for the corrected numbers.

## 4. Results: Composition Funnel

### 4.1 Classifier — leave-one-family-out ROC-AUC

| Family | ROC-AUC | PR-AUC | Notes |
|---|---|---|---|
| Cuprates | 0.96 | 0.997 | |
| Bismuthates | 0.96 | 0.987 | |
| Other | 0.93 | 0.998 | |
| Elemental | 0.93 | 0.71 | small class (64 positives) |
| Organic | 0.91 | 0.81 | only 75 positives, 0 confirmed negatives — noisy |
| Iron-based | 0.66 | 0.91 | usable, wide error bars |
| Borocarbides | 0.64 | 0.66 | see note below |

For reference, the random 90/10 split gives ROC-AUC 0.96, PR-AUC 0.995 — close to the strong end of the family-held-out range, but that agreement does not generalize to the weaker families and should not be quoted as a single headline number.

*Borocarbides note:* the drop to 0.64 is not a bug. Only 8% of SuperCon's own "Borocarbides" category actually contains copper, but those entries (e.g. `B0.65C0.35Ba1.4Sr0.6Ca2Cu3O`) are boron/carbon-*doped cuprates*, not the classic RNi₂B₂C-type intermetallic borocarbide. SuperCon's own labeling convention conflates two chemically distinct material classes under one tag; a held-out test of that combined class is genuinely harder than a chemically coherent one.

### 4.2 Regressor — leave-one-family-out (Tc prediction)

| Family | MSLE | R² | Notes |
|---|---|---|---|
| Random split | 0.12 | 0.92 | not representative — see below |
| Bismuthates | 0.38 | 0.67 | |
| Elemental | 0.42 | 0.38 | |
| Organic | 0.72 | 0.34 | |
| Cuprates | 0.32 | 0.33 | |
| Borocarbides | 0.43 | 0.13 | |
| Other | 0.75 | **−0.20** | worse than predicting the mean |
| Iron-based | 2.26 | **−2.47** | worse than predicting the mean |

Tc regression generalizes even worse than classification to genuinely novel families. The random-split R²=0.92 is not a usable estimate of real-world performance; a predicted-Tc number for a new Iron-based or "Other"-family compound should not be trusted without independent validation.

### 4.3 Sanity check
Applying the funnel to the unlabeled pool, its top-ranked candidates include two of the most famous known superconductors, recovered without being told about them: **H₃S** (rank 1, P=0.976, predicted Tc=134K against a real high-pressure Tc of ≈203K — a reasonable ambient-structure underestimate given the model has no pressure information) and **Ba₂YCu₃O₇ — YBCO** (rank 19, P=0.998, predicted Tc=94.7K against a real Tc of 92K).

### 4.4 Shortlist deliverable
The top-500 ranked candidates are annotated with:
- **Rediscovery flags** — whether the candidate is already a known superconductor under a differently formatted formula string (the same doped-formula matching problem from §2, now working in reverse). 121/500 share an element combination with a known superconductor (doping/stoichiometry variants of known families); 379/500 are genuinely novel element combinations.
- **Inferred family + reliability** — each candidate is tagged with the family it composition-wise resembles and that family's measured leave-one-family-out ROC-AUC from §4.1, rather than presenting a single funnel score with no indication of trustworthiness. 480/500 (96%) fall in a "high reliability" band (>0.90 ROC-AUC); 17 fall in "moderate" (Iron-based, 0.66); 3 — including H₃S and H₂S — have **no measured reliability at all**, because "Hydrogen-rich Superconductors" has only 6 known examples in the entire dataset, too few to evaluate honestly. This is disclosed rather than hidden.

## 5. Abridged: Structure-Based GNN Investigation (Path B)

This path does not appear in the deployed pipeline. It is summarized here because the negative result — and the process of establishing it — is a substantive part of this project's findings.

### 5.1 What was built
A Crystal Graph Convolutional Neural Network (CGCNN; Xie & Grossman, *Phys. Rev. Lett.* 2018), where atoms are nodes (one-hot atomic number) and bonds are edges (Gaussian-expanded distance, 5 Å cutoff), trained for binary superconductor classification, interpreted post-hoc with GNNExplainer.

### 5.2 What was wrong, and what it revealed once fixed

**The reported result was not reproducible.** The GNNExplainer script imported the wrong model class entirely (a never-trained `CGCNN`, when the actual checkpoint was a different, four-layer `ImprovedCGCNN` architecture) and called it with an incompatible signature. GNNExplainer had never successfully run; the "Cu-O plane" figure in the original report was labeled "(Simulated)".

**The original 0.9412 ROC-AUC was a PU-learning artifact**, structurally identical to Path A's problem, compounded by severe class imbalance (up to 46:1). Re-evaluated honestly on the corrected label design (3DSC's matched superconductors and confirmed negatives) with leave-one-family-out splits, the random-split ROC-AUC dropped to 0.84, and several families — Chevrel (0.46), "Other" (0.49), Cuprates (0.53 at best) — sit at or below chance.

**A second, more consequential bug was found and fixed during this correction.** Building the corrected dataset first appeared to succeed, but silently dropped 59.5% of all structures — including 88% of Cuprates specifically — because a pymatgen API call (`site.specie`) crashes on any site with fractional occupancy, which is exactly how 3DSC represents synthetic doping. The very data-completeness fix this project needed had been silently undone by a downstream compatibility bug. Fixing it (occupancy-weighted node features instead of a single-species one-hot) recovered the full 5,773-structure dataset and changed the honest numbers substantially (random-split ROC-AUC 0.70 → 0.84), without changing the qualitative conclusion that several families remain at or near chance.

**Pretrained structural embeddings did not rescue it.** CHGNet (Deng et al., *Nature Machine Intelligence* 2023), pretrained on ~1.5M Materials Project relaxation structures, was used as a frozen feature extractor (avoiding training a structural encoder from scratch on ~5,800 examples). A small classifier head on these frozen embeddings modestly improved some families (Chevrel 0.46→0.58, "Other" 0.49→0.58) but left Cuprates essentially unchanged (0.53→0.54) — evidence that the bottleneck for cuprates specifically is not encoder data-starvation, but that structure alone may not carry enough signal for a family whose physics is dominated by strong electron correlation rather than geometry.

### 5.3 What the explanations actually show
A single-structure GNNExplainer run on YBCO initially looked like validation of real physics (higher importance on Cu than chain-oxygen sites; the model's predicted probability responded correctly to the O₇-vs-O₆ doping difference). Neither result survived scaling up:

- **Aggregated across 40 real cuprate superconductors** (not one), GNNExplainer assigns *more* importance to spacer/dopant cations (Y, Ba, La, Sr, Bi...) than to Cu or O — the opposite of the hypothesis, at overwhelming statistical significance (Mann-Whitney U test, p ≈ 1.0 for "Cu+O > other-cation").
- **A doping-dose-response test** on the La₂₋ₓSrₓCuO₄ series (the original 1986 high-Tc cuprate family, spanning the full real superconducting dome from x=0 to x=0.28) showed the model's predicted probability is a flat, nearly linear function of the raw Sr-occupancy fraction (correlation 0.71 with dopant fraction, only 0.32 with actual Tc), and does *not* reproduce the real dome shape — its single highest-confidence point sits past the real Tc peak, in the region where superconductivity is actually suppressed by overdoping.

Together, these indicate the model is reading the numeric partial-occupancy value at a disordered site as a scalar predictive feature — a real, label-correlated signal in the training data, but not the electronic-structure mechanism it was reported to have learned. This is a concrete, falsifiable finding, not a subjective impression: it was established by testing at scale (40 structures, a full doping series) rather than trusting a single explainer run, which is standard practice in this project's GNN work but not universal in the wider literature.

### 5.4 Conclusion for Path B
As a shortlisting tool, the structural GNN offers no measurable enrichment over chance on several major superconductor families, even after every identified implementation bug was fixed and a pretrained encoder was substituted for the from-scratch one. It is excluded from the deployed pipeline on that basis. The investigation nonetheless produced a real, load-bearing finding: on this dataset and at this scale, structure-only graph representations do not yet capture the physics that separates a real cuprate superconductor from a chemically similar non-superconductor, and naive post-hoc explanations of such a model can look convincing while being wrong.

## 6. Comparison to Related Work

| Work | Target | Data | Result |
|---|---|---|---|
| Stanev et al. 2018 | Tc classification/regression from composition | SuperCon (Magpie features) | Foundational composition-ML baseline this project's Path A descends from |
| Sommer et al. 2023 (3DSC) | Tc regression, composition vs. structure | 3DSC (same dataset used here) | Structure improves MSLE only marginally (0.748 vs 0.776) and **not statistically significantly** under grouped cross-validation by chemical family — independently reaches this report's §5 conclusion |
| BETE-NET (2024) | Electron-phonon spectral function → Tc via Allen-Dynes | 818 DFT-computed examples | Tc MAE 2.1K, ~5× enrichment over random screening — the actual state of the art for this exact "GNN as high-recall shortlister" use case, achieved by predicting physics, not a composition/structure→binary label |
| AI-accelerated discovery workflow (2026) | Full candidate-to-synthesis funnel | 1.3M candidates | 741 DFT-confirmed stable compounds, 86% overall precision, via a staged classical→ML→DFT pipeline this project's funnel is a (composition-only, pre-DFT) fragment of |

The composition funnel in this report is a legitimate but considerably shallower baseline next to BETE-NET and the full discovery workflow: it has no DFT validation stage and predicts a literature label rather than a physical quantity. The honest path to closing that gap is not a better GNN architecture on the same target, but changing the target itself — predicting DFT-computable electron-phonon quantities (λ, ω_log) as BETE-NET does, which is future work (§8).

## 7. Explainability: What Held Up and What Didn't

**SHAP (Path A).** Two different classifiers have been explained, and they do not agree, so every result below is labeled by the model it came from. The *original-methodology* classifier uses unlabeled-as-negative labels (§1.3); the *shipped* funnel classifier uses confirmed labels (§3). Both are examined by `path_a_classical/explain_funnel.py` (W&B run `58lszl8z`), which retrains on the original labels for the first group of rows below and explains the shipped models directly for the second.

The original report's top design rule was "low mean covalent radius predicts superconductivity, via BCS phonon stiffness". Testing whether that feature is oxide-scoped (oxygen has a very small covalent radius, and cuprates dominate the positives):

| Model / subset | Rank of mean covalent radius (of 40) | corr(value, SHAP) |
|---|---|---|
| original labels, all data | 2 | +0.76 to +0.79 |
| original labels, oxygen-free only | 13–14 | +0.69 to +0.71 |
| original labels, oxide-only | 1 | +0.77 |
| **shipped** classifier, all data | 6 | +0.84 |
| **shipped**, SHAP on oxygen-free rows | 4 | +0.77 |
| **shipped**, SHAP on oxide rows | 8 | +0.91 |
| retrained on oxygen-free only | 3 | +0.81 |
| retrained on oxide-only | 6 | +0.88 |

(Every row is reproduced by `explain_funnel.py`; ranges reflect run-to-run variation of the refit models (§1.3), while the shipped-model rows are exactly reproducible. The "original labels" rows and the "retrained" rows fit a fresh classifier on each subset and explain its held-out split; the "shipped" rows slice the SHAP values of the deployed classifier without retraining.)

- **The oxide-scoping finding does not carry over to the shipped model.** With the original labels the feature collapsed from #2 to #13–14 without oxygen. On the shipped model it is #6 overall and, if anything, *more* important outside oxides (#4 and #3) than inside them (#8 and #6). The earlier result was a property of that label design (negatives = generic stable materials), not a general property of the feature.
- **The sign is consistently opposite to the original claim.** In every row above, a *higher* mean covalent radius pushes toward "superconductor" (positive correlation), not lower as the BCS-stiffness narrative requires. The narrative is unsupported by either model. Caveat: a Pearson correlation is a crude summary of a possibly non-monotonic SHAP dependence; the feature also correlates −0.66 with oxygen fraction, while oxygen fraction alone separates the classes only weakly (ROC-AUC 0.59).
- **What the shipped funnel relies on most is chemical contrast, not radius.** Range of electronegativity is the top feature of both the classifier and the regressor, followed by the range of Mendeleev number, atomic number and melting temperature — how *different* the constituent elements are. This is consistent with the original report's second rule ("high electronegativity range"), the one design rule that survives re-testing. The original third rule (low mean d-valence electrons) does not appear in the classifier's top 10 and only in the regressor's. Note that these are associations with a label that separates superconductors from literature-tested near-misses, not evidence of a mechanism.

Scope of these explanations: SHAP for the shipped models is computed on the data they were fit on, so it describes what the models use, not how well they generalize (§4 covers that); and the shipped models come from an untracked run, so the W&B run explains them but did not produce them.

**GNNExplainer (Path B).** Discussed in §5.3. The tool is mechanically valid; what it reveals, once used rigorously (aggregated across many examples, cross-checked against a controlled doping series) rather than trusted from one anecdote, is that the underlying model's reasoning does not match the physical story that was hoped for.

The general lesson, applicable to both paths: an explanation describes a *(model, label design)* pair, not the underlying physics. The same feature went from rank #2 and oxide-scoped to rank #6 and not oxide-scoped when only the label design changed. A single SHAP ranking or GNNExplainer heatmap is therefore not evidence of a physical mechanism until it has survived a confound check and been shown to hold across independent examples, subsets or a controlled physical variable.

## 8. Limitations and Future Work

- **No DFT validation stage.** The funnel's output is a triage shortlist, not a validated candidate list. A responsible next step is stability (DFT) and, ideally, electron-phonon calculation on the top-ranked, non-rediscovered candidates before any synthesis claim.
- **Regressor is unreliable for Iron-based and heterogeneous ("Other") families** (§4.2) — do not use predicted-Tc values for these families without independent checks.
- **Composition-only ceiling.** As shown independently by 3DSC and by this project's Path B investigation, 3D structure does not currently add much over composition on this label design — closing that gap likely requires predicting physical intermediates (electron-phonon coupling) rather than a literature-derived label, following BETE-NET's approach.
- **Confirmed-negative set is small and chemically narrow** (1,776 examples, concentrated in a few families) relative to the true diversity of non-superconducting materials — worth expanding if more curated negative data becomes available.
- **Cuprates remain the hard case for every approach tried** (the BCS-motivated covalent-radius narrative does not apply to them and is unsupported by SHAP anyway; structural GNN, even with pretrained embeddings, does not exceed 0.54 ROC-AUC leave-one-out) — consistent with cuprates being governed by strong electron correlation physics that neither composition statistics nor local structural geometry directly encode.

## 9. Conclusion

A composition-based, two-stage screening funnel — trained on properly confirmed labels and evaluated by leave-one-family-out validation rather than random splits — provides genuine, if chemistry-dependent, enrichment for superconductor candidate shortlisting (ROC-AUC 0.64–0.96 across families), recovers known superconductors from an unlabeled pool without being told about them, and ships with an honest per-candidate reliability annotation rather than a single overconfident score. A parallel structure-based GNN investigation, subjected to the same standard of evidence, does not clear that bar — its apparent success in the original report was an artifact of label design and broken interpretability tooling, and once both were fixed, its explanations turned out to reflect a chemically shallow shortcut rather than the physics it was reported to have learned. Both outcomes — one a working tool, one a well-evidenced negative result — are reported here at the same standard of scrutiny.

## References

1. V. Stanev et al., "Machine learning modeling of superconducting critical temperature," *npj Comput. Mater.* 4, 29 (2018).
2. T. Sommer, R. Willa, J. Schmalian, P. Friederich, "3DSC — a dataset of superconductors including crystal structures," *Sci. Data* 10, 816 (2023).
3. T. Xie, J. C. Grossman, "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties," *Phys. Rev. Lett.* 120, 145301 (2018).
4. B. Deng et al., "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling," *Nat. Mach. Intell.* (2023).
5. BETE-NET: "Accelerating superconductor discovery through tempered deep learning of the electron-phonon spectral function," *npj Comput. Mater.* (2024).
6. "Developing a complete AI-accelerated workflow for superconductor discovery," *npj Comput. Mater.* (2026).
7. A. Jain et al., "The Materials Project: A materials genome approach to accelerating materials innovation," *APL Mater.* 1, 011002 (2013).
8. SuperCon Dataset, NIMS Material Data Repository.
9. J. G. Bednorz, K. A. Müller, "Possible high Tc superconductivity in the Ba–La–Cu–O system," *Z. Phys. B* 64, 189 (1986).
10. R. Ramprasad et al., "Machine learning in materials informatics: recent applications and prospects," *npj Comput. Mater.* 3, 54 (2017).
11. A. Dunn et al., "Benchmarking materials property prediction methods: the Matbench test set and Automatminer automated machine learning pipeline," *npj Comput. Mater.* 6, 138 (2020).
