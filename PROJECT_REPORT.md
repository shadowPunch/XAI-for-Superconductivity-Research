# Superconductor Candidate Screening: Methodology, Findings, and an Abridged Structural-GNN Investigation

**Author:** Nithish Ravikkumar, IIT Roorkee
**Scope:** Revision and extension of the original "Theoretical Modeling of Superconductivity with Explainable AI" project (PHC-391)

---

## Abstract

The original project trained two models on the SuperCon database — a composition-based XGBoost classifier (ROC-AUC 0.9886) and a structure-based Graph Neural Network (ROC-AUC 0.9412) — and reported both as accurate, interpretable predictors of superconductivity. This report is a corrected and reframed version of that work. Re-evaluating both models with leave-one-family-out validation instead of random splits — the only honest test of performance on a genuinely novel material family — shows the original headline numbers were substantially inflated by two independent artifacts: chemical-family leakage in random train/test splits, and a positive-unlabeled (PU) label design in which "negative" examples were simply materials absent from the literature, not confirmed non-superconductors. Once corrected, composition-based screening remains genuinely useful (leave-one-family-out ROC-AUC 0.64–0.96 depending on chemistry) and is the basis of the final deliverable: a two-stage funnel that classifies candidates and ranks them by predicted critical temperature, applied to ~62,000 unlabeled candidate materials to produce a reliability-annotated shortlist. The structural GNN, by contrast, does not survive the same scrutiny — leave-one-family-out performance on cuprates and several other families is at or below chance, and its explanations (via GNNExplainer, aggregated across 40 real cuprate superconductors rather than a single anecdote) show it relies on a chemically shallow shortcut — reading raw dopant-site occupancy fractions — rather than the Cu–O-plane electronic structure it was reported to have learned. That negative result, and the methodology used to establish it, is summarized here; the full investigation and code live outside this repository, which contains only the effective, deployed screening pipeline.

---

## 1. Motivation

Discovering new superconductors has historically relied on chemical intuition and trial and error. Machine learning on existing databases (SuperCon, Materials Project) promises to accelerate this by learning which compositions or structures correlate with superconductivity, and by explaining *why* — providing design rules a physicist can act on. This project set out to build such a system along two paths: a fast composition-based classifier (Path A) and a higher-fidelity structure-based Graph Neural Network (Path B), each paired with an explainability method (SHAP and GNNExplainer respectively).

The central finding of this revision is that the value of such a system depends entirely on how it is labeled and evaluated. Both of these were wrong in the original implementation, in ways that are common in this literature and not specific to this project — the independent, peer-reviewed 3DSC dataset paper (Sommer et al., 2023) reports the same structure-doesn't-help conclusion this report reaches, using a similar grouped-evaluation methodology.

## 2. Data

**Positive examples (16,463):** SuperCon entries with Tc > 0. 85% of these have fractional/doped stoichiometry (e.g. `La1.85Sr0.15CuO4`) rather than integer formulas — a fact that turns out to be central to several of the problems described below.

**Negative examples — two designs, one wrong:**
- *Original design:* "Materials Project structures that are computationally stable (energy above hull ≤ 0.05 eV/atom) and not reported in SuperCon" (~62,000 materials). This is a positive-unlabeled (PU) design, not a binary one — absence from the superconductivity literature does not mean a material was tested and found non-superconducting; most have simply never been tested. Quantified consequence: 1,169 of these 62,019 "negatives" (1.9%) are composition-near-duplicates of real superconductors, including elemental aluminium — a textbook BCS superconductor — labeled negative.
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

**SHAP (Path A).** The original report's top design rule — "low mean covalent radius predicts superconductivity, via BCS phonon stiffness" — does not survive scrutiny. Removing oxygen-containing materials drops this feature from rank #2 to rank #13 of 40; it is rank #1 within oxides alone. The correlation between the feature's value and its SHAP contribution is *positive* (+0.76, consistently so within both the oxide and oxygen-free subsets), the opposite sign from the report's claim. Oxygen fraction alone is not predictive (ROC-AUC 0.51), so this is not a trivial oxygen detector — but it is an oxide-chemistry-scoped signal, and the specific physical narrative attached to it (BCS phonon coupling, which does not apply to cuprates — the majority chemistry in this dataset) is not supported.

**GNNExplainer (Path B).** Discussed in §5.3. The tool is mechanically valid; what it reveals, once used rigorously (aggregated across many examples, cross-checked against a controlled doping series) rather than trusted from one anecdote, is that the underlying model's reasoning does not match the physical story that was hoped for.

The general lesson, applicable to both paths: an explainability method is only as informative as the scrutiny applied to its output. A single SHAP ranking or a single GNNExplainer heatmap is not evidence of a physical mechanism until it has survived a confound check (SHAP) or been shown to hold across many independent examples and a controlled physical variable (GNNExplainer).

## 8. Limitations and Future Work

- **No DFT validation stage.** The funnel's output is a triage shortlist, not a validated candidate list. A responsible next step is stability (DFT) and, ideally, electron-phonon calculation on the top-ranked, non-rediscovered candidates before any synthesis claim.
- **Regressor is unreliable for Iron-based and heterogeneous ("Other") families** (§4.2) — do not use predicted-Tc values for these families without independent checks.
- **Composition-only ceiling.** As shown independently by 3DSC and by this project's Path B investigation, 3D structure does not currently add much over composition on this label design — closing that gap likely requires predicting physical intermediates (electron-phonon coupling) rather than a literature-derived label, following BETE-NET's approach.
- **Confirmed-negative set is small and chemically narrow** (1,776 examples, concentrated in a few families) relative to the true diversity of non-superconducting materials — worth expanding if more curated negative data becomes available.
- **Cuprates remain the hard case for every approach tried** (composition SHAP's physical narrative fails on them; structural GNN, even with pretrained embeddings, does not exceed 0.54 ROC-AUC leave-one-out) — consistent with cuprates being governed by strong electron correlation physics that neither composition statistics nor local structural geometry directly encode.

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
