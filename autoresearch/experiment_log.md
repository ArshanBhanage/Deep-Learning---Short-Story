# Autoresearch Experiment Log

## Experiment: TabPFN-2.5 Reproduction

**Date:** 2026-05-06
**Paper:** TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models (arXiv:2511.08667v2)
**Goal:** Reproduce core claim — tabular foundation models are strong default classifiers

---

### Setup

- **Python:** 3.9.6
- **scikit-learn:** 1.6.1
- **Evaluation:** 5-fold Stratified Cross-Validation
- **Datasets:** Breast Cancer Wisconsin (569×30, 2 classes), Wine (178×13, 3 classes)

### Models

| Model | Hyperparameters | Status |
|-------|----------------|--------|
| Logistic Regression | max_iter=2000, solver=lbfgs | ✅ Run |
| Random Forest | n_estimators=200 | ✅ Run |
| XGBoost | n_estimators=200, lr=0.1, max_depth=6 | ❌ Skipped (libomp missing) |
| TabPFN | N_ensemble_configurations=16 | ❌ Skipped (not installed) |

### Iteration 0 — Baseline (2026-05-06)

**Hypothesis:** Default Logistic Regression and Random Forest should provide a competitive baseline (>95% accuracy) on these clean, well-separated datasets.

**Results:**

| Dataset | Model | Accuracy | F1 |
|---------|-------|----------|----|
| Breast Cancer | Logistic Regression | 0.9737 | 0.9734 |
| Breast Cancer | Random Forest | 0.9543 | 0.9542 |
| Wine | Logistic Regression | 0.9833 | 0.9833 |
| Wine | Random Forest | 0.9775 | 0.9775 |

**Outcome:** ✅ Confirmed. Both models achieve >95% accuracy with default parameters.

**Observations:**
- Logistic Regression outperforms Random Forest on both datasets, likely because these are small, clean datasets with well-separated classes where a linear decision boundary is sufficient.
- This aligns with the paper's premise: on small tabular data, simpler methods can be strong. TabPFN's value proposition is being *at least* this good without any tuning, even on messier, more complex datasets.

### Next Steps

1. Install `libomp` and re-run with XGBoost
2. Install `tabpfn` and re-run to include the foundation model comparison
3. Consider adding more challenging datasets (e.g., OpenML datasets with missing values, categorical features) where TabPFN's advantages would be more visible

### Output Artifacts

- `reproduction/results/metrics.csv` — raw results
- `reproduction/figures/model_accuracy_comparison.png` — accuracy chart
- `reproduction/figures/model_f1_comparison.png` — F1 chart
- `reproduction/README.md` — experiment documentation
