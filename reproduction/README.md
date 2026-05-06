# Reproduction Experiment — TabPFN-2.5

## Goal

Reproduce the core claim from the paper *"TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models"*: that tabular foundation models can serve as strong default classifiers for tabular data, often matching or outperforming tuned tree-based methods without any hyperparameter search.

## Approach

We compare several classification models on two public sklearn datasets using 5-fold stratified cross-validation:

| Dataset | Samples | Features | Classes |
|---------|---------|----------|---------|
| Breast Cancer Wisconsin | 569 | 30 | 2 |
| Wine | 178 | 13 | 3 |

### Models Compared

| Model | Type | Status |
|-------|------|--------|
| Logistic Regression | Linear baseline | ✅ Included |
| Random Forest | Ensemble (200 trees) | ✅ Included |
| XGBoost | Gradient-boosted trees | ⚠️ Optional (requires `xgboost` + `libomp`) |
| TabPFN | Tabular foundation model | ⚠️ Optional (requires `tabpfn` + PyTorch) |

### Metrics

- **Accuracy** (mean ± std over 5 folds)
- **Weighted F1-Score** (mean ± std over 5 folds)

## How to Run

```bash
# Install dependencies
pip install -r reproduction/requirements.txt

# Optional: for XGBoost on macOS
brew install libomp

# Optional: for TabPFN
pip install tabpfn

# Run the experiment
python3 reproduction/run_experiment.py
```

## Results

Results from our run (without XGBoost / TabPFN due to environment constraints):

| Dataset | Model | Accuracy | F1-Score |
|---------|-------|----------|----------|
| Breast Cancer | Logistic Regression | 0.9737 ± 0.0166 | 0.9734 ± 0.0170 |
| Breast Cancer | Random Forest | 0.9543 ± 0.0102 | 0.9542 ± 0.0103 |
| Wine | Logistic Regression | 0.9833 ± 0.0136 | 0.9833 ± 0.0136 |
| Wine | Random Forest | 0.9775 ± 0.0213 | 0.9775 ± 0.0213 |

### Key Observation

Even without TabPFN, the results confirm a well-known baseline finding: on small, clean tabular datasets, even simple models like Logistic Regression can be surprisingly competitive. The paper's claim is that TabPFN would match or exceed these results *without any tuning*, acting as a "strong default." This is the same paradigm we tested here — default hyperparameters only, no search.

## Output Files

| File | Description |
|------|-------------|
| `results/metrics.csv` | Full numeric results |
| `figures/model_accuracy_comparison.png` | Accuracy bar chart |
| `figures/model_f1_comparison.png` | F1-score bar chart |

## Extending This Experiment

To fully reproduce the paper's comparison with TabPFN:

1. Install TabPFN: `pip install tabpfn`
2. Re-run: `python3 reproduction/run_experiment.py`
3. TabPFN will automatically be included in the comparison

The script is designed to gracefully handle missing optional dependencies.
