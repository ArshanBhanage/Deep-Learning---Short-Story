# Reproduction Experiment — TabPFN-2.5

## Goal

Reproduce the core claim from *"TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models"*: tabular foundation models can serve as strong default classifiers, matching or outperforming tuned tree-based methods without hyperparameter search.

## Approach

We compare four classification models on two public sklearn datasets using 5-fold stratified cross-validation:

| Dataset | Samples | Features | Classes |
|---------|---------|----------|---------|
| Breast Cancer Wisconsin | 569 | 30 | 2 |
| Wine | 178 | 13 | 3 |

### Models Compared

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble (200 trees) |
| XGBoost | Gradient-boosted trees |
| TabPFN | Tabular foundation model |

### Metrics

- **Accuracy** (mean ± std over 5 folds)
- **Weighted F1-Score** (mean ± std over 5 folds)

## How to Run

The experiment runs on **Google Colab with a GPU runtime** (TabPFN requires GPU for reasonable speed):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArshanBhanage/Deep-Learning---Short-Story/blob/main/reproduction/run_experiment_colab.ipynb)

1. Click the badge above or upload `run_experiment_colab.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells

## Results

| Dataset | Model | Accuracy | F1-Score |
|---------|-------|----------|----------|
| Breast Cancer | Logistic Regression | 0.9737 ± 0.0166 | 0.9734 ± 0.0170 |
| Breast Cancer | Random Forest | 0.9543 ± 0.0102 | 0.9542 ± 0.0103 |
| Breast Cancer | XGBoost | 0.9631 ± 0.0086 | 0.9629 ± 0.0088 |
| Breast Cancer | **TabPFN** | **0.9772 ± 0.0089** | **0.9771 ± 0.0090** |
| Wine | Logistic Regression | 0.9833 ± 0.0136 | 0.9833 ± 0.0136 |
| Wine | Random Forest | 0.9775 ± 0.0213 | 0.9775 ± 0.0213 |
| Wine | XGBoost | 0.9606 ± 0.0291 | 0.9604 ± 0.0295 |
| Wine | **TabPFN** | **0.9832 ± 0.0137** | **0.9832 ± 0.0138** |

**TabPFN achieves the highest accuracy on Breast Cancer and ties for the top on Wine — with zero hyperparameter tuning.** This confirms the paper's core claim.

## Output Files

| File | Description |
|------|-------------|
| `run_experiment_colab.ipynb` | Full experiment notebook (run on Colab with GPU) |
| `results/metrics.csv` | Numeric results |
| `figures/model_accuracy_comparison.png` | Accuracy bar chart |
| `figures/model_f1_comparison.png` | F1-score bar chart |
