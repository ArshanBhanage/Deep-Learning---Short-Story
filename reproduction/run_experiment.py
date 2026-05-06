"""
TabPFN-2.5 Reproduction Experiment
===================================
Reproduces the core claim from "TabPFN-2.5: Advancing the State of the Art
in Tabular Foundation Models": tabular foundation models can be strong
default classifiers on tabular data, often matching or beating tuned
tree-based methods without hyperparameter search.

We compare several models on two public sklearn datasets:
  1. Breast Cancer Wisconsin (binary classification)
  2. Wine (multiclass classification)

Models tested:
  - Logistic Regression (baseline)
  - Random Forest (ensemble baseline)
  - XGBoost (gradient-boosted trees, if available)
  - TabPFN (tabular foundation model, if available)

Metrics: Accuracy, Weighted F1-Score
Evaluation: 5-fold stratified cross-validation

Usage:
  python3 reproduction/run_experiment.py
"""

import os
import sys
import time
import warnings
import csv
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Check optional dependencies
# ---------------------------------------------------------------------------
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    print("[INFO] XGBoost not available — skipping XGBoost comparison.")
    print("       Install with: pip install xgboost")
    print("       On macOS, you may also need: brew install libomp\n")

TABPFN_AVAILABLE = False
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    print("[INFO] TabPFN not installed — skipping TabPFN comparison.")
    print("       Install with: pip install tabpfn")
    print("       Requires Python ≥ 3.9 and PyTorch.")
    print("       See https://github.com/PriorLabs/TabPFN for details.\n")

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS = {
    "Breast Cancer": load_breast_cancer(return_X_y=True),
    "Wine": load_wine(return_X_y=True),
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def get_models():
    """Return a dict of model_name -> model instance."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs",
            multi_class="auto",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
    if TABPFN_AVAILABLE:
        models["TabPFN"] = TabPFNClassifier(device="cpu", n_estimators=16)
    return models

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(X, y, models, dataset_name):
    """Run stratified k-fold CV and return a list of result dicts."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for model_name, model in list(models.items()):
        accs, f1s, times = [], [], []
        skip_model = False
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standard-scale for Logistic Regression (benefits from it)
            if model_name == "Logistic Regression":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            try:
                start = time.time()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                elapsed = time.time() - start
            except Exception as e:
                print(f"  {model_name:25s}  SKIPPED — error: {e}")
                skip_model = True
                break

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            accs.append(acc)
            f1s.append(f1)
            times.append(elapsed)

        if skip_model:
            continue

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
            "time_mean_s": np.mean(times),
        })
        print(f"  {model_name:25s}  acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
              f"time={np.mean(times):.3f}s")
    return results

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
COLORS = {
    "Logistic Regression": "#6C5CE7",
    "Random Forest": "#00B894",
    "XGBoost": "#E17055",
    "TabPFN": "#0984E3",
}

def plot_metric(all_results, metric_key, ylabel, title, filename):
    """Grouped bar chart: one group per dataset, one bar per model."""
    df = pd.DataFrame(all_results)
    datasets = df["dataset"].unique()
    models = df["model"].unique()
    n_datasets = len(datasets)
    n_models = len(models)
    bar_width = 0.18
    x = np.arange(n_datasets)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")

    for i, model in enumerate(models):
        means = []
        stds = []
        for ds in datasets:
            row = df[(df["dataset"] == ds) & (df["model"] == model)]
            means.append(row[f"{metric_key}_mean"].values[0])
            stds.append(row[f"{metric_key}_std"].values[0])
        color = COLORS.get(model, "#CCCCCC")
        bars = ax.bar(x + i * bar_width, means, bar_width,
                      yerr=stds, capsize=4, label=model,
                      color=color, edgecolor="white", linewidth=0.5,
                      alpha=0.9, zorder=3)
        # value labels
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8,
                    color="white", fontweight="bold")

    ax.set_xlabel("Dataset", fontsize=12, color="white")
    ax.set_ylabel(ylabel, fontsize=12, color="white")
    ax.set_title(title, fontsize=14, color="white", fontweight="bold", pad=15)
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=11, color="white")
    ax.tick_params(axis="y", colors="white")
    ax.legend(loc="lower right", fontsize=9, facecolor="#2D2D44",
              edgecolor="white", labelcolor="white")
    ax.set_ylim(0.80, 1.02)
    ax.grid(axis="y", alpha=0.15, color="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  TabPFN-2.5 Reproduction Experiment")
    print("  Core claim: tabular foundation models are strong defaults")
    print("=" * 65)
    print(f"\n  Datasets:  {', '.join(DATASETS.keys())}")
    print(f"  CV folds:  {N_SPLITS}")
    print(f"  XGBoost:   {'available' if XGBOOST_AVAILABLE else 'NOT available'}")
    print(f"  TabPFN:    {'available' if TABPFN_AVAILABLE else 'NOT available'}")
    print()

    models = get_models()
    all_results = []

    for ds_name, (X, y) in DATASETS.items():
        print(f"--- {ds_name} ({X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(np.unique(y))} classes) ---")
        results = evaluate(X, y, models, ds_name)
        all_results.extend(results)
        print()

    # Save metrics CSV
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    fieldnames = ["dataset", "model", "accuracy_mean", "accuracy_std",
                  "f1_mean", "f1_std", "time_mean_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[SAVED] {csv_path}")

    # Generate figures
    plot_metric(all_results, "accuracy", "Accuracy",
                "Model Accuracy Comparison (5-Fold CV)",
                "model_accuracy_comparison.png")
    plot_metric(all_results, "f1", "Weighted F1-Score",
                "Model F1-Score Comparison (5-Fold CV)",
                "model_f1_comparison.png")

    # Summary table
    print("\n" + "=" * 65)
    print("  SUMMARY TABLE")
    print("=" * 65)
    df = pd.DataFrame(all_results)
    summary = df.pivot_table(
        index="model",
        columns="dataset",
        values=["accuracy_mean", "f1_mean"],
        aggfunc="first"
    )
    print(summary.to_string())

    # TabPFN placeholder note
    if not TABPFN_AVAILABLE:
        print("\n" + "-" * 65)
        print("  NOTE ON TabPFN")
        print("-" * 65)
        print("  TabPFN was not included because the 'tabpfn' package is not")
        print("  installed. To run TabPFN and reproduce the full comparison:")
        print()
        print("    1. pip install tabpfn")
        print("    2. Re-run: python3 reproduction/run_experiment.py")
        print()
        print("  TabPFN requires PyTorch and will download ~100 MB model weights")
        print("  on first run. CPU inference is fine for these small datasets.")
        print("  Based on the paper, TabPFN should match or beat Random Forest")
        print("  and XGBoost on these small-to-medium datasets without any tuning.")
        print("-" * 65)

    print("\n[DONE] Experiment complete.")
    return all_results


if __name__ == "__main__":
    main()
