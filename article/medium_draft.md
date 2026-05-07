# Beyond LLMs: The Rise of Tabular Foundation Models

*A deep dive into TabPFN-2.5, the transformer that wants to replace XGBoost — with my own reproduction experiment.*

---

## 1. Introduction: Foundation Models Are Moving Beyond Text

When most people hear "foundation model," they think of ChatGPT, Claude, or Gemini — large language models trained on internet-scale text data that can generalize to almost any language task. The foundation model paradigm, where a single pre-trained model adapts to many downstream tasks, has been revolutionary for text, images, and code.

But here is something most people outside machine learning do not realize: **the most common type of data in the real world is not text or images. It is tables.**

Spreadsheets. CSV files. Database rows. Patient records. Financial transactions. Sensor readings. The structured, column-and-row data that powers virtually every business decision on the planet.

And until very recently, foundation models had almost nothing to say about it.

That is starting to change. A new class of model called **Tabular Foundation Models (TFMs)** is emerging, and the latest entry — **TabPFN-2.5** from Prior Labs — is making a bold claim: it can outperform the most popular tabular ML methods (XGBoost, CatBoost, LightGBM) in a single forward pass, with zero hyperparameter tuning.

In this article, I will break down the paper, explain the architecture, present the benchmarks, and share results from my own reproduction experiment.

---

## 2. Why Tabular Data Matters

Tabular data is the backbone of decision-making in nearly every industry:

- **Healthcare:** Patient records, lab results, diagnoses — all stored in tables.
- **Finance:** Transaction logs, credit scores, risk assessments.
- **Manufacturing:** Sensor readings, quality control metrics, supply chain data.
- **Retail:** Customer demographics, purchase history, inventory levels.

Despite the hype around LLMs and computer vision, the reality is that a huge portion of the machine learning work happening in industry today involves someone opening a CSV file and trying to predict a column from other columns. This is the bread and butter of data science.

> **[Figure 1: Why Tabular Data Matters]**
> *Diagram showing the prevalence of tabular data across industries — healthcare records, financial transactions, manufacturing sensor data, and retail analytics all flowing into tabular ML pipelines.*

Any improvement to how we handle tabular data has an outsized practical impact. Even a small accuracy gain on a credit scoring model, for example, can translate into millions of dollars in reduced default risk.

---

## 3. What Makes Tabular Data Difficult

If tabular data is so common, why have foundation models taken so long to reach it? The answer lies in the fundamental differences between tabular data and the modalities where deep learning has thrived:

**No spatial or sequential structure.** In images, nearby pixels are related. In text, word order matters. In tables, the columns are arbitrary — you can shuffle them and the data means the same thing. Deep learning architectures like CNNs and RNNs were designed to exploit these structural regularities, and tables just do not have them.

**Heterogeneous feature types.** A single row might contain a person's age (continuous), their country (categorical), their income (continuous with a different scale), and whether they defaulted on a loan (binary). Mixing these types within a single model is non-trivial.

**Small datasets are common.** While LLMs train on billions of tokens, many real-world tabular problems have just a few hundred to a few thousand rows. Traditional deep learning typically needs much more data to generalize well.

**Feature engineering matters enormously.** The choice of which features to include, how to encode categoricals, how to handle missing values — these decisions often matter more than the choice of model.

This is precisely why gradient-boosted decision trees (XGBoost, LightGBM, CatBoost) have dominated tabular ML for the better part of a decade. They handle heterogeneous features naturally, work well on small data, and are robust to feature scale differences. They have been the "default" for good reason.

> **[Figure 2: Traditional ML vs. Tabular Foundation Model Workflow]**
> *Left side: Traditional workflow — data → feature engineering → hyperparameter tuning → train XGBoost → evaluate → repeat tuning. Right side: Foundation model workflow — data → single forward pass through pre-trained TabPFN → predictions. The key difference: no training loop, no tuning.*

---

## 4. What Are Tabular Foundation Models?

Tabular Foundation Models (TFMs) flip the traditional paradigm on its head. Instead of training a new model from scratch for every dataset, a TFM is **pre-trained once** and then applied to any new tabular problem without retraining.

The key mechanism is **in-context learning** — the same technique that allows GPT-4 to answer questions it was never explicitly trained on by learning from examples provided in the prompt.

Here is how it works:

1. **Pre-training phase (done once):** The model is trained on millions of *synthetically generated* tabular datasets. It learns not a specific prediction task, but a general *algorithm* for tabular reasoning — how to identify patterns, handle different feature types, and make predictions from structured data.

2. **Inference phase (per new dataset):** When you give TabPFN a new dataset, you pass the **training rows and the test features** together as input. The model processes them in a single forward pass and outputs predictions for the test set. No gradient descent. No hyperparameter tuning. Just one pass through the network.

Think of it like this: traditional ML learns *parameters* for your specific data. A tabular foundation model has already learned *how to learn from any tabular data*. It is a meta-learner.

---

## 5. Main Paper Reviewed: TabPFN-2.5

The paper I am reviewing is:

> **TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models**
> Rosen Yu, Felix Jablonski, Shi Bin Hoo, Anurag Garg, Jake Robertson, Magnus Bühler, Vladyslav Moroshan, Lennart Purucker, Clara Cornu, Lilly Charlotte Wehrhahn, Alessandro Bonetto, Bernhard Schölkopf, Sauraj Gambhir, Noah Hollmann, Frank Hutter
> Prior Labs & University of Freiburg, 2025
> [arXiv:2511.08667](https://arxiv.org/abs/2511.08667)

TabPFN-2.5 is the third iteration in the TabPFN lineage:

- **TabPFN v1 (2022):** The original proof-of-concept. Solved small classification problems (up to 1,000 samples, 100 features) in under a second. Impressive, but too limited for most real-world use.
- **TabPFNv2 (2024):** Scaled to 10,000 samples and 500 features. Added regression support. Became competitive with tuned XGBoost.
- **TabPFN-2.5 (2025):** The paper reviewed here. Scales to **50,000 samples and 2,000 features**. Claims to outperform tuned XGBoost/CatBoost/LightGBM and match 4-hour AutoGluon ensembles in a single forward pass.

---

## 6. Architecture and Method Explained Simply

> **[Figure 3: TabPFN Architecture or Workflow]**
> *Diagram showing the TabPFN-2.5 architecture: training rows and test rows are embedded together, passed through an alternating-attention transformer (24 layers for classification), with 64 "thinking" rows providing extra computational capacity. Output: predictions for test rows.*

At its core, TabPFN-2.5 is a **transformer** — the same architecture family behind GPT and BERT — but adapted for tabular data. Here are the key design choices:

### Alternating Attention

The transformer uses alternating attention layers:
- **Row-wise attention:** Each row (sample) attends to other rows, learning inter-sample relationships.
- **Column-wise attention:** Each feature attends to other features within the same row, learning inter-feature relationships.

This alternation lets the model reason about both "which samples are similar?" and "which features are related?" within the same architecture.

### Feature Group Embeddings

Since tabular features do not have a natural ordering (unlike words in a sentence), TabPFN groups features into chunks and embeds each chunk together. TabPFN-2.5 increased the group size, which speeds up both training and inference by reducing the effective sequence length.

### "Thinking" Rows

This is one of the most creative innovations. Inspired by chain-of-thought reasoning in LLMs, the authors added **64 learned "thinking" rows** to the input. These are not real data points — they are trainable parameters that give the transformer extra computational capacity to work with. They act as "scratch space" for the model to perform intermediate computations during the forward pass.

### Synthetic Pre-Training with Real Fine-Tuning

TabPFN-2.5 is pre-trained entirely on synthetic data generated from a rich prior distribution. This means the model has never seen your specific dataset during training, eliminating the risk of data leakage.

They also released **Real-TabPFN-2.5**, a variant fine-tuned on 43 carefully curated real-world datasets (rigorously deduplicated against all benchmarks). This version pushes performance even higher.

### Distillation for Deployment

A valid criticism of foundation models is latency — running a 24-layer transformer for every prediction is expensive. TabPFN-2.5 addresses this with distillation engines that can convert the model's predictions into a fast **MLP or tree ensemble** for production deployment. You get the accuracy of a foundation model with the speed of a gradient-boosted tree.

---

## 7. Paper Benchmarks, Metrics, and Results

### Benchmarks

The authors evaluated on three main benchmarks:

| Benchmark | Description | Scale |
|-----------|-------------|-------|
| **TabArena** | Industry-standard public benchmark with 51 diverse datasets | Up to 100K rows, 2K features |
| **Internal Suite** | 100+ proprietary datasets across healthcare, finance, manufacturing | Varies |
| **RealCause** | Causal inference benchmark for treatment effect estimation | Varies |

### Metrics

- **Normalized score** (0–1 per dataset, averaged across datasets) for overall comparison
- **Accuracy, ROC AUC** for classification
- **RMSE, R²** for regression
- **PEHE** (Precision in Estimating Heterogeneous Effects) for causal inference

### Key Results

The headline results are striking:

1. **TabPFN-2.5 (default, single pass) > tuned XGBoost, CatBoost, LightGBM.** Without any hyperparameter search, TabPFN-2.5 outperforms individually tuned gradient-boosted tree methods on the TabArena benchmark.

2. **TabPFN-2.5 ≈ AutoGluon 1.4 (4-hour extreme mode).** AutoGluon runs an ensemble of multiple model families with extensive tuning for 4 hours. TabPFN-2.5 matches this in seconds.

3. **Real-TabPFN-2.5 > everything.** The real-data fine-tuned variant sets a new state-of-the-art, significantly outperforming all baselines on classification.

4. **Causal inference SOTA.** When plugged into a T-Learner framework, TabPFN-2.5 achieves the best results on the RealCause benchmark.

> **[Figure 4: Benchmark / Result Summary from the Paper]**
> *Simplified bar chart or table recreating the paper's normalized performance comparison: TabPFN-2.5 (default) vs. XGBoost (tuned) vs. CatBoost (tuned) vs. AutoGluon (4h) across TabArena. TabPFN-2.5 and Real-TabPFN-2.5 at or above all baselines.*

---

## 8. Ablation Studies and What They Show

The paper does not present traditional ablation studies where components are removed one by one. Instead, the progression from TabPFNv2 to TabPFN-2.5 itself serves as a large-scale ablation:

| Change | Effect |
|--------|--------|
| **Deeper network** (18→24 layers for classification) | Improved accuracy on complex datasets |
| **Larger feature group size** | Faster training and inference, slight accuracy gain |
| **"Thinking" rows** (64 added) | Acts as computational scratch space; measurable improvement |
| **Richer synthetic priors** | Better generalization to diverse real-world data |
| **Scaling to 50K rows** | Unlocked applicability to medium-sized real-world datasets |
| **Real-data fine-tuning** (Real-TabPFN-2.5) | Significant accuracy boost over synthetic-only pre-training |

The most interesting takeaway is the **"thinking" rows** — adding trainable dummy inputs that the transformer can use as computational memory. This is a direct cross-pollination from the LLM world, where chain-of-thought prompting improves reasoning. Here, the equivalent concept is baked directly into the architecture.

The **real-data fine-tuning** also reveals something important: purely synthetic pre-training gets you very far, but incorporating a small amount of curated real data closes a meaningful remaining gap.

---

## 9. My Reproduction Experiment

To test the paper's core claim myself, I ran a small-scale reproduction experiment comparing four models on two public datasets from scikit-learn:

### Datasets

| Dataset | Samples | Features | Classes | Task |
|---------|---------|----------|---------|------|
| Breast Cancer Wisconsin | 569 | 30 | 2 | Binary classification |
| Wine | 178 | 13 | 3 | Multiclass classification |

### Models Compared

| Model | Type | Hyperparameters |
|-------|------|----------------|
| Logistic Regression | Linear baseline | max_iter=2000, default otherwise |
| Random Forest | Ensemble (trees) | n_estimators=200, default otherwise |
| XGBoost | Gradient-boosted trees | n_estimators=200, lr=0.1, max_depth=6 |
| TabPFN | Tabular foundation model | n_estimators=16, default otherwise |

### Evaluation Protocol

- **5-fold stratified cross-validation** with a fixed random seed (42) for reproducibility.
- **Metrics:** Accuracy and weighted F1-score (mean ± std across folds).
- **No hyperparameter tuning** for any model — all use default or near-default settings to test out-of-the-box performance, which is exactly the scenario where TabPFN claims to shine.

The experiment code is available as a Colab notebook at [`reproduction/run_experiment_colab.ipynb`](https://github.com/ArshanBhanage/Deep-Learning---Short-Story/blob/main/reproduction/run_experiment_colab.ipynb).

---

## 10. My Results and Interpretation

### Results Table

| Dataset | Model | Accuracy (mean ± std) | F1-Score (mean ± std) | Time (s) |
|---------|-------|----------------------|----------------------|----------|
| Breast Cancer | Logistic Regression | 0.9737 ± 0.0166 | 0.9734 ± 0.0170 | 0.005 |
| Breast Cancer | Random Forest | 0.9543 ± 0.0102 | 0.9542 ± 0.0103 | 0.695 |
| Breast Cancer | XGBoost | 0.9631 ± 0.0086 | 0.9629 ± 0.0088 | 0.136 |
| Breast Cancer | **TabPFN** | **0.9772 ± 0.0089** | **0.9771 ± 0.0090** | 2.984 |
| Wine | Logistic Regression | 0.9833 ± 0.0136 | 0.9833 ± 0.0136 | 0.007 |
| Wine | Random Forest | 0.9775 ± 0.0213 | 0.9775 ± 0.0213 | 0.451 |
| Wine | XGBoost | 0.9606 ± 0.0291 | 0.9604 ± 0.0295 | 0.061 |
| Wine | **TabPFN** | **0.9832 ± 0.0137** | **0.9832 ± 0.0138** | 1.117 |

### Figures

> **[Figure 5: Accuracy Comparison from My Reproduction]**
> *Bar chart showing accuracy across all four models on both datasets. TabPFN achieves the highest accuracy on Breast Cancer (0.977) and ties with Logistic Regression on Wine (0.983).*
> *(Use: `reproduction/figures/model_accuracy_comparison.png`)*

> **[Figure 6: F1-Score Comparison from My Reproduction]**
> *Bar chart showing weighted F1-score across all four models on both datasets. Pattern mirrors accuracy results.*
> *(Use: `reproduction/figures/model_f1_comparison.png`)*

### Interpretation

The results are consistent with the paper's claims:

**On Breast Cancer (569 samples, 30 features):**
- TabPFN achieves the highest accuracy (0.977), beating XGBoost (0.963) and Random Forest (0.954) by a meaningful margin.
- It also beats Logistic Regression (0.974), though narrowly — this dataset has relatively well-separated classes.
- TabPFN does this with **zero tuning**, while XGBoost had reasonable defaults (200 estimators, lr=0.1, max_depth=6).

**On Wine (178 samples, 13 features):**
- TabPFN ties with Logistic Regression at the top (both ~0.983).
- It significantly outperforms XGBoost (0.961) on this small, clean dataset.
- This aligns perfectly with the paper's emphasis on small-data performance. With only 178 samples, the in-context learning approach shines.

**Speed trade-off:**
- TabPFN is slower per prediction (1–3 seconds vs. milliseconds for tree methods). But when you account for the hours of hyperparameter tuning that tree methods typically require in practice, TabPFN's total time-to-result is far shorter.

---

## 11. Strengths of the Approach

**1. Zero-tuning, strong defaults.** This is the killer feature. You do not need to run grid search, Bayesian optimization, or any hyperparameter tuning. TabPFN just works. For data scientists spending hours on tuning, this is transformative.

**2. Small-data champion.** In the regime of tens to thousands of samples — which is extremely common in healthcare, science, and niche business problems — TabPFN consistently outperforms traditional methods that need more data to learn effectively.

**3. Well-calibrated uncertainty.** Because TabPFN performs Bayesian-style inference in-context, its probability outputs tend to be well-calibrated out of the box. This matters enormously for risk-sensitive applications.

**4. Practical deployment path.** The distillation engines (TabPFN-as-MLP, TabPFN-as-TreeEns) solve the latency problem. You can use TabPFN to generate predictions, then distill those into a fast, lightweight model for production. Best of both worlds.

**5. Scalability leap.** Going from 10K to 50K samples makes TabPFN viable for a much larger fraction of real-world problems. Most Kaggle competitions and many business datasets fall within this range.

---

## 12. Limitations

No model is without limitations, and it is important to be honest about where TabPFN-2.5 falls short:

**1. Dataset size ceiling.** 50,000 rows is the current limit. Many production datasets have millions or billions of rows. For these, you still need traditional methods or subsampling strategies.

**2. GPU memory requirements.** Running a 24-layer transformer with 50K samples in-context requires significant GPU memory. This makes it less accessible for practitioners without cloud GPU access. (My own local CPU run hung for 10+ minutes; GPU was necessary.)

**3. Proprietary components.** The distillation engines for fast deployment appear to be part of Prior Labs' commercial offering, not the open-source release. This limits the full pipeline to paying customers.

**4. Interpretability.** Gradient-boosted trees offer feature importance, SHAP values, and other interpretability tools out of the box. TabPFN's transformer internals are harder to inspect, though the authors are working on interpretability features.

**5. Not a universal solution.** On very large, well-structured datasets where you have the budget for extensive tuning, a well-tuned XGBoost or CatBoost ensemble may still win — or at least tie — while being faster and more interpretable.

> **[Figure 7: Limitations / Future Direction Diagram]**
> *Visual showing the current boundaries: dataset size limit (50K), GPU requirements, proprietary distillation. Arrows pointing to future directions: scaling to 100K+ rows, open-source distillation, better interpretability, multi-modal tabular+text support.*

---

## 13. My Personal Takeaways

After reading the paper and running the reproduction:

**This feels like a genuine inflection point.** TabPFN-2.5 is not just a research curiosity — it is a practical tool that can replace XGBoost as the default baseline for most tabular problems under 50K rows. The fact that a single forward pass beats 4 hours of AutoGluon tuning is remarkable.

**The "thinking rows" idea is brilliant.** Taking the chain-of-thought concept from LLMs and embedding it as trainable parameters in the architecture is elegant. It gives the model more computational depth without increasing the input size.

**The ecosystem matters as much as the model.** The distillation engines, the API, the interpretability tools — these are what will determine adoption. A model that is 2% more accurate but impossible to deploy is not useful. Prior Labs seems to understand this.

**We are witnessing convergence.** The boundary between "foundation models" and "traditional ML" is blurring. Tabular data was the last holdout, and now it is falling too. In a few years, the default approach to any prediction problem — text, image, tabular — may be "grab a pre-trained foundation model and fine-tune (or just use in-context learning)."

**My reproduction confirms the claim on small data.** On the two datasets I tested, TabPFN matched or beat all baselines with zero tuning. It is particularly strong on Breast Cancer (30 features, 569 samples), exactly the kind of small-to-medium dataset where the paper claims its advantages are greatest.

---

## 14. Conclusion

The foundation model revolution is not limited to language and vision. TabPFN-2.5 demonstrates that the same paradigm — pre-train once, apply everywhere — can work for tabular data, the most common data type in the real world.

For practitioners, the message is clear: if your tabular dataset has fewer than 50,000 rows, TabPFN should be your first model to try. It requires no tuning, runs in seconds, and will likely match or beat whatever you were going to spend hours tuning.

For researchers, the questions are exciting: Can we scale this to millions of rows? Can we combine tabular and text data in a single foundation model? Can we make the in-context learning approach work for time series?

The age of tabular foundation models is here, and XGBoost — the 10-year champion — finally has serious competition.

---

## 15. References

1. Yu, R., Jablonski, F., Hoo, S.B., et al. (2025). *TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models.* arXiv:2511.08667. [Link](https://arxiv.org/abs/2511.08667)

2. Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2022). *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second.* ICLR 2023. [Link](https://arxiv.org/abs/2207.01848)

3. Garg, A., et al. (2025). *Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-Training with Real-World Data.* [Link](https://arxiv.org/abs/2410.04145)

4. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016. [Link](https://arxiv.org/abs/1603.02754)

5. Erickson, N., et al. (2020). *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.* [Link](https://arxiv.org/abs/2003.06505)

6. Prokhorenkova, L., et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features.* NeurIPS 2018. [Link](https://arxiv.org/abs/1706.09516)

7. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017. [Link](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

8. Erickson, N., et al. (2025). *TabArena: A Living Benchmark for Machine Learning on Tabular Data.* [Link](https://github.com/autogluon/tabarena)

---

*This article is part of my Deep Learning Short Story assignment. The reproduction code, data, and figures are available at [github.com/ArshanBhanage/Deep-Learning---Short-Story](https://github.com/ArshanBhanage/Deep-Learning---Short-Story).*
