# Analysis Report — Iris Flower Classification

## 1. Objective

Classify iris flowers into one of three species — *setosa*, *versicolor*, or *virginica* — from four physical measurements (sepal length, sepal width, petal length, petal width), and compare multiple classification models to identify the best performer.

## 2. Dataset

The built-in `sklearn.datasets.load_iris()` dataset: 150 samples, 4 numeric features, 3 perfectly balanced classes (50 samples each). No missing values were found.

## 3. Exploratory Data Analysis

Per-species descriptive statistics showed that setosa's petal measurements are tightly clustered and clearly separated from the other two species, while versicolor and virginica overlap more on sepal measurements. This was confirmed visually:

- **Pairplot:** setosa forms a fully separate cluster on every feature pair involving petals; versicolor and virginica overlap somewhat, most heavily on sepal length/width.
- **Box plots:** petal length and petal width show the cleanest separation across all three species; sepal width shows the most overlap.
- **Correlation heatmap:** petal length and petal width are strongly correlated (0.96), indicating redundancy between them.

## 4. Feature Selection

Two features were selected for modeling: **petal length** (primary, given its strong discriminative power) and **sepal width** (secondary, chosen over petal width specifically to avoid the 0.96 multicollinearity between the two petal measurements).

This was a deliberate trade-off, not an oversight: avoiding multicollinearity is a reasonable general instinct, particularly for coefficient-based models like Logistic Regression. To test whether this trade-off cost predictive accuracy, the analysis was extended to directly compare this pairing against **petal length + petal width** across many random seeds (Section 6).

## 5. Modeling and Evaluation

Four classifiers were trained on an 80/20 stratified train/test split (`random_state=256`): Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest.

**Single-seed results (petal length + sepal width, and petal length + petal width):**

| Feature Pair | Model | Accuracy | Balanced Accuracy |
|---|---|---|---|
| petal length + petal width | K-Nearest Neighbors | 0.967 | 0.967 |
| petal length + petal width | Logistic Regression | 0.967 | 0.967 |
| petal length + petal width | Random Forest | 0.967 | 0.967 |
| petal length + sepal width | K-Nearest Neighbors | 0.967 | 0.967 |
| petal length + petal width | Decision Tree | 0.933 | 0.933 |
| petal length + sepal width | Logistic Regression | 0.933 | 0.933 |
| petal length + sepal width | Random Forest | 0.933 | 0.933 |
| petal length + sepal width | Decision Tree | 0.833 | 0.833 |

At this single seed, four model/feature-pair combinations tie for the top score (96.7%). A single train/test split isn't enough to declare one model definitively best, since ties like this can be an artifact of one particular random split — which motivated the robustness check below.

## 6. Extended Investigation: Robustness Across 40 Seeds

To move past a single lucky (or unlucky) split, all four models were retrained across 40 different random seeds, for both feature pairs, with results averaged.

**Unscaled, 40-seed average accuracy:**

| Feature Pair | Model | Accuracy |
|---|---|---|
| petal length + petal width | Logistic Regression | 0.966 |
| petal length + petal width | K-Nearest Neighbors | 0.963 |
| petal length + petal width | Random Forest | 0.958 |
| petal length + sepal width | Logistic Regression | 0.952 |
| petal length + sepal width | K-Nearest Neighbors | 0.948 |
| petal length + petal width | Decision Tree | 0.945 |
| petal length + sepal width | Random Forest | 0.938 |
| petal length + sepal width | Decision Tree | 0.925 |

**Finding 1 — the feature-selection trade-off had a real, if modest, cost.** Averaged across all 40 seeds, **every one of the four models** performed better with petal length + petal width than with petal length + sepal width. The original choice of sepal width (to avoid multicollinearity) was reasonable, but it did trade away some accuracy — low correlation between two features signals that they carry different information, not that the different information is necessarily useful.

**Finding 2 — Logistic Regression, not KNN, was the strongest model on average.** While KNN tied for the top single-seed score, Logistic Regression came out ahead once averaged over 40 seeds for both feature pairs — a reminder that single-split results can be misleading.

## 7. Extended Investigation: Effect of Feature Scaling

The same 40-seed test was repeated with `StandardScaler` applied (fit on train, transformed on test).

**Scaled, 40-seed average accuracy:**

| Feature Pair | Model | Accuracy |
|---|---|---|
| petal length + petal width | Random Forest | 0.959 |
| petal length + petal width | K-Nearest Neighbors | 0.958 |
| petal length + petal width | Logistic Regression | 0.956 |
| petal length + petal width | Decision Tree | 0.948 |
| petal length + sepal width | Logistic Regression | 0.938 |
| petal length + sepal width | Random Forest | 0.937 |
| petal length + sepal width | Decision Tree | 0.925 |
| petal length + sepal width | K-Nearest Neighbors | 0.925 |

**Finding 3 — scaling helped or hurt depending on how a model uses feature magnitude, not uniformly.** Comparing scaled vs. unscaled accuracy per model:

- **Logistic Regression and K-Nearest Neighbors both dropped** (up to 2.3 points) — both compute distances or fit coefficients that are sensitive to a feature's raw scale.
- **Decision Tree and Random Forest were essentially unaffected** (well within noise, ±0.3 points) — both split on feature *order*, not magnitude, so a linear rescaling shouldn't change their structure, and it didn't.

The original hypothesis was that petal length's larger raw variance was unfairly dominating distance/coefficient calculations, and that scaling would fix this and improve accuracy. The experiment showed the opposite: petal length's dominance wasn't distortion, it was correctly weighting the more informative feature more heavily. Scaling forced the weaker feature to contribute equally, which pulled some predictions in the wrong direction. This is a case where the diagnosis (petal length dominates the math) was correct, but the assumption that this domination was harmful was not.

## 8. Conclusions

- **Best overall model:** Logistic Regression, when averaged across 40 random seeds — narrowly ahead of K-Nearest Neighbors and Random Forest, and consistently so across both feature pairs.
- **Feature selection trade-off:** choosing sepal width over petal width to reduce multicollinearity cost a small but consistent amount of accuracy across every model tested; it was a defensible choice for coefficient interpretability, not for maximizing raw predictive performance.
- **Feature scaling:** not a universal best practice. It should be applied when a feature's scale is arbitrary, not when that scale reflects genuine signal strength — here it measurably hurt distance- and coefficient-based models while leaving tree-based models unchanged.
- **Methodological note:** single train/test splits can produce ties or misleading rankings (as seen at `random_state=256`); averaging across multiple seeds gave a more reliable picture of true model performance.
