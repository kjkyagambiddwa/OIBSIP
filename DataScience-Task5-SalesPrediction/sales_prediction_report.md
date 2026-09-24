# Sales Prediction Using Python — Analysis Report

**Track:** Data Science — Oasis Infobyte SIP
**Task:** Task 5 · Sales Prediction Using Python
**Objective:** Build a regression model that predicts product sales based on advertising expenditure across TV, Radio, and Newspaper channels, and identify which channel drives sales the most.

---

## 1. Data Source

- **Dataset:** `Advertising.csv`
- **Description:** 200 observations of advertising budgets (in thousands of dollars) across three media channels — **TV**, **Radio**, and **Newspaper** — alongside the resulting **Sales** (in thousands of units) for a single product.
- **Origin:** This is the classic "Advertising" dataset originally distributed with *An Introduction to Statistical Learning* (James, Witten, Hastie & Tibshirani). It is one of the most widely used teaching datasets for linear regression, and is commonly mirrored on Kaggle under "Advertising Dataset" — the version used in this project was sourced from there, as suggested by the task's self-sourcing guideline.
- **Columns:**
  | Column | Description |
  |---|---|
  | TV | Advertising expenditure on TV (in $1000s) |
  | Radio | Advertising expenditure on Radio (in $1000s) |
  | Newspaper | Advertising expenditure on Newspaper (in $1000s) |
  | Sales | Product sales (in 1000s of units) — target variable |
- **Note:** The raw CSV includes a redundant unnamed index column (`Unnamed: 0`), which was dropped immediately after loading.

---

## 2. Exploratory Data Analysis

- **Shape:** 200 rows × 4 columns (after dropping the index column).
- **Missing values:** None (`isna().sum()` returned 0 across all columns).
- **Duplicates:** None.
- **Descriptive statistics:**

  | | TV | Radio | Newspaper | Sales |
  |---|---|---|---|---|
  | mean | 147.04 | 23.26 | 30.55 | 14.02 |
  | std | 85.85 | 14.85 | 21.78 | 5.22 |
  | min | 0.70 | 0.00 | 0.30 | 1.60 |
  | max | 296.40 | 49.60 | 114.00 | 27.00 |

  On average, TV advertising expenditure is far higher than Radio or Newspaper expenditure. Radio and Newspaper show right-skewed distributions with likely extreme values.

### Outlier check

Using the IQR method on `Newspaper` expenditure (the most skewed feature), two rows exceeded the upper bound of 93.625: rows with Newspaper expenditure of 114.0 and 100.9. These were **winsorized** (capped at the upper bound) rather than dropped, to preserve sample size while limiting their leverage on the model.

### Visualizations

- **Pairplot** of TV, Radio, Newspaper, and Sales: showed a clear positive linear relationship between Sales and both TV and Radio expenditure, with inconsistent (funnel-shaped) variance. Newspaper showed a much weaker relationship with Sales.
- **Individual scatter plots** (Sales vs. each channel): confirmed the pairplot pattern — TV and Radio expenditure visibly track with higher Sales; Newspaper expenditure does not.
- **Correlation heatmap:** confirmed Newspaper's low linear correlation with Sales relative to TV and Radio.

---

## 3. Train/Test Split

An 80/20 split was used (`test_size=0.2`, `random_state=256`), with `Sales` as the target and `TV`, `Radio`, `Newspaper` as features.

---

## 4. Modeling

### 4.1 Baseline: Linear Regression

| Metric | Value |
|---|---|
| R² | 0.8461 |
| RMSE | 1.7672 |
| MAE | 1.2978 |

**Assumption checks on the baseline model:**
- *Linearity:* Residuals-vs-fitted plot showed a downward-bending pattern, indicating heteroscedasticity and an unmodeled non-linear relationship.
- *Normality:* Shapiro-Wilk test rejected normality (statistic 0.9079, p ≈ 0.0000).
- *Multicollinearity:* VIF values were all low (TV: 2.50, Radio: 3.30, Newspaper: 3.12) — no evidence of multicollinearity.
- *Independence:* Durbin-Watson statistic of 2.0097 — residuals are independent.

The failed linearity/normality checks motivated the next step: modeling non-linear effects directly.

### 4.2 Polynomial Regression with Interaction Terms

Features were centered/scaled (`StandardScaler`) and expanded to degree-2 polynomial + interaction terms (`TV`, `Radio`, `Newspaper`, `TV²`, `TV·Radio`, `TV·Newspaper`, `Radio²`, `Radio·Newspaper`, `Newspaper²`).

| Metric | Value |
|---|---|
| R² | 0.9867 |
| RMSE | 0.5198 |
| MAE | 0.4349 |

Rechecking assumptions on this model: residuals passed the Shapiro-Wilk normality test (statistic 0.9812, p = 0.7328), and VIFs across all 9 terms stayed below 2.1 — no multicollinearity concerns despite the added interaction/polynomial terms.

An OLS summary (`statsmodels`) showed that while `TV`, `Radio`, `TV²`, and `TV·Radio` were highly statistically significant (p < 0.001), the `Newspaper` term and all of its interactions were **not** statistically significant (p-values ranging from 0.10 to 0.77).

### 4.3 Regularization and Feature Selection

To address the statistical noise from the Newspaper terms, four feature-selection approaches were compared on the scaled polynomial feature set:

| Model | Test R² | Features Kept |
|---|---|---|
| Elastic Net | 0.9876 | 7 |
| Lasso | 0.9876 | 7 |
| Ridge | 0.9873 | 9 |
| **RFECV** | **0.9859** | **4** |

**RFECV** (Recursive Feature Elimination with Cross-Validation) was selected as the final approach — not because it had the single highest R², but because it produced the most parsimonious model (only 4 features: `TV`, `Radio`, `TV²`, `TV·Radio`) while sacrificing virtually no predictive accuracy relative to Ridge/Lasso/Elastic Net. This trades a marginal ~0.002 drop in R² for a materially simpler, more interpretable, and more stable model — a reasonable bias-variance tradeoff for a model intended to guide real budget decisions.

### 4.4 Final Linear Model (Post-RFECV)

Refit via OLS on the 4 selected features:

| Term | Coefficient | p-value |
|---|---|---|
| const | 14.2650 | < 0.001 |
| TV | 3.7539 | < 0.001 |
| Radio | 2.9335 | < 0.001 |
| TV² | -0.7595 | < 0.001 |
| TV·Radio | 1.3445 | < 0.001 |

R² = 0.986 (train), Test R² = 0.9859, RMSE = 0.5355, MAE = 0.4518. All four retained terms are highly significant.

**A caveat surfaced here:** despite the strong R², the residuals of this final model *failed* the Shapiro-Wilk normality test (statistic 0.7916, p ≈ 0.0000) — worse than the full polynomial model. Investigating the largest negative residuals identified two clear outlier observations (very low TV/Radio expenditure paired with unexpectedly low Sales) that were distorting the tails of the residual distribution. Rather than discard these as noise, this was treated as a signal that a purely parametric linear model is sensitive to a small number of anomalous market observations — motivating the parallel use of a tree-based model.

### 4.5 Random Forest Regressor

An initial default Random Forest already performed strongly:

| Metric | Value |
|---|---|
| R² | 0.9797 |
| RMSE | 0.6426 |
| MAE | 0.4808 |

**Hyperparameter tuning** via `GridSearchCV` (5-fold CV, grid over `n_estimators`, `max_depth`, `min_samples_split`) selected `max_depth=10`, `min_samples_split=2`, `n_estimators=200`:

| Metric | Value |
|---|---|
| CV R² (avg) | 0.9749 |
| Test R² | 0.9802 |
| Test RMSE | 0.6342 |
| Test MAE | 0.4650 |

The closeness between cross-validated and held-out test R² indicates the model generalizes well and isn't overfitting.

**Feature importance (Random Forest):**

| Feature | Importance |
|---|---|
| TV | ~62% |
| Radio | ~37% |
| Newspaper | < 1% |

Residuals-vs-fitted for the Random Forest showed no systematic pattern — errors are effectively random noise, and (unlike the linear models) this model isn't reliant on distributional assumptions.

---

## 5. Model Performance Summary

| Model | R² | RMSE | MAE | Features Used |
|---|---|---|---|---|
| Base Linear Regression | 0.8461 | 1.7672 | 1.2978 | 3 (raw main effects) |
| Full Polynomial Regression | 0.9867 | 0.5198 | 0.4349 | 9 (all interaction terms) |
| Optimized RFECV Linear Model | 0.9859 | 0.5355 | 0.4518 | 4 (TV, Radio, TV², TV·Radio) |
| Tuned Random Forest | 0.9802 | 0.6342 | 0.4650 | 3 (raw main effects) |

---

## 6. Key Insights

- **TV is the dominant driver of sales**, both in raw correlation and in every feature-selection method applied (RFECV, Lasso, Elastic Net, Random Forest importance).
- **TV shows diminishing marginal returns**: the negative `TV²` coefficient indicates that sales gains from additional TV expenditure shrink as expenditure gets very high (a saturation effect).
- **TV and Radio have a synergistic (interaction) effect**: running both channels simultaneously produces more sales than either channel alone would predict — captured by the significant, positive `TV·Radio` coefficient.
- **Newspaper expenditure is not a meaningful driver of sales.** It was statistically insignificant in every linear specification and contributed less than 1% feature importance in the Random Forest. Every feature-selection method independently arrived at the same conclusion.
- **A small number of anomalous observations** (very low expenditure, unexpectedly low sales) limit how well a purely parametric linear model satisfies its residual-normality assumption — motivating the use of Random Forest as a robustness check alongside the interpretable linear model.

## 7. Recommendation

Based on this analysis, advertising budget currently allocated to Newspaper should be reallocated toward TV and Radio, with particular attention to campaigns that run TV and Radio concurrently to capture their synergistic effect. The RFECV-selected linear model (`Sales = 14.27 + 3.75·TV + 2.93·Radio − 0.76·TV² + 1.34·TV·Radio`, on standardized inputs) is recommended for budget-scenario planning due to its interpretability, while the tuned Random Forest is recommended where maximum predictive robustness against irregular market conditions is the priority.

---

## 8. Tech Stack

Python, pandas, NumPy, scikit-learn, statsmodels, SciPy, matplotlib, seaborn — in a Jupyter Notebook (`Sales_Prediction.ipynb`).

## 9. How to Reproduce

1. Place `Advertising.csv` in the same directory as `Sales_Prediction.ipynb`.
2. Install dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `scipy`.
3. Run the notebook top to bottom (Kernel → Restart & Run All).
