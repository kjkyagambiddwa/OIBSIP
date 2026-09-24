# Sales Prediction Using Python

**Oasis Infobyte SIP — Data Science Track — Task 5**

Predicting product sales from advertising expenditure across TV, Radio, and Newspaper channels, and identifying which channel actually drives sales.

## Objective

Build a regression model that predicts sales based on advertising expenditure, and determine which advertising channel has the greatest impact on sales.

## Dataset

- **File:** `Advertising.csv` (included in this folder)
- **Source:** The classic "Advertising" dataset from *An Introduction to Statistical Learning*, widely mirrored on Kaggle as the "Advertising Dataset."
- **Size:** 200 records, 3 features (`TV`, `Radio`, `Newspaper` expenditure in $1000s) + 1 target (`Sales`, in 1000s of units).

## Project Structure

```
DataScience-Task5-SalesPrediction/
├── Advertising.csv
├── Sales_Prediction.ipynb
├── sales_prediction_report.md
└── README.md
```

## Approach

1. Loaded and cleaned the data (dropped a redundant index column, confirmed no nulls/duplicates).
2. Ran EDA: descriptive statistics, pairplot, per-channel scatter plots vs. Sales, correlation heatmap.
3. Detected and winsorized two Newspaper-expenditure outliers using the IQR method.
4. Trained a baseline **Linear Regression** model and checked its assumptions (linearity, normality, multicollinearity via VIF, independence via Durbin-Watson) — the baseline failed the linearity/normality checks.
5. Engineered polynomial and interaction terms to capture non-linear effects (TV saturation, TV×Radio synergy), which fixed the assumption violations and lifted R² from 0.85 to 0.99.
6. Compared four feature-selection methods (Ridge, Lasso, Elastic Net, RFECV) to strip out statistically insignificant Newspaper terms, and selected the RFECV model for its balance of accuracy and interpretability.
7. Trained and hyperparameter-tuned a **Random Forest Regressor** (via `GridSearchCV`) as a non-parametric alternative, and compared feature importances.
8. Evaluated every model with MAE, RMSE, and R², and interpreted the results in terms of channel-level business impact.

See [`sales_prediction_report.md`](./sales_prediction_report.md) for the full write-up, including all diagnostic plots' findings, coefficients, and reasoning.

## Results

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Base Linear Regression | 0.8461 | 1.7672 | 1.2978 |
| Full Polynomial Regression | 0.9867 | 0.5198 | 0.4349 |
| Optimized RFECV Linear Model | 0.9859 | 0.5355 | 0.4518 |
| Tuned Random Forest | 0.9802 | 0.6342 | 0.4650 |

## Key Insights

- **TV** is the dominant driver of sales (~62% feature importance), with a clear saturation effect at high expenditure levels.
- **Radio** is a strong secondary driver, and shows a synergistic interaction with TV — running both channels together outperforms either alone.
- **Newspaper** expenditure has virtually no impact on sales (< 1% feature importance; statistically insignificant across every model) and its budget could be reallocated to TV and Radio.

## Tech Stack

Python · pandas · NumPy · scikit-learn · statsmodels · SciPy · matplotlib · seaborn · Jupyter Notebook

## How to Run

1. Clone this repository and navigate to this folder.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy`
3. Open `Sales_Prediction.ipynb` and run all cells (Kernel → Restart & Run All).

---

*Part of the Oasis Infobyte Summer Internship Program (Data Science Track).*
