# Car Price Prediction with Machine Learning

**Oasis Infobyte SIP — Data Science Track — Task 3**

Predicting the selling price of a used car from features like brand, age, mileage, fuel type, and transmission, and comparing a diagnostics-checked linear model against a tuned Random Forest.

## Objective

Build a regression model that predicts a used car's selling price from its listing attributes, and determine which of a car's properties actually drive its price.

## Dataset

- **File:** `Car details v3.csv` (included in this folder)
- **Source:** "Vehicle dataset from cardekho" on Kaggle — real used-car listings scraped from CarDekho.
- **Size:** 8,128 records, 12 features (`name`, `year`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`) + 1 target (`selling_price`). Reduced to 6,702 records after cleaning.

## Project Structure

```
DataScience-Task3-CarPricePrediction/
├── Car details v3.csv
├── Car_Price_Prediction.ipynb
├── REPORT.md
└── README.md
```

## Approach

1. Loaded and cleaned the data — dropped 1,202 exact-duplicate listings (14.8% of the dataset) and ~209 rows missing spec fields (~3%); parsed unit-embedded text columns (`mileage`, `engine`, `max_power`, `torque`) into numeric values; caught and removed 15 rows with a physically impossible `mileage_value` of zero.
2. Engineered `car_age` from the manufacture year and extracted `brand` from the car name.
3. Ran EDA: selling-price distribution (log-transformed to correct right-skew), price-vs-fuel-type boxplots, price-vs-age scatter plots, and a correlation heatmap.
4. Encoded categorical variables (one-hot for nominal features, label encoding for ordinal `owner` and high-cardinality `brand`); caught and removed a mileage-unit flag that turned out to be perfectly collinear with an existing fuel dummy (VIF = inf).
5. Trained a baseline **Linear Regression** and checked its assumptions (multicollinearity via VIF, autocorrelation via Durbin-Watson, homoscedasticity via Breusch-Pagan, normality via Jarque-Bera) — heteroscedasticity was detected, so all inference uses HC3 robust standard errors; a partial F-test confirmed four statistically insignificant variables could be jointly dropped.
6. Trained and hyperparameter-tuned a **Random Forest Regressor** (via `GridSearchCV`), validated across 40 random train/test splits and 10-fold cross-validation to confirm the result wasn't a lucky split.
7. Evaluated every model with MAE, RMSE, and R², and interpreted the linear model's coefficients as percentage price effects.

See [`REPORT.md`](./REPORT.md) for the full write-up, including every diagnostic test's result, the final model's coefficients, and the reasoning behind each cleaning decision.

## Results

| Model | R² | RMSE | MAE |
|---|---|---|---|
| OLS Linear Regression (40-seed average) | 0.8396 | 0.2993 | 0.2277 |
| Random Forest (single seed) | 0.9040 | 0.2321 | 0.1655 |
| Random Forest (40-seed average) | 0.9121 | 0.2217 | 0.1603 |
| **Optimized Random Forest** (best CV score) | **0.9200** | 0.2208 | 0.1603 |

## Key Insights

- **Car age** is the dominant price driver by far — each additional year reduces estimated selling price by ~11.1%, holding everything else constant.
- **Engine power** (`max_power_value`) is the second-strongest driver, consistent across both the correlation analysis and the Random Forest's feature importances.
- **Manual transmission** cars sell for ~15% less than automatics; cars sold by **individual sellers** go for ~9.5% less than through a dealer.
- Random Forest beats the linear model on every metric because price depreciation is nonlinear (steep early, flattening with age) — exactly the kind of curve a tree-based model captures naturally and a linear model structurally can't.

## Tech Stack

Python · pandas · NumPy · scikit-learn · statsmodels · SciPy · matplotlib · seaborn · Jupyter Notebook

## How to Run

1. Clone this repository and navigate to this folder.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy`
3. Open `Car_Price_Prediction.ipynb` and run all cells (Kernel → Restart & Run All).

---

*Part of the Oasis Infobyte Summer Internship Program (Data Science Track).*
