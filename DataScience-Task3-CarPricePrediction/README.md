# Car Price Prediction with Machine Learning

**OASIS INFOBYTE — Data Science Internship, Task 3**

Predicts the selling price of a used car from features like brand, age, mileage, fuel type, and transmission, using a real-world CarDekho listings dataset. Compares a diagnostics-checked OLS linear model against a tuned Random Forest Regressor.

## Results at a glance

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Optimized Random Forest** (best CV score) | **0.920** | 0.221 | 0.160 |
| OLS Linear Regression (40-seed average) | 0.840 | 0.299 | 0.228 |

Random Forest outperforms OLS across every metric — consistent with the depreciation curve in the data being nonlinear, which a linear model can't fully capture. Full diagnostics, coefficient interpretations, and validation methodology are in `REPORT.md`.

## Dataset

`Car details v3.csv` — CarDekho used-car listings via Kaggle (8,128 rows → 6,702 after cleaning). Columns: `name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats`.

## Tech stack

Python, pandas, NumPy, scikit-learn, statsmodels, matplotlib, seaborn — Jupyter Notebook.

## What's in this folder

- `Car_Price_Prediction.ipynb` — the full analysis: cleaning, feature engineering, EDA, OLS with a full assumption-diagnostic pass (VIF, Durbin-Watson, Breusch-Pagan, Jarque-Bera), Random Forest with cross-validation and hyperparameter tuning, and a final model comparison.
- `REPORT.md` — full narrative writeup for anyone who won't open the notebook.
- `README.md` — this file.

## Approach summary

1. **Cleaning** — dropped 1,202 exact-duplicate listings (14.8%) and ~209 rows with missing spec fields (~3%); parsed unit-embedded text columns (`mileage`, `engine`, `max_power`, `torque`) into numeric values; caught and removed 15 rows with a physically impossible `mileage_value == 0`.
2. **Feature engineering** — `car_age` from `year`; `brand` extracted from `name`.
3. **EDA** — price is right-skewed (log-transformed for modeling); car age and engine power show the strongest relationships with price (r ≈ −0.71 and 0.65).
4. **Modeling** — OLS taken through a full diagnostic pipeline before trusting its coefficients (heteroscedasticity found → robust HC3 standard errors used; four statistically insignificant variables jointly dropped via partial F-test). Random Forest validated across 40 random seeds and 10-fold CV, then hyperparameter-tuned via grid search.
5. **Result** — Random Forest is recommended for prediction accuracy; OLS is retained for interpretable, statistically defensible coefficient effects (e.g., "each year of age costs ~11% of value").

## Author

Kelly — Data Science Track, OASIS INFOBYTE SIP
