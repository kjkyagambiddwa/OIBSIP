# Car Price Prediction with Machine Learning
### OASIS INFOBYTE — Data Science Internship, Task 3
**Author:** Kelly

---

## 1. Objective

Build a regression model that predicts the selling price of a used car from features such as brand, age, mileage, fuel type, transmission, and ownership history, using a real-world listings dataset scraped from CarDekho. The project compares a classical linear (OLS) approach against a Random Forest ensemble, with the linear model additionally subjected to a full set of regression-assumption diagnostics rather than being fit and reported blind.

## 2. Dataset

**Source:** CarDekho used-car listings (`Car details v3.csv`), 8,128 rows, 13 raw columns: `name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats`.

This is the richest of several CarDekho-branded dataset variants available on Kaggle under the same listing — it was chosen over smaller variants specifically because it includes `mileage`, `engine`, `max_power`, `torque`, and `seats`, giving the model genuine vehicle-performance signal rather than relying on listing metadata alone.

## 3. Data Cleaning

### 3.1 Duplicate listings

1,202 rows (14.79% of the dataset) were exact duplicates across every column — including `km_driven` and `selling_price`, which are close to continuous and effectively impossible to match by coincidence across two genuinely different cars. These were judged to be the same listing recorded more than once (re-scrapes or repeated dealer postings) rather than distinct vehicles, and were dropped to prevent the same data point being counted multiple times and to avoid train/test leakage. This left **6,926 rows**.

### 3.2 Missing values

`mileage`, `engine`, `max_power`, `torque`, and `seats` shared the same block of ~208–209 missing rows (~3.0% of the deduplicated data) — the same incomplete listings were missing all five fields together, rather than missingness being scattered randomly. At under 3%, dropping these rows was judged to have minimal effect on model formulation, leaving **6,717 rows**.

### 3.3 Categorical consistency

`fuel`, `seller_type`, `transmission`, and `owner` were checked for casing/spelling inconsistencies (e.g. "Petrol" vs. "petrol"). None were found — this dataset happens to be clean on these specific fields, contrary to the common assumption that such inconsistencies are guaranteed.

### 3.4 Unit-mixing in mileage, and a zero-value anomaly

`mileage`, `engine`, `max_power`, and `torque` were stored as text with embedded units (`"23.4 kmpl"`, `"1248 CC"`, `"74 bhp"`, `"190Nm@ 2000rpm"`) rather than as numbers, and were parsed out via regex extraction. `torque` was the messiest of the four, mixing `Nm` and `kgm` units with inconsistent RPM formatting — `kgm` values were converted to Nm using the standard 9.80665 conversion factor.

`mileage` carried a second, more subtle problem: **two incompatible units**. Petrol/Diesel cars report mileage in `kmpl` (kilometers per liter), while CNG/LPG cars report it in `km/kg` (kilometers per kilogram of gas) — not directly comparable measurements. A `groupby` comparison showed the two groups have overlapping but distinct ranges (km/kg: mean 21.9, n=86; kmpl: mean 19.4, n=6,631) — a real but modest ~13% average difference, not large enough to ignore, not large enough to demand a full unit-conversion model either. A binary flag, `mileage_is_gas_unit`, was created to let the model distinguish the two measurement systems rather than treating them as interchangeable.

That same range check also surfaced an anomaly: a minimum `mileage_value` of exactly **0.0** among the kmpl group. Since this dataset has no Electric category (only Diesel/Petrol/LPG/CNG), a manufacturer-rated fuel efficiency of zero doesn't correspond to any real vehicle type — it's a data-entry artifact. Fifteen rows were affected (10 Petrol, 5 Diesel — no concentration in either fuel type, supporting the "isolated entry error" read rather than a systematic scraping bug), and these were dropped, leaving a final cleaned dataset of **6,702 rows**.

**A note on `mileage_is_gas_unit`:** when checked against Variance Inflation Factor (VIF) during the multicollinearity pass, this flag turned out to be perfectly collinear with the existing `fuel_Other` dummy (CNG/LPG cars are exactly the cars measured in km/kg) — the two columns encoded the identical partition of rows. It was therefore dropped from the final modeling feature set; no information is lost, since `fuel_Other` already carries that signal.

## 4. Feature Engineering

- **`car_age`** — computed as `2026 − year` (reference year fixed for reproducibility, rather than computed dynamically at run time).
- **`brand`** — extracted from the first token of the `name` column (e.g. `"Maruti Swift Dzire VDI"` → `"Maruti"`).

## 5. Exploratory Data Analysis

- **Selling price distribution:** strongly right-skewed — most listings cluster at modest prices with a long tail of expensive vehicles. This motivated a **natural log transform** of the target (`log_selling_price`), which is markedly closer to symmetric and is the actual target variable used for both models below.
- **Price vs. fuel type:** boxplots (on both raw and log-transformed price) show diesel and CNG/LPG vehicles occupying different price bands than petrol, reflecting both vehicle segment and residual value trends associated with fuel type.
- **Price vs. car age:** a clear depreciation curve — price falls sharply in the first several years of age and flattens out later, a nonlinear pattern that a plain linear model can only approximate.
- **Correlation heatmap** (on the final encoded feature set): `car_age` shows the strongest relationship with the target (r ≈ −0.71 — older cars sell for less), followed by `max_power_value` (r ≈ 0.65 — more powerful cars sell for more). The heatmap also flagged the fuel dummy variables as a multicollinearity concern from one-hot encoding, addressed by dropping `fuel_Diesel` as the reference category and combining the minority CNG/LPG groups into an `"Other"` fuel category to avoid sparse dummy features.

## 6. Encoding

| Feature | Method | Rationale |
|---|---|---|
| `fuel`, `seller_type`, `transmission` | One-hot (drop-first) | Low cardinality, no ordinal relationship |
| `owner` | Label encoding | Genuine ordinal structure (First Owner < Second Owner < ...) |
| `brand` | Label encoding | High cardinality — one-hot would explode dimensionality |

CNG and LPG were merged into a single `"Other"` fuel category before encoding to avoid sparse minority dummies.

## 7. Linear Regression: Assumption Checks & Model Selection

Rather than fitting one OLS model and reporting its R², the linear model was put through a full diagnostic pass before its coefficients were trusted for interpretation.

### 7.1 Multicollinearity (VIF)

| Feature | VIF |
|---|---|
| engine_volume | 5.68 |
| mileage_value | 3.38 |
| max_power_value | 3.26 |
| fuel_Petrol | 2.64 |
| seats | 2.34 |
| torque_nm | 2.02 |
| car_age | 1.96 |
| km_driven | 1.37 |
| transmission_Manual | 1.34 |
| owner | 1.28 |
| seller_type_Individual | 1.13 |
| brand | 1.07 |
| fuel_Other | 1.06 |
| seller_type_Trustmark Dealer | 1.04 |

`engine_volume` sits just above 5, indicating moderate but not severe multicollinearity (the conventional concern threshold is VIF = 10). No evidence of severe multicollinearity across the feature set.

### 7.2 Autocorrelation of residuals (Durbin-Watson)

Durbin-Watson statistic: **1.818** — close enough to the "no autocorrelation" benchmark of 2 that there's no sufficient evidence of autocorrelated residuals.

### 7.3 Linearity

A residuals-vs-fitted-values plot showed residuals randomly scattered around zero with no systematic pattern — the linearity assumption holds.

### 7.4 Homoscedasticity (Breusch-Pagan)

LM statistic 616.67, p < 0.0001 — the null hypothesis of constant error variance is rejected. The model's residual variance is **not constant** (heteroscedastic), so all subsequent OLS inference uses **heteroscedasticity-robust (HC3) standard errors** rather than the classical (non-robust) ones.

### 7.5 Normality of residuals (Jarque-Bera)

Jarque-Bera statistic 884.41, p ≈ 9 × 10⁻¹⁹³ — the null hypothesis of normally distributed residuals is rejected. Given the large sample size (n = 6,702), the Central Limit Theorem means the OLS coefficient estimates remain approximately normally distributed regardless, so this does not undermine the coefficient-level inference below.

### 7.6 Variable selection: confounder and joint-significance testing

The full 14-feature robust model flagged four variables as candidates for removal on statistical grounds: `torque_nm` and `km_driven` appeared to be overshadowed by `car_age`/`max_power_value`/`engine_volume`; `brand` added little once vehicle specs were already in the model; and `seller_type_Trustmark Dealer` was not statistically distinguishable from the standard-dealer baseline category. Before dropping them, coefficient-shift comparisons (full model vs. each candidate reduction) showed no variable crossing the 15%-change confounder threshold, and a joint Wald partial F-test across all four candidates returned **F = 1.50, p = 0.199** — not significant, confirming they can be dropped together without materially harming the model.

### 7.7 Final OLS model

The final specification retains 10 predictors: `owner, seats, mileage_value, engine_volume, max_power_value, car_age, fuel_Other, fuel_Petrol, seller_type_Individual, transmission_Manual` (target: `log_selling_price`, HC3 robust standard errors, n = 6,702).

| Predictor | Coefficient | p-value |
|---|---|---|
| const | 12.926 | < 0.001 |
| owner | −0.0305 | < 0.001 |
| seats | 0.0407 | < 0.001 |
| mileage_value | 0.0134 | < 0.001 |
| engine_volume | 0.0002 | < 0.001 |
| max_power_value | 0.0096 | < 0.001 |
| car_age | −0.1113 | < 0.001 |
| fuel_Other | −0.1263 | < 0.001 |
| fuel_Petrol | −0.1244 | < 0.001 |
| seller_type_Individual | −0.0995 | < 0.001 |
| transmission_Manual | −0.1612 | < 0.001 |

R² = 0.842 (Adj. R² = 0.842), F-statistic = 2,348 (p < 0.001) — the model explains 84.2% of the variance in log-transformed selling price, and every retained predictor is statistically significant.

**Interpretation (converting log-scale coefficients to approximate percentage effects):**

- **Vehicle age** is the single strongest negative driver: each additional year reduces the estimated selling price by roughly **11.1%**, holding everything else constant.
- **Manual transmission** vehicles sell for roughly **15.0% less** than automatics (the reference category).
- Cars sold by **individual private sellers** go for about **9.5% less** than those sold through professional dealers.
- **Petrol** cars sell for roughly **12.4–12.75% less** than diesel cars (the omitted reference fuel category).
- Each additional **seat** adds approximately **4.1%** to estimated price; each **km/l** of fuel efficiency improvement, about **1.3%**; each additional **brake horsepower**, about **0.95%**; each additional **CC** of engine capacity, about **0.02%**.
- Moving down each successive **ownership tier** (e.g. First → Second Owner) reduces price by roughly **3.0%**.

The primary positive price drivers are engine power and vehicle capacity (`max_power_value`, `engine_volume`, `seats`); the strongest penalties come from vehicle age, manual transmission, petrol fuel, and individual (non-dealer) sellers.

## 8. Random Forest Regressor

### 8.1 Baseline model

A baseline `RandomForestRegressor` (100 trees, `random_state=256`) was trained on an 80/20 split using the full 14-feature set (including `torque_nm`, `brand`, `seller_type_Trustmark Dealer`, and `km_driven` — variables the OLS model dropped for statistical insignificance, but which tree ensembles can still exploit for nonlinear splits and interactions).

| Metric | Value |
|---|---|
| Test R² | 0.9040 |
| Test RMSE | 0.2321 |
| Test MAE | 0.1655 |

### 8.2 Feature importance

A Gini-importance bar chart was generated from the baseline model (see `Car_Price_Prediction.ipynb`, "Feature Importance Chart" section). Consistent with the correlation and OLS findings above, `car_age` and `max_power_value` are the two dominant predictors — the same two features that showed the strongest linear correlations with price (−0.71 and 0.65 respectively) also carry the most weight in the tree ensemble's splits, reinforcing that vehicle age and engine power are the two properties driving used-car pricing most consistently across both modeling approaches.

### 8.3 Cross-validation and hyperparameter tuning

To validate the baseline result wasn't an artifact of one particular 80/20 split, evaluation was repeated across **40 random seeds** and separately via **10-fold cross-validation**:

| Evaluation | R² | RMSE | MAE |
|---|---|---|---|
| 40-seed average (out-of-sample) | 0.9121 ± 0.0049 | 0.2217 ± 0.0057 | 0.1603 ± 0.0034 |
| 10-fold CV average | 0.9127 [95% CI: 0.9083, 0.9172] | 0.2208 [95% CI: 0.2145, 0.2272] | 0.1603 [95% CI: 0.1568, 0.1637] |

Both robustness checks agree closely with the single-split baseline and with each other — R² stays in a tight band around 0.91 regardless of which 20% of the data is held out, indicating the model's performance is stable rather than lucky.

A grid search (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, evaluated via the same 10-fold CV) selected `{max_depth: 20, max_features: 'sqrt', min_samples_leaf: 1, min_samples_split: 5, n_estimators: 200}` as the best configuration, achieving a **best CV R² of 0.9200** — a modest further gain over the untuned baseline.

## 9. Model Comparison

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Optimized Random Forest (best CV score)** | **0.9200** | 0.2208 | 0.1603 |
| Optimized Random Forest (10-fold CV average) | 0.9127 [0.9083, 0.9172] | 0.2208 [0.2145, 0.2272] | 0.1603 [0.1568, 0.1637] |
| Random Forest (40-seed average) | 0.9121 ± 0.0049 | 0.2217 ± 0.0057 | 0.1603 ± 0.0034 |
| Random Forest (single seed 256) | 0.9040 | 0.2321 | 0.1655 |
| OLS Linear Regression (full data, in-sample) | 0.8422 | 0.2977 | 0.2272 |
| OLS Linear Regression (40-seed out-of-sample average) | 0.8396 [0.8371, 0.8421] | 0.2993 [0.2972, 0.3015] | 0.2277 [0.2264, 0.2289] |
| OLS Linear Regression (single seed 256) | 0.8242 | 0.3141 | 0.2371 |

Random Forest outperforms OLS across every metric and every evaluation scheme, by a wide and consistent margin (~0.07–0.08 R², a ~26% reduction in RMSE, a ~29% reduction in MAE at the fully-tuned level). This is consistent with the EDA finding that price-vs-age follows a nonlinear depreciation curve that a purely additive linear model structurally cannot capture, while a tree ensemble can.

## 10. Conclusions and Recommendations

- **For predictive accuracy / deployment:** the optimized Random Forest Regressor is the clear choice, delivering R² ≈ 0.91–0.92 out-of-sample with tight, stable confidence intervals across both 40-seed and 10-fold validation — meaningfully more accurate than the linear baseline at every metric.
- **For explaining *why* a car is priced the way it is:** the OLS model, despite its lower predictive accuracy, remains valuable — its HC3-robust coefficients give statistically defensible, interpretable percentage effects for each feature (e.g. "each year of age costs ~11% of value"), which the Random Forest's feature-importance ranking cannot provide on its own.
- **Vehicle age and engine power** are the two most consistent price drivers across both modeling approaches — confirmed independently by the correlation heatmap, the OLS coefficients, and the Random Forest feature importances.
- **Practically:** the two models serve complementary purposes — Random Forest for accurate valuation, OLS for structural interpretation and hypothesis testing about *which* features move price and by how much.

## 11. Limitations and Future Work

- **Mileage unit mixing:** the `mileage_is_gas_unit` flag lets the model distinguish kmpl-rated from km/kg-rated vehicles, but a plain linear model only absorbs this as an intercept shift (via the collinear `fuel_Other` dummy) rather than a true slope correction — an interaction term (`mileage_value × fuel_Other`) would model this more precisely, at the cost of added complexity, and was left as a documented simplification rather than implemented.
- **Brand encoding:** `brand` was label-encoded (an arbitrary integer per brand) for practicality given its high cardinality. This is a defensible compromise for tree-based models but is not ideal for a linear model, where it implies a spurious ordinal relationship between brands — this is one reason `brand` was ultimately dropped from the final OLS specification rather than interpreted directly.
- **Reference year:** `car_age` is computed against a fixed 2026 reference year rather than the current system date, so the notebook's outputs are reproducible on any future re-run rather than silently drifting with the calendar.
- **Model scope:** hyperparameter search covered a modest grid; a wider search (or gradient-boosted alternatives such as XGBoost/LightGBM) could plausibly push Random Forest's accuracy further, but was outside this task's required scope.
