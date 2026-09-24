# Iris Flower Classification

**Oasis Infobyte SIP — Data Science Track — Task 1**

Classifying iris flowers into species from petal and sepal measurements, and comparing multiple classification models — plus testing whether the feature-selection reasoning actually held up under repetition.

## Objective

Build a classification model that predicts an iris flower's species (*setosa*, *versicolor*, or *virginica*) from its physical measurements, and identify the best-performing model.

## Dataset

- **Source:** The classic Iris dataset, loaded via `sklearn.datasets.load_iris()`.
- **Size:** 150 records, 4 numeric features (sepal length, sepal width, petal length, petal width), 3 perfectly balanced classes (50 samples each).

## Project Structure

```
DataScience-Task1-IrisClassification/
├── Iris_Classification.ipynb
├── REPORT.md
└── README.md
```

## Approach

1. Loaded the dataset and confirmed no missing values.
2. Built a combined dataframe with a species column and ran EDA: per-species descriptive statistics, pairplot, box plots, and a correlation heatmap.
3. Selected petal length and sepal width as modeling features — chosen specifically over petal width to reduce the 0.96 multicollinearity between the two petal measurements.
4. Trained and compared four classifiers — Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest — on an 80/20 stratified train/test split.
5. Evaluated every model with accuracy, balanced accuracy, confusion matrix, and classification report.
6. Extended the analysis: retested the feature-selection trade-off (petal length + sepal width vs. petal length + petal width) across 40 random seeds to check whether the multicollinearity reasoning cost predictive accuracy.
7. Repeated the 40-seed test with `StandardScaler` applied, to test the hypothesis that scaling would fix petal length's variance-driven dominance in distance/coefficient calculations.

See [`Iris_Report.md`](./Iris_Report.md) for the full write-up, including all findings, tables, and reasoning behind each conclusion.

## Results

| Feature Pair | Model | Accuracy (40-seed avg, unscaled) |
|---|---|---|
| petal length + petal width | Logistic Regression | 0.966 |
| petal length + petal width | K-Nearest Neighbors | 0.963 |
| petal length + petal width | Random Forest | 0.958 |
| petal length + sepal width | Logistic Regression | 0.952 |
| petal length + sepal width | K-Nearest Neighbors | 0.948 |
| petal length + petal width | Decision Tree | 0.945 |
| petal length + sepal width | Random Forest | 0.938 |
| petal length + sepal width | Decision Tree | 0.925 |

## Key Insights

- **Logistic Regression** was the strongest model overall once averaged across 40 random seeds — a single-seed run had shown a 4-way tie at 96.7%, which only resolved once tested more robustly.
- **Petal length + petal width outperformed petal length + sepal width for all four models**, meaning the multicollinearity-avoidance trade-off had a real, if modest, accuracy cost — low correlation between two features doesn't guarantee both carry useful signal.
- **Feature scaling reduced accuracy** for Logistic Regression and KNN (both sensitive to feature magnitude) but had virtually no effect on Decision Tree or Random Forest (both split on feature order, not magnitude) — scaling is not a universal best practice, only appropriate when a feature's raw scale is arbitrary rather than meaningful.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## How to Run

1. Clone this repository and navigate to this folder.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open `Iris_Classification.ipynb` and run all cells (Kernel → Restart & Run All).

---

*Part of the Oasis Infobyte Summer Internship Program (Data Science Track).*
