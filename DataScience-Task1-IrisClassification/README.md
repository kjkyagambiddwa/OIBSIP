# Task 1 — Iris Flower Classification

**Objective:** Train a machine learning classification model to identify the species of an iris flower (Setosa, Versicolor, or Virginica) from its physical measurements.

**Tech stack:** Python, scikit-learn, pandas, matplotlib, seaborn, Jupyter Notebook

**Status:** Complete — EDA, visualizations, modeling, and evaluation done, including an extended robustness investigation.

## Approach
1. Data quality check (no missing values)
2. EDA — per-species descriptive statistics, pairplots, box plots, correlation heatmap
3. Feature selection — petal length + sepal width chosen, with the multicollinearity trade-off discussed
4. Modeling — 4 classifiers trained (Logistic Regression, KNN, Decision Tree, Random Forest) on an 80/20 stratified split
5. Evaluation — accuracy, balanced accuracy, confusion matrix, classification report
6. Extended: tested the feature-selection trade-off across 40 random seeds, with and without feature scaling, to check whether the reasoning held up

## Results
Best single-seed performance: a 4-way tie at 96.7% accuracy (KNN, Logistic Regression, Random Forest on petal length + petal width; KNN also on petal length + sepal width). Averaged across 40 seeds, Logistic Regression led on unscaled features; Random Forest led after scaling. Scaling reduced accuracy for KNN and Logistic Regression but had negligible effect on the tree-based models — a useful reminder that feature scaling isn't a universal best practice.

## Contents
- `Iris_Classification.ipynb` — full analysis notebook