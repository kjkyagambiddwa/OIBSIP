# OIBSIP — Data Science Internship

**Author:** Kelly
**Program:** Oasis Infobyte Summer Internship Program (OIBSIP)
**Track:** Data Science

---

## Overview

This repository contains the three projects completed for the Data Science track of the Oasis Infobyte Summer Internship Program. Each project folder includes a Jupyter notebook, a written analysis report, a requirements file, and a demo video link.

The projects were chosen to cover a spread of core data science work: multi-class classification, regression with heavy data cleaning, and regression with feature selection and model comparison.

---

## Projects

### Task 1 — Iris Flower Classification

Classify iris flowers into one of three species (*setosa*, *versicolor*, *virginica*) from four physical measurements, and compare four classifiers (Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest).

**Highlights:**
- Showed that a single train/test split can produce misleading ties; reran all models across 40 random seeds to get a stable ranking.
- Tested the common assumption that feature scaling always helps. It did not. Scaling hurt Logistic Regression and KNN, while leaving tree-based models unchanged, because petal length's larger raw variance was correctly weighting the more informative feature.

**Folder:** [`DataScience-Task1-IrisClassification`](./DataScience-Task1-IrisClassification)
**Notebook:** `Iris_Classification.ipynb` · **Report:** `Iris_Report.md`

---

### Task 3 — Car Price Prediction

Predict the selling price of a used car from features like brand, age, mileage, fuel type, transmission, and ownership history, using a real-world CarDekho listings dataset (8,128 raw rows).

**Highlights:**
- The bulk of the work was data cleaning: removing 1,202 exact duplicate listings, parsing unit-embedded text fields (`"23.4 kmpl"`, `"1248 CC"`, `"74 bhp"`), converting mixed torque units, and handling a zero-value mileage anomaly.
- Compared an interpretable OLS model (with a full diagnostic pass: VIF, Breusch-Pagan, Jarque-Bera, HC3 robust standard errors) against a Random Forest that predicted more accurately but offered less direct interpretability.

**Folder:** [`DataScience-Task3-CarPricePrediction`](./DataScience-Task3-CarPricePrediction)
**Notebook:** `Car_Price_Prediction.ipynb` · **Report:** `Car_Price_Prediction_Report.md`

---

### Task 5 — Sales Prediction

Predict product sales from advertising expenditure across TV, Radio, and Newspaper channels, using the classic `Advertising.csv` dataset (200 observations).

**Highlights:**
- Progressed from a base linear model (R² 0.85) through a full polynomial + interaction model (R² 0.99), then used four feature-selection methods (Lasso, Ridge, Elastic Net, RFECV) to settle on a parsimonious 4-feature model.
- All four feature-selection methods independently agreed: TV and Radio expenditure reinforce each other, while Newspaper expenditure adds no measurable value. That is a finding a marketing team can act on directly.

**Folder:** [`DataScience-Task5-SalesPrediction`](./DataScience-Task5-SalesPrediction)
**Notebook:** `Sales_Prediction.ipynb` · **Report:** `sales_prediction_report.md`

---

## Demo Videos

Walkthroughs of each project are available as a YouTube playlist:

https://youtube.com/playlist?list=PLf1KiZ-gFAYw

---

## What Tied the Three Together

Across all three projects, the pattern was the same: **state the assumption, test it properly, and report what did not work as honestly as what did.**

A model that scores well on one split is not proof of anything. A model that holds up across 40 seeds, survives its own diagnostic checks, and comes with an honest account of where it fails is the one worth trusting.

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, statsmodels, SciPy, matplotlib, seaborn — all work done in Jupyter Notebooks.

---

## Repository Structure
OIBSIP/
├── DataScience-Task1-IrisClassification/
│ ├── Iris_Classification.ipynb
│ ├── Iris_Report.md
│ ├── README.md
│ └── requirements.txt
├── DataScience-Task3-CarPricePrediction/
│ ├── Car details v3.csv
│ ├── Car_Price_Prediction.ipynb
│ ├── Car_Price_Prediction_Report.md
│ ├── README.md
│ └── requirements.txt
├── DataScience-Task5-SalesPrediction/
│ ├── Advertising.csv
│ ├── Sales_Prediction.ipynb
│ ├── sales_prediction_report.md
│ ├── README.md
│ └── requirements.txt
├── .gitignore
└── README.md

text

---

## How to Reproduce

Each project folder contains its own notebook, dataset (where redistributable), and `requirements.txt`. To reproduce any project:

1. Navigate into the relevant task folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook and run it top to bottom (Kernel → Restart & Run All).

---

## Contact

**GitHub:** [@kjkyagambiddwa](https://github.com/kjkyagambiddwa)

---

## Acknowledgements

Thank you to Oasis Infobyte for a structure that made this kind of independent, hands-on learning possible.
What I corrected based on your screenshots
Item	From the screenshots
Task 1 report filename	Iris_Report.md (not REPORT.md)
Task 3 report filename	Car_Price_Prediction_Report.md (renamed from REPORT.md per your commit history)
Task 5 report filename	sales_prediction_report.md (lowercase, per GitHub view)
Task 1 notebook	Iris_Classification.ipynb
Task 3 notebook	Car_Price_Prediction.ipynb
Task 5 notebook	Sales_Prediction.ipynb
Each folder has	A README.md and requirements.txt of its own
Root has	.gitignore and no root README yet (that's what this file becomes)
Dataset files present	Advertising.csv (Task 5), Car details v3.csv (Task 3). Task 1 uses sklearn.datasets.load_iris(), so no CSV.
One thing to double-check before committing
Your Task 3 folder shows a .csv file named Car details v3.csv (with spaces). In the notebook, the code that loads it needs to reference that exact filename (with the space) or use a path-safe equivalent. If the notebook currently uses Car_details_v3.csv or a different name, the "How to Reproduce" step will fail for anyone cloning the repo. Worth a quick check.
