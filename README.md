# Loan Approval Prediction

A binary classification project that predicts whether a loan application will be **Approved** or **Rejected**, using the [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) from Kaggle.

## What the notebook does

`loan-prediction.ipynb` walks through the full workflow:

1. Downloads and extracts the dataset into `datasets/`
2. Cleans the data (strips whitespace from column names and values) and explores it
3. Encodes the target (`loan_status`: Approved = 1, Rejected = 0) and checks feature correlations (`cibil_score` is the strongest)
4. Makes a stratified 80/20 train/test split
5. Builds a preprocessing pipeline (one-hot encoding for categoricals, numeric passthrough)
6. Tunes three classifiers with `RandomizedSearchCV` and stratified 5-fold CV:
   - Logistic Regression
   - Decision Tree
   - Random Forest
7. Compares the best models on the test set (precision, recall, F1, ROC AUC) and plots confusion matrices

## Getting started

```bash
pip install pandas scikit-learn scipy matplotlib jupyter
jupyter notebook loan-prediction.ipynb
```

Run the cells in order. The first cell downloads the dataset automatically (a Kaggle download may require network access).

## Project structure

```
.
├── loan-prediction.ipynb   # analysis and modelling
├── datasets/               # downloaded dataset (created on first run)
└── README.md
```
