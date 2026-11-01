# Loan Approval Prediction

A binary classification project that predicts whether a loan application will be **Approved** or **Rejected**, comparing logistic regression, a decision tree and a random forest. It uses the [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) from Kaggle (4,269 applications, MIT licence).

> **Important:** the dataset's owner describes it as *"a computer and self generated Dataset. None of the values depict real life valuations."* The results below describe how these methods behave on this synthetic data. They say little about real lending.

## What the notebook does

`loan-prediction.ipynb` runs from top to bottom in one pass:

1. Records the environment (Python and library versions).
2. Downloads the dataset into `datasets/` and cleans it (strips whitespace, checks for missing values, drops `loan_id`).
3. Exploratory data analysis: summary statistics, data-quality checks, approval rates by education, employment and CIBIL-score band, and an overview figure.
4. Encodes the target (`loan_status`: Approved = 1, Rejected = 0) and inspects each numeric attribute's correlation with it (no attribute is removed on this basis).
5. Makes a stratified 80/20 train/test split (seed 42) and builds a preprocessing pipeline (one-hot encoding for the categorical attributes, numeric passthrough).
6. Fits a **majority-class baseline**.
7. Tunes three classifiers with `RandomizedSearchCV` (50 draws, stratified 5-fold CV, F1 scoring):
   - Logistic regression (with scaling)
   - Decision tree
   - Random forest
8. Compares the tuned models and the baseline on the test set, with confusion-matrix counts and false-approval / false-rejection rates.
9. Post-hoc analyses on the final model:
   - an illustrative cost-sensitive decision threshold (the 5:1 false-approval to false-rejection cost ratio is an assumption, not taken from lending data);
   - stability: 5-fold CV repeated 10 times, and 10 different train/test splits with hyperparameters held fixed;
   - permutation feature importance;
   - a group-wise screening on `education` and `self_employed` (the dataset has no protected attributes such as gender or age, so this is not a fairness audit).

## Results (test set, 854 applications)

| Model | F1 | Precision | Recall | ROC AUC | False approvals | False rejections |
|---|---|---|---|---|---|---|
| Logistic regression | 0.9446 | 0.9759 | 0.9153 | 0.9738 | 12 | 45 |
| Decision tree | 0.9850 | 0.9795 | 0.9906 | 0.9891 | 11 | 5 |
| Random forest | 0.9916 | 0.9869 | 0.9962 | 0.9987 | 7 | 2 |
| Majority-class baseline | 0.7668 | 0.6218 | 1.0000 | 0.5000 | 323 | 0 |

The same ordering (random forest, decision tree, logistic regression) holds under repeated cross-validation and over ten alternative splits, although the forest-tree gap is small. Permutation importance shows the models rely overwhelmingly on `cibil_score`: approval rates jump from about 10% to about 99.5% at a score of 550, so the label is largely a threshold function of one variable. See the notebook outputs for the full numbers.

Figures are saved to `figures/`:

- `eda_overview.png`: class balance, CIBIL score by outcome, approval rate by CIBIL band
- `confusion_matrices.png`: test-set confusion matrices of the three tuned models
- `permutation_importance.png`: permutation importance for the random forest

## Reproducing the results

The reported run used Python 3.11.16 on macOS (arm64) with the library versions pinned in `requirements.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab loan-prediction.ipynb
```

Run all cells in order. The first code cell downloads the dataset from Kaggle's public download endpoint (network access required) and unpacks it into `datasets/`, which is git-ignored. Random seeds are fixed at 42 for the split, the folds and the estimators, but exact numbers can differ slightly with other library versions or platforms.

## Project structure

```
.
├── loan-prediction.ipynb   # analysis and modelling
├── figures/                # figures produced by the notebook
├── datasets/               # downloaded dataset (created on first run, git-ignored)
├── requirements.txt        # pinned library versions
└── README.md
```

## Limitations

- The data are synthetic, and the near-perfect scores mainly reflect one threshold-like variable.
- The main comparison uses a single split and a single seed; the stability checks help but overlap the data used for tuning.
- The search budget is modest, and the cost ratio in the threshold analysis is assumed.
- Only three model families are compared.
