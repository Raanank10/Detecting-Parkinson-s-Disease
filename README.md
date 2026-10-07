# Detecting Parkinson's Disease from Voice: a Leakage-Aware Evaluation

**Python · scikit-learn · XGBoost** | Can voice features flag Parkinson's disease in a person the model has never heard before?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Raanank10/Detecting-Parkinson-s-Disease/blob/main/Detecting_Parkinsons_Disease.ipynb)

## The short version

My original 2021 version of this project reported **~97% accuracy**. When I revisited it, I found the number was inflated by **data leakage**:

1. **Patient overlap.** The dataset has 195 recordings from only **32 people**. A random row split put recordings of the same person in both train and test, so the model learned to recognize *voices*, not *disease*.
2. **An ID column used as a feature.** The encoded recording name contains the subject ID, and it was the model's #2 most important feature.

I rebuilt the evaluation with **patient-level cross-validation**, so every test fold contains only unseen people:

| Setup | Model | ROC-AUC | Balanced accuracy | Sensitivity |
|---|---|---|---|---|
| Leaky (row split + ID feature) | XGBoost | 0.99 ± 0.01 | 0.95 ± 0.05 | 0.98 ± 0.02 |
| **Patient-level CV** | Random Forest | **0.80 ± 0.10** | 0.66 ± 0.17 | 0.89 ± 0.06 |
| **Patient-level CV** | XGBoost | **0.78 ± 0.09** | 0.67 ± 0.11 | 0.97 ± 0.03 |
| **Patient-level CV** | Logistic Regression | **0.75 ± 0.12** | 0.63 ± 0.17 | 0.77 ± 0.05 |

*4-fold StratifiedGroupKFold, grouped by subject. Mean ± std across folds.*

## What this shows

- **Evaluation design changes the answer.** The same models drop from near-perfect to a moderate ROC-AUC of about 0.8 once the test set contains only new people. That is the realistic estimate for a screening use case.
- **Accuracy is misleading on this data.** 75% of recordings are PD, so a model that always says "PD" already scores 75% accuracy. I report ROC-AUC, balanced accuracy and sensitivity instead.
- **The models over-call PD.** Sensitivity is high but specificity is low, so many healthy people get flagged.
- **Sample size is the real limit.** There are only 8 healthy subjects, so one misclassified person swings fold metrics by double digits. The result is a feasibility signal, not a diagnostic claim.
- **Signal sits in nonlinear dysphonia measures** (PPE, spread1), which is consistent with the original study.

## Data

[UCI Parkinson's dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) (Little et al., 2008). It has 195 sustained-phonation recordings from 32 subjects (24 PD, 8 healthy) and 22 acoustic features: jitter, shimmer, HNR/NHR, RPDE, DFA, D2, PPE and spread measures.

## Repository

| File | Contents |
|---|---|
| `Detecting_Parkinsons_Disease.ipynb` | Current analysis: leaky vs. patient-level evaluation side by side |
| `parkinsons.csv` | Raw dataset |
| `archive/2021_original_row_split.ipynb` | Original 2021 notebook, kept for transparency. It started from a DataFlair tutorial |

**Run it:** open it in Colab with the badge above, or run `pip install pandas scikit-learn xgboost matplotlib` and then `jupyter notebook`.

## Next steps

- Aggregate predictions per person (majority vote across each person's recordings).
- Tune the decision threshold for a target specificity.
- Validate on an external cohort before making any clinical claim.

---
**Raanan Kelner**, B.Sc. Medical Engineering · Data Analyst · [LinkedIn](https://www.linkedin.com/in/raanan-kelner)
