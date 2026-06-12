# Multivitamin Treatment Effectiveness in Pregnant Women

An end-to-end machine learning project predicting the effectiveness of multivitamin supplementation across the three trimesters of pregnancy, based on changes in key health biomarkers (Haemoglobin, Vitamin D, Folic Acid, B12, Fatigue).

---

## Overview

Multivitamin supplementation during pregnancy aims to address maternal nutritional deficiencies. This project asks: **can we predict, from biomarker data, how effective the supplementation will be for a given patient?**

Three classifiers — Logistic Regression, Random Forest, and Gradient Boosting — are trained and compared on a multi-class effectiveness target. The most informative features are engineered **delta features** that capture how much each biomarker changes after supplementation.

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Notebook:** Jupyter

---

## Methodology

1. **Exploratory Data Analysis** — distribution of the effectiveness target, biomarker changes by trimester, and correlation heatmap.
2. **Feature Engineering** — five delta features (`Hb_Delta`, `VitD_Delta`, `Folic_Delta`, `B12_Delta`, `Fatigue_Delta`) capturing post-treatment changes; label-encoded target; binary-encoded compliance.
3. **Data Preparation** — 80/20 stratified train/test split with `StandardScaler` normalization.
4. **Modeling** — Logistic Regression (linear baseline), Random Forest, and Gradient Boosting classifiers.
5. **Evaluation** — accuracy, per-class precision/recall/F1, confusion matrix, and feature importance.
6. **Class Imbalance Experiment** — reframed as binary classification with downsampled majority class to test whether balancing improves performance.

---

## Results

| Model               | Test Accuracy |
| ------------------- | ------------- |
| Logistic Regression | **52%**       |
| Random Forest       | 45%           |
| Gradient Boosting   | 43%           |

**Key takeaways:**

- The simple linear baseline (Logistic Regression) outperformed both tree ensembles, suggesting the underlying signal is largely linear in the engineered features and the more complex models are overfitting on the small sample (500 patients).
- Engineered delta features (`Hb_Delta`, `Fatigue_Delta`) ranked highest in feature importance — consistent with clinical intuition that *change* in a biomarker is more diagnostic than its absolute level.
- All models struggled with the minority *Low Effectiveness* class — a class imbalance challenge that downsampling alone did not resolve.

The modest accuracy is an *honest* outcome on a small, imbalanced medical dataset. The transferable contribution is the methodology: clean pipeline, multiple model comparison, and proper evaluation.

---

## Repository Contents

```
.
├── Multivitamin_Effectiveness_ML.ipynb   # Main analysis notebook
├── multivitamin_pregnancy_data.csv       # Dataset (500 synthetic patient records)
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/multivitamin-effectiveness-ml.git
cd multivitamin-effectiveness-ml

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Launch Jupyter
jupyter notebook Multivitamin_Effectiveness_ML.ipynb
```

---

## Future Work

- Acquire a larger, multi-site dataset for stronger generalization.
- Cross-validated hyperparameter search (GridSearchCV / Optuna).
- Class-weighting or SMOTE oversampling for minority-class recall.
- Interaction terms between trimester and biomarker deltas.
- Benchmark against LightGBM and XGBoost.
- Calibrated probability outputs for clinical decision-support use.

---

## Author

**Charu Bajaj**  
PhD Candidate, Mathematics — Amity University, Noida  
M.Sc. Applied Mathematics (9.29 CGPA)  
📧 charub98@gmail.com
