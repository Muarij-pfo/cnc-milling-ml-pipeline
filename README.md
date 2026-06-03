# CNC Milling Process Monitoring — ML Pipeline

Machine learning pipeline for predicting machining process states and spindle output power from CNC milling sensor data.

---

## Overview

This project applies supervised machine learning to real-time CNC sensor data to:
- Classify the active machining process (9 classes: drilling, milling, idle, etc.)
- Predict machine speed state (3 classes: Low / Medium / High)
- Predict spindle output power (regression)

The dataset is sourced from the [UC Irvine CNC Mill Tool Wear Dataset](https://archive.ics.uci.edu/ml/datasets/CNC+Mill+Tool+Wear).

---

## Results Summary

| Task | Model | Metric | Score |
|---|---|---|---|
| Machining Process Classification | Random Forest | Accuracy | 65.6% |
| Machine Speed State Classification | Random Forest | Accuracy | 100% |
| Spindle Power Regression | Random Forest | R² | 0.9999 |
| Spindle Power Regression | Random Forest | Relative RMSE | 0.37% |

Statistical significance confirmed via Wilcoxon signed-rank test (p = 0.00195).

---

## Project Structure

```
cnc-milling-ml-pipeline/
│
├── dataset/          # Raw and cleaned sensor data
├── code/             # Preprocessing, feature engineering, models
├── report/           # Final project report (PDF)
└── README.md
```

---

## Models Used

**Classification**
- Logistic Regression
- Support Vector Machine (Linear kernel)
- Random Forest (200 trees)

**Regression**
- Linear Regression
- Random Forest Regressor (100 trees)
- Support Vector Regressor (SVR)

---

## Methodology

- Missing values imputed with column means
- Outlier removal using Z-score threshold (|z| > 3 on >5% of features)
- Features standardized using zero-mean, unit-variance scaling
- Two engineered features: Resultant Velocity and Calculated Spindle Power
- Stratified 70/15/15 train/validation/test split
- Hyperparameters tuned via 5-fold cross-validation

---

## Key Findings

- Random Forest significantly outperforms linear models on non-linear CNC data
- Linear Regression and Random Forest achieve near-identical regression performance (R² > 0.9998), indicating a predominantly linear relationship between features and spindle power
- SVR fails without proper kernel tuning (R² = 0.164)
- Top predictive features: S1_CommandPosition, S1_ActualPosition, Y1_CurrentFeedback

---

## Authors

- Zil Muarij (456149)
- Ayman Arshad (479690)
- Khwaja Qais (475156)
- Abdullah Naeem (465448)

Department of Mechanical Engineering, CEME — National University of Sciences and Technology (NUST), Rawalpindi, Pakistan

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Cortes, C. & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Dua, M. & Graff, C. (2019). CNC Mill Tool Wear Dataset. UCI Machine Learning Repository.
