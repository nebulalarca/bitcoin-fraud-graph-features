#  Anti-Money Laundering (AML) & Bitcoin Fraud Detection on Elliptic Dataset

This repository contains an end-to-end Machine Learning pipeline to detect fraudulent transactions on the Bitcoin blockchain using the **Elliptic Data Set**. The project addresses real-world financial challenges including severe class imbalance, graph feature extraction, and temporal data leakage.

---

##  Dataset Overview

* **Source:** Elliptic Data Set (203,769 Bitcoin transactions)
* **Classes:** 
  * `Legitimate (0)`: ~90.2%
  * `Fraud (1)`: ~9.8%
  * `Unknown`: ~77% of total network (filtered out for supervised training)
* **Features:** 166 features (local transaction features + aggregated 1-hop neighbor features)

---

##  Methodology & Key Learnings

1. **Handling Class Imbalance:**
   - Evaluated models using **PR-AUC (Precision-Recall Area Under Curve)**, **Recall**, and **F1-Score** instead of misleading accuracy metrics.
   - Configured XGBoost with `scale_pos_weight = Negative / Positive` ratio (~7.63 - 9.24).

2. **Validation Strategy (Temporal Split vs. Random Split):**
   - **Random Split Data Leakage:** A naive `train_test_split` yielded an over-optimistic **95% Recall**, as future and past transactions leaked into training.
   - **Temporal Split (Realistic Evaluation):** Split data chronologically by `time_step` (First 70% for Training, Last 30% for Testing).

---

##  Model Performance (Temporal Test)

Below are the test results evaluated on unseen future time steps (Steps 35 to 49):

| Metric | Score |
|---|---|
| **PR-AUC** | **0.7971** |
| **ROC-AUC** | **0.9295** |
| **Fraud Recall** | **74%** (800 / 1,083 fraudulent transactions caught) |
| **Fraud Precision** | **76%** |
| **Fraud F1-Score** | **0.75** |

---

## Repository Structure

```text
bitcoin-fraud-graph-features/
├── data/                       # Raw datasets (Git ignored)
├── notebooks/
│   ├── 01_eda.py               # Exploratory Data Analysis & Filtering
│   ├── 02_train.py             # XGBoost model with random split
│   ├── 03_shap_analysis.py     # Feature Importance & SHAP explanations
│   └── 04_temporal_split.py    # Realistic chronological validation
├── .gitignore
├── README.md
└── requirements.txt
