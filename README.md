# 💳 Credit Card Approval System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-9cf)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

An end-to-end machine learning project that predicts whether a bank customer's credit card application should be approved, based on their personal, financial, and credit history data — built the way real banks automate credit risk decisions.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Tech Stack & Libraries](#-tech-stack--libraries)
- [Project Workflow](#-project-workflow)
- [Models Used](#-models-used)
- [Results](#-results)
- [Business Recommendations](#-business-recommendations)
- [Limitations](#-limitations)
- [How to Run](#-how-to-run)
- [Repository Structure](#-repository-structure)
- [References](#-references)
- [Author](#-author)

---

## 📌 Overview

Banks process a huge volume of credit card applications every day. A large number get rejected due to factors like high loan balances, low income, or too many credit inquiries. Doing this manually is slow and error-prone.

This project automates that decision using machine learning — built and trained on real-world style application data, following the same general approach banks use for credit scoring and risk assessment.

---

## 🎯 Problem Statement

Credit cards are a common risk-control instrument in the financial industry. Banks use applicant data to predict the probability of future default, and credit scores help objectively quantify risk.

Traditional models like Logistic Regression are simple and interpretable but can struggle with non-linear relationships. Modern ML methods (Random Forest, Boosting, etc.) capture complexity better but often sacrifice transparency. This project explores both approaches and compares their trade-offs for a real classification problem with **severe class imbalance**.

---

## 📊 Dataset

- **Source:** [Kaggle – Credit Card Approval Prediction](https://www.kaggle.com/datasets/laotse/credit-card-approval?select=credit_card_approval.csv)
- **Size:** ~25,000+ unique applicant records after cleaning (537K+ raw transaction-level records)
- **Target Variable:** `TARGET` — `1` = risky applicant, `0` = not risky

**Key features include:**

| Category | Features |
|---|---|
| Demographics | Gender, Age, Number of Children, Marital Status |
| Financial | Annual Income, Days Employed |
| Assets | Car Ownership, Property/Home Ownership |
| Background | Education Level, Housing Type, Job |
| Contact Info | Mobile, Work Phone, Phone, Email flags |
| Credit History | Status, Begin Months |

> ⚠️ Note: The raw dataset file is **not included** in this repository due to size constraints. Download it directly from the Kaggle link above to run the notebook.

---

## 🛠 Tech Stack & Libraries

**Language:** Python 3.x
**Environment:** Jupyter Notebook

| Purpose | Libraries |
|---|---|
| Data Handling | `numpy`, `pandas` |
| Visualization | `matplotlib`, `seaborn` |
| Statistics | `statsmodels` |
| Preprocessing | `scikit-learn` (`LabelEncoder`, `StandardScaler`) |
| Class Imbalance | `imbalanced-learn` (`SMOTE`) |
| ML Models | `scikit-learn` (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, AdaBoost, Gradient Boosting), `xgboost`, `lightgbm` |
| Model Evaluation | `scikit-learn` (`classification_report`, `confusion_matrix`, `roc_curve`, `roc_auc_score`, `cross_val_score`, `KFold`, `GridSearchCV`) |

---

## 🔄 Project Workflow

1. **Data Loading & Exploration** – Inspecting structure, datatypes, and shape of the dataset
2. **Variable Categorization** – Splitting columns into numerical vs categorical
3. **Missing Value Analysis** – Checking and handling nulls
4. **Feature Engineering** – Converting days to years, simplifying status categories, removing insignificant variables
5. **Exploratory Data Analysis (EDA)**
   - Univariate analysis of categorical & numerical variables
   - Bivariate analysis (Job vs Income, Education vs Job, Age vs Status, etc.)
   - Correlation heatmap & pairplots
6. **Outlier Detection & Treatment** – On `ANNUAL_INCOME` and `DAYS_EMPLOYED`
7. **Encoding** – Label encoding categorical variables, binary encoding for ownership flags
8. **Feature Scaling** – Standardizing numerical features
9. **Train-Test Split** – 80/20 split with stratification
10. **Handling Class Imbalance** – Applying **SMOTE** for oversampling the minority (risky) class
11. **Model Building** – Training multiple base and ensemble models
12. **Hyperparameter Tuning** – Using `GridSearchCV` to optimize each model
13. **Model Validation** – K-Fold Cross Validation + Boxplot comparison across models
14. **Final Model Selection** – Choosing the best performing, most balanced model

---

## 🤖 Models Used

| Type | Models |
|---|---|
| Linear | Logistic Regression |
| Tree-Based | Decision Tree |
| Ensemble – Bagging | Random Forest |
| Instance-Based | K-Nearest Neighbors (KNN) |
| Probabilistic | Naive Bayes |
| Ensemble – Boosting | AdaBoost, Gradient Boosting, XGBoost, LightGBM |

All models were tuned using **GridSearchCV** and validated using **K-Fold Cross Validation** with the **ROC-AUC** metric, since the dataset is heavily imbalanced and accuracy alone is a misleading metric here.

---

## 📈 Results

Due to the severe class imbalance in the dataset (very few "risky" applicants), raw accuracy scores across most models appeared misleadingly high. After applying SMOTE and properly evaluating with ROC-AUC and cross-validation:

- **Random Forest Classifier (tuned)** emerged as the most balanced and reliable model, showing the lowest variance across cross-validation folds and the best trade-off between precision, recall, and AUC score.
- Boosting methods (XGBoost, AdaBoost) also performed reasonably well after tuning.
- Simpler models like basic Logistic Regression and KNN tended to overfit or underperform on the minority class.

> Detailed metrics (AUC, Precision, Recall, F1-score per model) are available in the full project report and inside the notebook's model validation section.

---

## 💼 Business Recommendations

- **Income, employment length, and family size** were found to be the most predictive features for credit approval decisions.
- **Car ownership and housing type** were among the least predictive features.
- Lending strategy should adapt to economic conditions:
  - **Bull market (economy expanding):** Institutions can afford a slightly higher risk tolerance.
  - **Bear market (economy contracting):** A more conservative approval threshold is recommended to minimize default risk.
- Applicants with **higher income and at least one partner** showed a higher likelihood of approval, while applicants with **low income, no partner, and 2+ dependents** showed a slightly higher risk profile.

---

## ⚠️ Limitations

- Severe class imbalance in the target variable (very few risky applicants)
- Some inconsistent / incomplete records required imputation
- High dimensionality after encoding categorical variables
- Hyperparameter tuning across multiple models was computationally expensive

---

## 🚀 How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/Tejas-Padole/Credit-Card-Approval-System.git
   cd Credit-Card-Approval-System
   ```

2. **Install the required dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn statsmodels scikit-learn imbalanced-learn xgboost lightgbm jupyter
   ```

3. **Download the dataset**
   - Get `credit_card_approval.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/laotse/credit-card-approval?select=credit_card_approval.csv)
   - Place it in the project's root folder

4. **Launch the notebook**
   ```bash
   jupyter notebook Credit_Card_Approval_System_Notebook_File.ipynb
   ```

5. Run all cells in order to reproduce the EDA, preprocessing, model training, and evaluation.

---

## 📁 Repository Structure

```
Credit-Card-Approval-System/
│
├── Credit_Card_Approval_System_Notebook_File.ipynb   # Main analysis & modeling notebook
├── Credit Card Approval System Presentation.pptx     # Project presentation
├── Credit_Card_Approval_System_Project_Report.docx   # Detailed project report
├── Dataset.docx                                       # Dataset source / Kaggle link reference
├── .gitignore
└── README.md
```

---

## 📚 References

1. RBI: Bankwise ATM/POS/Card Statistics
2. PWC: The changing landscape of India's credit industry
3. Harsha Vardhan, H., Gupta, T., Rathod, N., Bose, T. and Sharma, N. (2022). *International Journal of Soft Computing and Engineering (IJSCE)*, 11(2)
4. Markova, M. (2021). Credit card approval model: An application of deep neural networks. *Seventh International Conference on New Trends in The Applications of Differential Equations in Sciences (NTADES 2020)*

---

## 👤 Author

**Tejas Padole**

[![GitHub](https://img.shields.io/badge/GitHub-Tejas--Padole-181717?logo=github)](https://github.com/Tejas-Padole)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tejas-padole01)

---

⭐ If you found this project useful, consider giving it a star on GitHub!
