# 📉 ChurnGuard AI — Customer Churn Prediction

> An end-to-end Machine Learning project that predicts customer churn probability using **Logistic Regression** and provides an interactive **Streamlit dashboard** for customer risk assessment.

---

## 🚀 Project Overview

Customer churn is a major business challenge for subscription-based and service-oriented companies. Identifying customers who are likely to leave can help businesses take proactive retention actions.

**ChurnGuard AI** analyzes customer demographic, service, contract, tenure, and billing information to predict whether a customer is likely to churn.

The project covers the complete Machine Learning workflow:

**Data → Cleaning → Preprocessing → Model Training → Evaluation → Model Serialization → Streamlit Deployment**

---

## 🎯 Problem Statement

The objective of this project is to build a Machine Learning model that can predict whether a customer is likely to **Churn** or **Stay** based on historical customer information.

### Business Goal

Use customer-level information to:

- Identify customers at higher risk of churn
- Estimate individual churn probability
- Understand factors associated with customer churn
- Support proactive customer retention strategies

---

## 💡 Solution

The project uses **Logistic Regression** as the classification model.

A complete preprocessing pipeline is created using Scikit-learn:

- Numerical feature imputation
- Numerical feature scaling using `StandardScaler`
- Categorical feature encoding using `OneHotEncoder`
- Logistic Regression classification
- Probability-based churn prediction

The preprocessing and model are saved together as a `.joblib` pipeline so that the **same transformations used during training are applied during Streamlit prediction**.

---

## 🧠 Machine Learning Workflow

```text
                    Customer Churn Dataset
                              │
                              ▼
                       Data Inspection
                              │
                              ▼
                       Data Cleaning
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Numerical Features        Categorical Features
                 │                         │
                 ▼                         ▼
          Missing Value                 Missing Value
            Imputation                   Imputation
                 │                         │
                 ▼                         ▼
          StandardScaler             OneHotEncoder
                 │                         │
                 └────────────┬────────────┘
                              ▼
                       Logistic Regression
                              │
                              ▼
                       Model Evaluation
                              │
                              ▼
                    Saved ML Pipeline (.joblib)
                              │
                              ▼
                         Streamlit App
                              │
                              ▼
                  Customer Churn Prediction
