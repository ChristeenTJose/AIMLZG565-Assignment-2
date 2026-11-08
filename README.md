# AIMLZG565 Assignment-2
## Problem statement:  Predicting Diabetes from Diagnostic Measurements

This repository contains the implementation and evaluation of various machine learning models aimed at predicting diabetes based on healthcare statistics and diagnostic measurements. Early diagnosis is critical as it enables timely lifestyle changes and more effective treatment interventions, preventing severe long-term complications.

The goal of this project is to accurately predict the presence of diabetes from clinical and lifestyle diagnostic survey data. In a healthcare context, **minimizing false negatives is critical** because delayed treatment can lead to severe long-term health complications. 

**Problem Type:** Binary Classification

## 📊 Dataset Description
* **Dataset:** [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
* **Source:** Kaggle
* **Samples:** 70,692
* **Features:** 21
* **Description:** Contains comprehensive healthcare statistics and lifestyle survey information collected from individuals, alongside their definitive diabetes diagnosis status.

## GitHub Repository Link: [AIMLZG565-Assignment-2](https://github.com/ChristeenTJose/AIMLZG565-Assignment-2)

## 📉 Model Comparison & Evaluation
Five machine learning models were trained and evaluated across multiple metrics including Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7488 | 0.8247 | 0.7353 | 0.7751 | 0.7547 | 0.4984 |
| **Decision Tree** | 0.6547 | 0.6546 | 0.6523 | 0.6584 | 0.6553 | 0.3095 |
| **KNN** | 0.6894 | 0.7469 | 0.6821 | 0.7062 | 0.6939 | 0.3792 |
| **Naive Bayes** | 0.6275 | 0.7393 | 0.7223 | 0.4107 | 0.5237 | 0.2815 |
| **Support Vector Machine** | 0.6291 | 0.7161 | 0.6123 | 0.7365 | 0.6343 | 0.3296 |

## 🔍 Performance Observations

* **Logistic Regression:** Leads across almost all evaluation metrics. It exhibits a high capacity to separate classes (AUC: 0.8247) and a balanced harmony between catching true positives and minimizing false alarms (F1: 0.7547). Its high MCC (0.4984) confirms it is the most reliable model.
* **Decision Tree:** While its Precision, Recall, and Accuracy hover uniformly around 65%, its low AUC (0.6546) indicates a weak ability to distinguish between classes compared to the alternative models.
* **KNN:** Ranks as the second-best model across most evaluation metrics. It provides decent baseline capabilities but does not reach the predictive strength of the Logistic Regression model.
* **Naive Bayes:** High precision but weak recall. It is highly selective when predicting positive instances (Precision: 0.7223), but it misses more than half of the actual positive cases in the dataset (Recall: 0.4107), leading to the lowest overall F1-score and MCC.
* **Support Vector Machine:** High recall but low precision. It is effective at capturing positive instances (Recall: 0.7365) but struggles with a high false-positive rate, resulting in the lowest precision (0.6123) among all tested models.

## 🏆 Overall Winner
**Logistic Regression** is the clear winner for this dataset. It achieves the highest Accuracy (74.88%), AUC (82.47%), Recall (77.51%), F1-Score (75.47%), and MCC (0.4984). Given the critical medical requirement to minimize false negatives, its high Recall combined with a robust, well-rounded predictive capability makes it the safest and most effective choices for this clinical task.

