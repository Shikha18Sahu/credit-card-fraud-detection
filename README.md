# 💳 Credit Card Fraud Detection

## 📌 Problem Statement
Detecting fraudulent credit card transactions from highly 
imbalanced real-world data (49,605 transactions, only 148 fraud cases).

## 📊 Dataset
- Source: Kaggle (creditcard_2023.csv)
- 49,605 transactions | 31 features | Binary classification

## 🛠️ Tech Stack
Python | Scikit-learn | Pandas | NumPy | Streamlit | Google Colab

## 🤖 Models & Results
| Model | Accuracy | F1-Score (Fraud) |
|-------|----------|-----------------|
| Logistic Regression | 99.82% | 0.72 |
| Random Forest | 99.92% | 0.88 ✅ |

## ⚙️ Methodology
- EDA & statistical analysis on imbalanced dataset
- Preprocessing: StandardScaler + SimpleImputer Pipeline
- Stratified Train-Test Split (80-20)
- Model comparison & evaluation
- Deployed via Streamlit dashboard + ngrok

## 📁 Project Structure
- CreditCard.ipynb — Main notebook
- fraud_detection_model.pkl — Saved RF model
- app.py — Streamlit dashboard
