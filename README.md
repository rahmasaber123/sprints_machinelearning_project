# 🩺 Heart Disease Prediction – End-to-End Machine Learning Pipeline
## 📌 Project Overview

### This project builds a complete machine learning pipeline to predict heart disease using patient health records.
### It covers data preprocessing, dimensionality reduction, feature selection, supervised & unsupervised learning, model evaluation, and deployment.

### 👉 The ultimate goal is to help identify patients at risk through accurate and interpretable ML models.

## 📊 Dataset Information

Source: UCI Heart Disease Dataset

Rows: 303

Columns: 14 (13 features + 1 target)

Target Variable: num (0 = no disease, 1 = disease)

Missing Values: Present in ca, thal

Focus: Only the 13 main features were used

## 🛠️ Tools & Technologies

Language: Python 🐍

Libraries:

Data: Pandas, NumPy

## Visualization: Matplotlib, Seaborn, Plotly

ML: Scikit-learn

## 🔄 Project Workflow
1️⃣ Data Preprocessing & Cleaning

Handled missing values (ca, thal)

Encoded categorical features

Standardized numerical variables

Conducted Exploratory Data Analysis (EDA)

## 2️⃣ Feature Engineering & Reduction

PCA (Dimensionality Reduction) → retained maximum variance

Feature Selection → RFE, Chi-Square, and feature importance

## 3️⃣ Model Building

Supervised Learning: Logistic Regression, Decision Tree, Random Forest, SVM

Unsupervised Learning: K-Means & Hierarchical Clustering

## 4️⃣ Model Evaluation

Metrics: Accuracy, Precision, Recall, F1-score, AUC

Visualization: ROC curves, Confusion Matrix

## 5️⃣ Optimization & Deployment

Hyperparameter tuning with GridSearchCV & RandomizedSearchCV

Final model saved using Pickle for deployment

## 📈 Model Performance
Model	Accuracy	Precision	Recall	F1-Score	AUC
Logistic Regression	0.90	0.87	0.93	0.90	0.96
Decision Tree	0.85	0.85	0.82	0.84	0.88
Random Forest	0.93	0.90	0.96	0.93	0.96
SVM	0.92	0.87	0.96	0.92	0.93

## ✅ Best Model: Random Forest Classifier

### 🏆 Final Model – Random Forest

Hyperparameters Used:

n_estimators = 100

max_depth = 8

min_samples_split = 2

min_samples_leaf = 2

max_features = "log2"

bootstrap = False

random_state = 42

Test Set Performance:

Accuracy: 0.93

Precision: 0.90

Recall: 0.96

F1-score: 0.93

AUC: 0.96

## Confusion Matrix:

                 Predicted Negative   Predicted Positive
Actual Negative        30                 3
Actual Positive        1                 27

## 🚀 How to Run the Project

## Clone the Repository

git clone https://github.com/rahmasaber123/sprints_machinelearning_project.git
cd heart-disease-prediction


## Install Dependencies

pip install -r requirements.txt


Run Training & Evaluation

Open the Jupyter Notebook

Click Run All to execute the pipeline

## 📌 Future Improvements

Deploy with Streamlit or Flask for real-time prediction

Collect larger datasets for improved generalization

Apply deep learning models for advanced accuracy

## ✨ This project demonstrates how machine learning can assist healthcare by detecting heart disease risk early and accurately.
