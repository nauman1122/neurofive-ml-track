# Week 5 — Deploy Your Model as a Live Web App

## 🚢 Titanic Survival Predictor

This project deploys a machine learning model as an interactive Streamlit web application.

The model predicts whether a Titanic passenger is likely to survive based on passenger information.

## 📊 Model Performance

The Logistic Regression model achieved the following results:

| Metric | Score |
|---|---:|
| Accuracy | 79.33% |
| Precision | 71.05% |
| Recall | 78.26% |
| F1 Score | 74.48% |

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Seaborn
- Joblib
- Streamlit

## 🔧 Features

The web app allows users to enter:

- Passenger Class
- Sex
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare
- Port of Embarkation

After clicking **Predict Survival**, the app displays:

- Survival prediction
- Survival probability
- Not-survive probability

## 🚀 Live Demo

👉 [Try the Titanic Survival Predictor](https://neurofive-ml-track-6vs2dopn4hnyhyvsbdwnza.streamlit.app/)

## 📁 Project Files

- `app.py` — Streamlit web application
- `train_model.py` — Model training script
- `titanic_model.joblib` — Saved trained model
- `requirements.txt` — Required Python packages
- `handling_imbalanced_data.ipynb` — Week 5 notebook

## 🎯 Objective

The goal of this project is to take a machine learning model from a notebook and turn it into a usable, shareable web application.

---

**Built as part of the NeuroFive Machine Learning Fundamentals Track.**
