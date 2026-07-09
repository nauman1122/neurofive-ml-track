# Titanic Survival Prediction using Logistic Regression

## Overview

This project uses Machine Learning to predict whether a Titanic passenger survived or did not survive.

The Titanic dataset was cleaned and prepared before building a Logistic Regression classification model.

## Approach

The following steps were performed:

1. Loaded the Titanic dataset using Pandas.
2. Inspected and cleaned the dataset.
3. Handled missing values in important columns.
4. Selected relevant features for prediction.
5. Encoded categorical columns (`sex` and `embarked`) using `pd.get_dummies()`.
6. Split the dataset into training and testing sets using `train_test_split`.
7. Trained a Logistic Regression model.
8. Predicted passenger survival on the test dataset.
9. Evaluated the model using `accuracy_score`.
10. Created a confusion matrix to analyze the model's predictions.

## Features Used

The following features were used:

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

## Encoding

Categorical features such as `Sex` and `Embarked` were converted into numerical features using `pd.get_dummies()`.

## Train-Test Split

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

The `random_state` was set to `42` to make the results reproducible.

## Machine Learning Model

**Logistic Regression**

Logistic Regression was selected because the target variable has two possible outcomes:

- `0` = Did not survive
- `1` = Survived

## Model Accuracy

The Logistic Regression model achieved a final accuracy of:

**80.45%**

## Confusion Matrix

The confusion matrix compares the actual survival status with the predicted survival status.

It shows:

- Correct predictions for passengers who did not survive
- Correct predictions for passengers who survived
- Incorrect predictions for both classes

This helps us understand the types of mistakes made by the model in addition to the overall accuracy.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Conclusion

The Logistic Regression model achieved **80.45% accuracy** on the test dataset. The model was able to classify Titanic passengers into survived and did-not-survive categories, while the confusion matrix provided additional insight into its correct and incorrect predictions.