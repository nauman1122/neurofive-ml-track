import pandas as pd
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =========================================================
# 1. LOAD TITANIC DATASET
# =========================================================

df = sns.load_dataset("titanic")

print("Titanic dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# 2. SELECT FEATURES
# =========================================================

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df["survived"]


# =========================================================
# 3. DEFINE FEATURES
# =========================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# =========================================================
# 4. NUMERICAL PIPELINE
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# =========================================================
# 5. CATEGORICAL PIPELINE
# =========================================================

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# =========================================================
# 6. COLUMN TRANSFORMER
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# =========================================================
# 7. FINAL MODEL
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000
            )
        )
    ]
)


# =========================================================
# 8. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =========================================================
# 9. TRAIN MODEL
# =========================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# =========================================================
# 10. PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 11. EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n" + "=" * 55)
print("TITANIC MODEL PERFORMANCE")
print("=" * 55)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("=" * 55)


# =========================================================
# 12. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "titanic_model.joblib"
)

print("\nModel saved successfully!")
print("File: titanic_model.joblib")