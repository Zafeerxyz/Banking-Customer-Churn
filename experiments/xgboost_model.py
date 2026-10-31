
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import clean_data, build_preprocessor
from src.feature_engineering import create_features


# 1. Load and clean data
df = pd.read_csv("data/raw/Customer_Churn.csv")
df = clean_data(df)

# 2. Separate features and target
X = df.drop(columns=["Exited", "Complain"])
y = df["Exited"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 5. Build XGBoost pipeline
model = Pipeline(steps=[
    (
        "feature_engineering",
        FunctionTransformer(create_features, validate=False)
    ),
    ("preprocessor", build_preprocessor()),
    (
        "classifier",
        XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    )
])

# 6. Train
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save model
joblib.dump(model, "artifacts/xgboost_model.joblib")

print("\nXGBoost model saved successfully.")
