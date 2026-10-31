import pandas as pd
import joblib
import os

from scipy.stats import randint, uniform

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.preprocessing import clean_data, build_preprocessor
from src.feature_engineering import create_features

# Load dataset
df = pd.read_csv("data/raw/Customer_Churn.csv")

# Clean dataset
df = clean_data(df)

# Separate features and target
X = df.drop(columns=["Exited", "Complain"])
y = df["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

gb_model = Pipeline(steps=[
    ("feature_engineering", FunctionTransformer(
        create_features,
        validate=False
    )),
    ("preprocessor", build_preprocessor()),
    ("classifier", GradientBoostingClassifier(
        random_state=42
    ))
])

param_distributions = {
    "classifier__n_estimators": randint(100, 501),
    "classifier__learning_rate": uniform(0.01, 0.19),
    "classifier__max_depth": randint(2, 6),
    "classifier__min_samples_split": randint(2, 21),
    "classifier__min_samples_leaf": randint(1, 11),
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=gb_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=1,
    return_train_score=True,
    verbose=1
)

# random_search.fit(X_train, y_train)

best_model = joblib.load(
    "artifacts/gradient_boosting_model.joblib"
)

# best_model = random_search.best_estimator_




# print("Best Parameters:")
# print(random_search.best_params_)

# print("\nBest Cross-Validation F1 Score:")
# print(random_search.best_score_)

# print("\nTraining F1 Score:")
# print(random_search.cv_results_["mean_train_score"][random_search.best_index_])


y_pred = best_model.predict(X_test)

print("\nTest Set Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

results = X_test.copy()

results["Actual"] = y_test
results["Predicted"] = y_pred

misclassified = results[
    results["Actual"] != results["Predicted"]
]

print("Total misclassified:", len(misclassified))
print(misclassified.head(10))


os.makedirs("artifacts", exist_ok=True)

joblib.dump(
    best_model,
    "artifacts/gradient_boosting_model.joblib"
)

print("\nGradient Boosting model saved successfully!")

false_positives = results[
    (results["Actual"] == 0) &
    (results["Predicted"] == 1)
]

false_negatives = results[
    (results["Actual"] == 1) &
    (results["Predicted"] == 0)
]

print("False Positives:", len(false_positives))
print("False Negatives:", len(false_negatives))

print("\nFalse Negative feature averages:")
print(false_negatives.select_dtypes(
    include="number"
).mean())

print("\nCorrectly predicted churners:")
correct_churners = results[
    (results["Actual"] == 1) &
    (results["Predicted"] == 1)
]

print(correct_churners.select_dtypes(
    include="number"
).mean())

#ERROR ANALYSIS

# 1. Age

# The model correctly identifies more churners among older customers. The missed churners are younger on average.

# 2. Number of products

# Missed churners have an average of 1.29 products, compared with 1.65 for correctly identified churners.

# This suggests that customers with fewer products may be harder for the model to identify.

# 3. Active membership

# The average IsActiveMember is higher among missed churners (0.395 vs 0.254).

# This may indicate that some active customers still churn, making them harder to identify.

# 4. Tenure and balance

# These averages are quite similar between the two groups. They don't show a strong difference in this comparison.