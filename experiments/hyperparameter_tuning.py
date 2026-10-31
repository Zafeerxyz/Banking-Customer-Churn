import pandas as pd

from sklearn.model_selection import (train_test_split,
                                     RandomizedSearchCV,
                                     StratifiedKFold)

from scipy.stats import randint

from src.preprocessing import clean_data
from src.model import build_random_forest_model

df = pd.read_csv("data/raw/Customer_Churn.csv")

df = clean_data(df)

X = df.drop(columns=["Exited","Complain"])
y =df["Exited"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf_model = build_random_forest_model()

# Hyperparameter search space
param_distributions = {
    "classifier__n_estimators": randint(100, 501),
    "classifier__max_depth": [5, 8, 10, 12, 15, None],
    "classifier__min_samples_split": randint(2, 21),
    "classifier__min_samples_leaf": randint(1, 11),
    "classifier__max_features": ["sqrt", "log2"]
}

# Cross-validation strategy
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=1,
    return_train_score=True,
    verbose=1
)

# Perform hyperparameter tuning
random_search.fit(X_train, y_train)

print("\nBest Hyperparameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(round(random_search.best_score_, 3))

print("\nTraining F1 Score:")
print(round(random_search.cv_results_["mean_train_score"][
    random_search.best_index_
], 3))

# Get the best model
best_model = random_search.best_estimator_


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import os
import joblib

os.makedirs("artifacts", exist_ok=True)

joblib.dump(
    best_model,
    "artifacts/best_model.joblib"
)

print("Best model saved successfully!")



# RandomizedSearchCV was performed using 30 parameter combinations
# and 5-fold StratifiedKFold cross-validation, resulting in 150 model fits.
# The best cross-validation F1-score was 0.614. The selected model achieved a test accuracy of 82.1% and a churn-class F1-score of 0.625.
# Its recall increased compared with the initial Random Forest, indicating improved identification of potential churners, although precision decreased.