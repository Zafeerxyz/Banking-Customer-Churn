import pandas as pd


from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from src.preprocessing import clean_data
from src.model import build_random_forest_model

df = pd.read_csv("data/raw/Customer_Churn.csv")
df = clean_data(df)

X = df.drop(columns=["Exited", "Complain"])
y = df["Exited"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf_model = build_random_forest_model()


param_grid = {
    "classifier__n_estimators": [200, 261, 300],
    "classifier__max_depth": [None, 10, 15],
    "classifier__min_samples_split": [10, 13, 15],
    "classifier__min_samples_leaf": [2, 3, 5],
    "classifier__max_features": ["sqrt"]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
    )

grid_search.fit(X_train, y_train)


# Results
print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV F1 Score:")
print(round(grid_search.best_score_, 3))


# Best model
best_model = grid_search.best_estimator_

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))