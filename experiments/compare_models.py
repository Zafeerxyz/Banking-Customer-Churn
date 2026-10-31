import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.preprocessing import clean_data, build_preprocessor
from src.feature_engineering import create_features

from src.model import (
    build_model,
    build_knn_model,
    build_random_forest_model
)


# Load and clean dataset
df = pd.read_csv("data/raw/Customer_Churn.csv")
df_clean = clean_data(df)

# Separate features and target
X = df_clean.drop(columns=["Exited", "Complain"])
y = df_clean["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build Dummy Classifier pipeline
dummy_model = Pipeline(steps=[
    ("feature_engineering", FunctionTransformer(
        create_features,
        validate=False
    )),
    ("preprocessor", build_preprocessor()),
    ("classifier", DummyClassifier(strategy="most_frequent"))
])

# Train
dummy_model.fit(X_train, y_train)

# Predict
y_pred = dummy_model.predict(X_test)

# Evaluate
print("Dummy Classifier Results")
print("------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))





# Build Logistic Regression pipeline
logistic_model = build_model()

# Train
logistic_model.fit(X_train, y_train)

# Predict
logistic_pred = logistic_model.predict(X_test)

# Evaluate
print("\nLogistic Regression Results")
print("---------------------------")
print("Accuracy:", accuracy_score(y_test, logistic_pred))
print("Precision:", precision_score(y_test, logistic_pred, zero_division=0))
print("Recall:", recall_score(y_test, logistic_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, logistic_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, logistic_pred, zero_division=0))




from sklearn.tree import DecisionTreeClassifier


# Build Decision Tree pipeline
decision_tree_model = Pipeline(steps=[
    ("feature_engineering", FunctionTransformer(
        create_features,
        validate=False
    )),
    ("preprocessor", build_preprocessor()),
    ("classifier", DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

# Train
decision_tree_model.fit(X_train, y_train)

# Predict
tree_pred = decision_tree_model.predict(X_test)

# Evaluate
print("\nDecision Tree Results")
print("---------------------")
print("Accuracy:", accuracy_score(y_test, tree_pred))
print("Precision:", precision_score(y_test, tree_pred, zero_division=0))
print("Recall:", recall_score(y_test, tree_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, tree_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, tree_pred, zero_division=0))    



# Build Random Forest pipeline
random_forest_model = random_forest_model = build_random_forest_model()

# Train
random_forest_model.fit(X_train, y_train)

# Predict
rf_pred = random_forest_model.predict(X_test)

# Evaluate
print("\nRandom Forest Results")
print("---------------------")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred, zero_division=0))
print("Recall:", recall_score(y_test, rf_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, rf_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred, zero_division=0))

from sklearn.ensemble import GradientBoostingClassifier

# Build Gradient Boosting pipeline
gradient_boosting_model = Pipeline(steps=[
    ("feature_engineering", FunctionTransformer(
        create_features,
        validate=False
    )),
    ("preprocessor", build_preprocessor()),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

# Train
gradient_boosting_model.fit(X_train, y_train)

# Predict
gb_pred = gradient_boosting_model.predict(X_test)

# Evaluate
print("\nGradient Boosting Results")
print("-------------------------")
print("Accuracy:", accuracy_score(y_test, gb_pred))
print("Precision:", precision_score(y_test, gb_pred, zero_division=0))
print("Recall:", recall_score(y_test, gb_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, gb_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, gb_pred, zero_division=0))


# Build KNN pipeline
knn_model = build_knn_model()

# Train
knn_model.fit(X_train, y_train)

# Predict
knn_pred = knn_model.predict(X_test)

# Evaluate
print("\nKNN Results")
print("-----------")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print("Precision:", precision_score(y_test, knn_pred, zero_division=0))
print("Recall:", recall_score(y_test, knn_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, knn_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, knn_pred, zero_division=0))


