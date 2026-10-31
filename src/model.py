
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import build_preprocessor,build_scaled_preprocessor
from src.feature_engineering import create_features
from sklearn.neighbors import KNeighborsClassifier


def build_model():

    model = Pipeline(steps=[
        ("feature_engineering",
         FunctionTransformer(create_features, validate=False)),

        ("preprocessor", build_scaled_preprocessor()),

        ("classifier",
         LogisticRegression(
             max_iter=1000,
             class_weight="balanced"
         ))
    ])

    return model


def train_model(X_train, y_train):

    model = build_model()
    model.fit(X_train, y_train)

    return model


def build_random_forest_model():

    model = Pipeline(steps=[
        ("feature_engineering",
         FunctionTransformer(create_features, validate=False)),

        ("preprocessor", build_preprocessor()),

        ("classifier",
         RandomForestClassifier(
             n_estimators=200,
             class_weight="balanced",
             max_depth=10,
             min_samples_split=10,
             min_samples_leaf=5,
             random_state=42,
             n_jobs=-1
         ))
    ])

    return model


def build_knn_model():
    
    model = Pipeline(steps=[
        ("feature_engineering",
         FunctionTransformer(create_features, validate=False)),

        ("preprocessor", build_scaled_preprocessor()),

        ("classifier",
         KNeighborsClassifier(
             n_neighbors=5,
             weights="distance"
         ))
    ])

    return model