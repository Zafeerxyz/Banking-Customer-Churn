import pandas as pd
import joblib

from src.preprocessing import clean_data

# Load dataset
df = pd.read_csv("data/raw/Customer_Churn.csv")
df = clean_data(df)

# Separate features and target
X = df.drop(columns=["Exited", "Complain"])
y = df["Exited"]

# Load tuned Random Forest
model = joblib.load("artifacts/best_model.joblib")

# Get trained Random Forest classifier
rf = model.named_steps["classifier"]

# Get transformed feature names
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

# Get feature importance
importance = rf.feature_importances_

# Create importance dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(15))

# Key observations
# 1. Age is the strongest feature. The model relies heavily on customer age when making churn predictions.
# 2. Number of products is important. The number of products a customer uses contributes significantly to the predictions.
# 3. Age_Group also appears important. This is an engineered feature derived from Age, so some importance is shared between the original and engineered age features.
# 4. Product_Risk_Flag contributes useful information. It identifies customers with three or more products.
# 5. Balance and Balance_To_Salary both contribute to the model's decisions.