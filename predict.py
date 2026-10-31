import joblib
import pandas as pd

from src.preprocessing import clean_data


model = joblib.load(
    "artifacts/gradient_boosting_model.joblib"
)

df = pd.read_csv("data/raw/Customer_Churn.csv")
df = clean_data(df)

# Create an independent customer record
new_customer = pd.DataFrame([{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 42,
    "Tenure": 5,
    "Balance": 80000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 75000,
    "Satisfaction_Score": 3,
    "Card_Type": "Silver",
    "Point_Earned": 600
}])
# # 5. Make prediction
# prediction = model.predict(new_customer)

# # 6. Display result
# if prediction[0] == 1:
#     print("Prediction: Customer is likely to churn.")
# else:
#     print("Prediction: Customer is likely to stay")

# probability = model.predict_proba(new_customer)

# print("Stay Probability:", probability[0][0])
# print("Churn Probability:", probability[0][1])


# Get churn probability
probability = model.predict_proba(new_customer)

churn_probability = probability[0][1]

threshold = 0.3

if churn_probability >= threshold:
    print("Prediction: Customer is likely to churn.")
else:
    print("Prediction: Customer is likely to stay.")

print(f"Churn Probability:{churn_probability:.2%}" )
print(f"Decision Threshold: {threshold:.0%}")