
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained ML pipeline
model = joblib.load("artifacts/gradient_boosting_model.joblib")

THRESHOLD = 0.3


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get customer details from HTML form
        customer_data = request.form.to_dict()

        # Convert numerical inputs
        numeric_columns = [
            "CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "EstimatedSalary",
            "Satisfaction_Score", "Point_Earned",
            "HasCrCard", "IsActiveMember"
        ]

        for column in numeric_columns:
            customer_data[column] = float(customer_data[column])
        
        # Validate customer information

        if not 300 <= customer_data["CreditScore"] <= 900:
            return "Credit Score must be between 300 and 900", 400

        if not 18 <= customer_data["Age"] <= 120:
            return "Age must be between 18 and 120", 400

        if not 0 <= customer_data["Tenure"] <= 10:
            return "Tenure must be between 0 and 10", 400

        if customer_data["Balance"] < 0:
            return "Balance cannot be negative", 400

        if not 1 <= customer_data["NumOfProducts"] <= 4:
            return "Number of Products must be between 1 and 4", 400

        if customer_data["EstimatedSalary"] < 0:
            return "Estimated Salary cannot be negative", 400

        if not 1 <= customer_data["Satisfaction_Score"] <= 5:
            return "Satisfaction Score must be between 1 and 5", 400

        if customer_data["Point_Earned"] < 0:
            return "Point Earned cannot be negative", 400

        if customer_data["HasCrCard"] not in [0, 1]:
            return "Has Credit Card must be 0 or 1", 400

        if customer_data["IsActiveMember"] not in [0, 1]:
            return "Active Member must be 0 or 1", 400

                
        # Convert to DataFrame
        new_customer = pd.DataFrame([customer_data])

        # Predict churn probability
        probability = model.predict_proba(new_customer)[0][1]

        # Apply decision threshold
        prediction = int(probability >= THRESHOLD)

        result = "Likely to churn" if prediction == 1 else "Likely to stay"

        return render_template(
            "index.html",
            prediction=result,
            churn_probability=f"{probability:.2%}",
            threshold=f"{THRESHOLD:.0%}"
        )

    except (ValueError, TypeError, KeyError) as e:
        return f"Invalid customer input: {e}", 400

    except Exception as e:
        return f"Prediction error: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)
