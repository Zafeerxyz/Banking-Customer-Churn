# Banking Customer Churn Prediction

A machine learning project that predicts the likelihood of customer churn in the banking sector using customer information and behavioural attributes. The project includes an end-to-end machine learning workflow, from data exploration and preprocessing to model development and Flask application integration.

## Project Overview

Customer churn is a significant challenge for banking institutions. Identifying customers who are likely to leave can help businesses take proactive steps to improve customer retention.

This project develops a supervised machine learning classification system to predict whether a bank customer is likely to exit. A **Gradient Boosting Classifier** was selected for the final implementation and integrated into a Flask web application to make predictions through a user-friendly interface.

## Objectives

- Analyse customer data to identify patterns associated with churn.
- Perform data cleaning and preprocessing.
- Apply feature engineering to create meaningful predictive features.
- Develop and compare multiple classification models.
- Optimize model hyperparameters and evaluate performance.
- Integrate the selected model into a Flask-based web application.

## Technologies Used

| Category | Tools and Technologies |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Integration | Flask |
| Frontend | HTML, CSS, Bootstrap |
| Model Storage | Joblib |

## Machine Learning Workflow

### 1. Data Exploration and Analysis
- Inspected dataset structure, data types and statistical summaries.
- Analysed missing values and duplicate records.
- Performed univariate and bivariate analysis.
- Explored relationships between customer attributes and churn.

### 2. Data Preprocessing
- Handled missing values and duplicate records.
- Processed numerical and categorical variables.
- Applied numerical imputation and categorical encoding.
- Removed irrelevant identifier columns.
- Investigated and excluded features that could introduce target leakage.

### 3. Feature Engineering

Created additional features to capture useful customer information and relationships between existing variables.

Examples include:
- Age Group
- Balance-to-Salary Ratio
- Zero Balance Indicator
- Product Risk Flag

### 4. Model Development

Developed and compared the following classification algorithms:

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gradient Boosting

### 5. Hyperparameter Tuning and Evaluation

Applied hyperparameter tuning to improve model performance.

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The Gradient Boosting Classifier was selected for the final application based on the project's evaluation criteria, with particular consideration given to accuracy and precision.

### 6. Flask Application

Integrated the trained Gradient Boosting model into a Flask web application.

The application allows users to:
- Enter customer information through a web form.
- Submit customer details for prediction.
- View the predicted churn outcome.
- View the estimated churn probability.

## Project Structure

```text
Banking-Customer-Churn/
│
├── artifacts/
│   └── Saved model artifacts
│
├── data/
│   └── raw/
│       └── Customer_Churn.csv
│
├── experiments/
│   └── Model comparison and tuning scripts
│
├── notebooks/
│   └── ml_project.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── utils.py
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── predict.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9 or later
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Zafeerxyz/Banking-Customer-Churn.git
```

### 2. Navigate to the Project Directory

```bash
cd Banking-Customer-Churn
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Flask Application

```bash
python app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://127.0.0.1:5000/
```

## Key Learnings

Through this project, I gained practical experience in:

- Implementing an end-to-end supervised machine learning workflow.
- Performing exploratory data analysis and feature engineering.
- Identifying potential data leakage during model development.
- Comparing classification algorithms and tuning hyperparameters.
- Understanding the trade-off between precision and recall.
- Integrating a trained machine learning model into a Flask application.

## Future Improvements

- Deploy the application to a cloud platform.
- Add model monitoring and prediction logging.
- Improve model performance through further experimentation.
- Develop additional insights to support customer-retention strategies.


*This project was developed as a practical application of machine learning concepts, with a focus on solving a business-oriented classification problem.*


<img width="1726" height="922" alt="Screenshot 2026-11-02 162906" src="https://github.com/user-attachments/assets/b4729dd9-9558-4876-aed8-a17b8e8b4388" />
<img width="793" height="188" alt="Screenshot 2026-11-02 163119" src="https://github.com/user-attachments/assets/d356c445-efa6-449c-a48a-c340ff8b6db8" />
<img width="795" height="220" alt="Screenshot 2026-11-02 163208" src="https://github.com/user-attachments/assets/5b4406d7-e4cf-463d-9dc7-b6184a4094f7" />

