import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


np.random.seed(42)
n_records = 500

# Generate 4 financial risk variables
credit_score = np.random.randint(300, 850, n_records)
debt_to_income_ratio = np.random.uniform(0, 1, n_records)
years_of_credit_history = np.random.randint(0, 50, n_records)
num_late_payments = np.random.choice(
    range(11),
    n_records,
    p=[0.5] + [0.05] * 10
)


df = pd.DataFrame({
    'credit_score': credit_score,
    'debt_to_income_ratio': debt_to_income_ratio,
    'years_of_credit_history': years_of_credit_history,
    'num_late_payments': num_late_payments
})


df['high_risk'] = (
    (df['credit_score'] < 600) |
    (df['debt_to_income_ratio'] > 0.5) |
    (df['years_of_credit_history'] < 5) |
    (df['num_late_payments'] > 5)
).astype(int)
print("High risk records: ", df['high_risk'].sum())

X = df.drop('high_risk', axis=1)
y = df['high_risk']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("\nSummary Statistics:")
print(df.describe())


# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow_Logistic_Regression")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the metrics
    METRIC_LIST = {
        "accuracy": accuracy,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1
    }

    for metric_name, metric_value in METRIC_LIST.items():
        mlflow.log_metric(metric_name, metric_value)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Logistic Regression model")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="risk_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="MLflow_tracking",
    )

# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

feature_names = X.columns.tolist()

result = pd.DataFrame(X_test, columns=feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions
result[:5]
