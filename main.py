import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# Data Ingestion

data = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

pd.options.display.max_rows = 100

df = pd.read_csv(data)

print(df.info)
print(df["Churn"].value_counts())

# Pre-processing - Looking through the visualized data on kaggle

df = df.drop(columns=["customerID"], axis=1) # Removing customerID column because we don't need it as we can just use an index

df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()

df = df.dropna()

print(df.isnull().sum())

# Feature selection

# CustomerID: A unique ID that identifies each customer.
# Gender: The customer’s gender: Male, Female
# Senior Citizen: Indicates if the customer is 65 or older: Yes, No
# Partner: Indicates if the customer has a partner: Yes, No
# Dependents: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
# Tenure: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
# Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No
# Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
# Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
# Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
# Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
# Device Protection: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
# Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
# Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
# Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
# Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
# Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No
# Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
# Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.
# Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.
# Churn: Yes = the customer left the company this quarter. No = the customer remained with the company.

features = df[["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]].to_numpy()

target = df["Churn"].to_numpy()

# Standardization (Z-score normalization)

# mus = np.mean(features, axis=0)
# sds = np.std(features, axis=0)

X = (features - np.mean(features)) / np.std(features)

y = (target - np.mean(target)) / np.std(target)

# Shuffling data

indices = np.arange(X.shape[0])
np.random.seed(29)
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# Data Splitting - 80% train, 20% test

split = int(.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# def sigmoid(theta, X):
#     z = X @ theta
#     return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.1, iterations=1000):

    m, n = X.shape

    X = np.hstack([np.ones((m, 1)), X])
    y = y.reshape(-1, 1)
    theta = np.zeros(n+1, 1)
    costs = []

    for i in range(iterations):

        z = X @ theta

        X_t = np.transpose(X)

        preds = 1 / (1 + np.exp(-z))

        error = preds - y

        cost = -1/iterations * np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))

        costs.append[cost]

        descent = 1/iterations * X_t * (error)

        theta -= lr * descent

    return theta, costs


