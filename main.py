import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

matplotlib.use("TkAgg")

# Data Ingestion

data = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(data)

print(df.info)
print(df["Churn"].value_counts())

# Pre-processing - Looking through the visualized data on kaggle

df = df.drop(columns=["customerID"], axis=1) # Removing customerID column because we don't need it as we can just use an index

# Changing TotalCharges to numeric data, we did this because it was in string format and because it would not appear as null values if it had empty strings
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()

df = df.dropna() # Dropping missing values

print(df.isnull().sum())

# Direct encoding the target this is done to the target because if we used one-hot encoding, we would instead get 2 columns:
# Churn_No   Churn_Yes
#    1          0
#    0          1
# For logistic regression we are trying to predict a binary value and getting 2 columns would be problematic because it wouldn't be a binary value
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Converting all categorical dat into numerical data using one-hot encoding
# This is done by creating a new column for each category and then assigning it a binary value. For example:
# id    colour
# 1     red
# 2     blue
# 3     green
# 4     blue
# This would instead get transformed into -
# id    colour_red      colour_blue     colour_green    colour_blue
# 1         1                0               0               0
# 2         0                1               0               0
# 3         0                0               1               0
# 4         0                0               0               1
df = pd.get_dummies(df)

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

# features = df[["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
#                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
#                "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]].to_numpy()
#
# target = df["Churn"].to_numpy()

# Creating a variable for all the features and removing the Churn as that is our target, not a label.
# Afterward we convert all the data into the same type so we don't run into any type errors because otherwise some of our data is objects
features = df.drop("Churn", axis=1).astype(np.float64).to_numpy()

# Here we are using the Churn as the target variable and converting it to the same datatype as the features so again we don't run into type errors
# We are also reshaping it into a 2d array because before it was just a list and since we are doing matrix multiplication we need so each value has its own row
# rather than one row with all the values
target = df["Churn"].to_numpy().astype(np.float64).reshape(-1, 1)

# Standardization (Z-score normalization)
# We only standardize the features/labels in classification because we want a binary output.
# If were to standardize the target we would have numbers ranging from 0-1 i.e 0.845 or 0.542 which aren't binary.
# While we have some binary data in the features it is fine because it creates a consistent scale for all of the featuers.
# Although it would have been slightly more correct to only standardize the numerical features.
# Due to the fact we are using a simple algorithm of logistic regression it is fine
means = np.mean(features, axis=0)
stds = np.std(features, axis=0)

X = (features - means) / stds

# y = (target - np.mean(target)) / np.std(target)



# Shuffling data

indices = np.arange(X.shape[0])
np.random.seed(29)
np.random.shuffle(indices)

X = X[indices]
y = target[indices]

# Data Splitting - 80% train, 20% test

split = int(.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# def sigmoid(theta, X):
#     z = X @ theta
#     return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.01, iterations=1000):

    m, n = X.shape # Defining m as the number of data points and n has the number of features

    # We are creating another array filled with ones to represent the biases and then stacking it on beside the feature array
    # This is because our orignal features don't have a bias value so we can add it doing this
    # Another way to do this is creating a seperate array and doing the dot product because z is techically just a linear function
    # So z = X @ theta
    X = np.hstack([np.ones((m, 1)), X])


    # y = y.reshape(-1, 1) reduntant because we reshaped it before

    # Because we added another column for biases for all the features we need to add another column for theta value as well
    # We initialize them all to zero so all weights can be updated and the extra column is so our shape isn't mismatched for the dot product
    theta = np.zeros((n+1, 1))

    # Initalizing array to capture the cost value at each iteration
    costs = []

    for i in range(iterations):

        # Note since we are calculating z for all data points in one go we use X @ theta
        # If we were doing it for a single example we would use X @ theta_transposed because when doing dot product the shape would be mismatched
        z = X @ theta

        # Debugging step
        # print("X.dtype:", X.dtype)
        # print("theta.dtype:", theta.dtype)
        # print("z.dtype:", z.dtype)
        # print("type(z):", type(z), " type(z[0,0]):", type(z.flatten()[0]))

        # In the code below we are tranposing X for gradient desecnt function
        # Getting the predictions and then finding the error so we can update the weights of each feature
        # Once we have the error we can use it gradient descent and then update values for each feature
        # We also use the cost function to measure our model so we know gradient descent is working
        X_t = np.transpose(X)

        preds = 1 / (1 + np.exp(-z))

        error = preds - y

        cost = -1/m * np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds))

        gradient = 1/m * (X_t @ error)

        costs.append(cost)

        theta -= lr * gradient

    return theta, costs

theta, costs = logistic_regression(X_train, y_train)

# Prediction function so we can call to predict future values
def predict(X, theta):
    m = X.shape[0]
    X = np.hstack([np.ones((m, 1)), X])
    z = X @ theta
    return 1 / (1 + np.exp(-z))

threshold = 0.5
y_preds_probs = predict(X_test, theta)
y_preds = (y_preds_probs >= threshold).astype(int)

accuracy = np.mean(y_preds == y_test)
print(f"Accuracy: {accuracy:.4f}")

lambda_manual = 0.1
clf = LogisticRegression(penalty="l2", C=1/lambda_manual, max_iter=1000, solver="lbfgs")
clf.fit(X_train, y_train.ravel())

print("sklearn accuracy: ", clf.score(X_test, y_test))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))
print("\nClassification Report:\n", classification_report(y_test, y_preds))

feature_names = df.drop("Churn", axis=1).columns
weights = theta.flatten()[1:]
imp = pd.Series(weights, index=feature_names).sort_values()
print("Top positive churn drivers:\n", imp.tail(10))
print("Top negative churn drivers:\n", imp.head(10))

plt.figure()
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")


thresholds = np.linspace(0, 1, 101)
precisions = []
recalls = []
f1s = []

for t in thresholds:
    y_preds = (y_preds_probs >= t).astype(int)
    precisions.append(precision_score(y_test, y_preds))
    recalls.append(recall_score(y_test, y_preds))
    f1s.append(f1_score(y_test, y_preds))

plt.figure()
plt.plot(thresholds, precisions)
plt.xlabel("Threshold")
plt.ylabel("Precision")
plt.title("Precision vs. Threshold")


plt.figure()
plt.plot(thresholds, recalls)
plt.xlabel("Threshold")
plt.ylabel("Recall")
plt.title("Recall vs. Threshold")


plt.figure()
plt.plot(thresholds, f1s)
plt.xlabel("Threshold")
plt.ylabel("F1-Score")
plt.title("F1-Score vs. Threshold")
plt.show()