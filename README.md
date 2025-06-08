# Telco Customer Churn Prediction

This project aims to predict customer churn for a fictional telecommunications company using a logistic regression model. The project includes a complete machine learning pipeline, from data ingestion and preprocessing to model training and evaluation.

A key feature of this repository is the implementation of a logistic regression model from scratch using `NumPy`, which is then compared against the standard implementation from `scikit-learn`.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results and Analysis](#results-and-analysis)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Project Overview

Customer churn, the rate at which customers stop doing business with an entity, is a critical metric for subscription-based companies. This project tackles the problem by building a binary classification model that predicts whether a customer will churn or not based on their account information and services.

The primary model used is Logistic Regression, implemented in two ways:
1.  **Manual Implementation:** A from-scratch model built using `NumPy` to demonstrate the underlying mechanics of gradient descent and the logistic/sigmoid function.
2.  **Scikit-learn Implementation:** A model built using `sklearn.linear_model.LogisticRegression` for comparison, benchmarking, and leveraging a robust, optimized library.

## Dataset

The project uses the **"Telco Customer Churn"** dataset, which is publicly available on Kaggle.

- **File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Content:** The dataset contains 7043 customer records with 21 attributes, including:
    - **Customer Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
    - **Account Information:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
    - **Subscribed Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
    - **Target Variable:** `Churn` (Yes/No).

## Project Workflow

The script `main.py` follows a standard machine learning pipeline:

1.  **Data Ingestion:** The dataset is loaded from the CSV file into a pandas DataFrame.

2.  **Data Preprocessing & Cleaning:**
    - The `customerID` column is dropped as it is an identifier with no predictive value.
    - The `TotalCharges` column is converted from an object (string) type to a numeric type. Any entries that cannot be converted (e.g., empty strings) are set to `NaN`.
    - Rows with missing values (`NaN`) are dropped.
    - The target variable `Churn` is label-encoded, mapping "Yes" to `1` and "No" to `0`.
    - All categorical features are converted into numerical format using **one-hot encoding** (`pd.get_dummies`).

3.  **Feature Engineering & Scaling:**
    - The preprocessed data is separated into a feature matrix (`X`) and a target vector (`y`).
    - The features are standardized using **Z-score normalization** (`(feature - mean) / std_dev`) to ensure all features are on a consistent scale, which helps the gradient descent algorithm converge faster and more reliably.

4.  **Data Splitting:**
    - The dataset is shuffled randomly to prevent any ordering bias. A `seed` is used for reproducibility.
    - The data is split into an **80% training set** and a **20% testing set**.

5.  **Model Training:**
    - **Manual Logistic Regression:**
        - A function `logistic_regression` implements the training process using gradient descent.
        - It iteratively updates the model weights (`theta`) by minimizing the binary cross-entropy cost function.
        - The bias term is incorporated by adding a column of ones to the feature matrix.
    - **Scikit-learn Logistic Regression:**
        - An instance of `LogisticRegression` is created with L2 regularization.
        - The model is trained on the training data using the `.fit()` method.

6.  **Prediction & Evaluation:**
    - Predictions are made on the test set using both the manual and `sklearn` models.
    - For the manual model, a decision threshold of `0.5` is initially used to convert predicted probabilities into binary classes (0 or 1).
    - The models' performance is evaluated using:
        - **Accuracy Score**
        - **Confusion Matrix**
        - **Classification Report** (including Precision, Recall, and F1-Score)

## Results and Analysis

The script generates several outputs to analyze the model's performance:

1.  **Accuracy Scores:** Prints the accuracy for both the manually implemented model and the `scikit-learn` model.
2.  **Performance Metrics:** Displays a detailed confusion matrix and classification report for the manual model's predictions.
3.  **Convergence Plot:** A plot of **Cost vs. Iterations** is generated, showing how the cost function of the manual logistic regression model decreases over time, indicating that the gradient descent algorithm is successfully learning.
4.  **Threshold Analysis Plots:** To find the optimal decision threshold, the script generates three plots:
    - Precision vs. Threshold
    - Recall vs. Threshold
    - F1-Score vs. Threshold
    
    These plots help visualize the trade-off between precision and recall and identify a threshold that balances them effectively for the business problem.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    Ensure you have the required Python libraries installed. You can install them using pip:
    ```bash
    pip install numpy pandas scikit-learn matplotlib
    ```

3.  **Set up the data:**
    - Create a directory named `data`.
    - Download the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset and place it inside the `data` directory.

4.  **Execute the script:**
    Run the `main.py` script from your terminal:
    ```bash
    python main.py
    ```
    The script will print the evaluation metrics to the console and display the analysis plots.

**Note:** The script uses `matplotlib.use("TkAgg")`. If you encounter issues with a missing GUI backend, you may need to install a toolkit like `python3-tk` or change the backend to one compatible with your system (e.g., `"Agg"` for non-interactive plotting).

## Dependencies

- Python 3.x
- NumPy
- pandas
- Matplotlib
- scikit-learn