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
print(df.isnull().sum())

