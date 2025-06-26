import pandas as pd


new_data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


print("\nFirst 5 Rows:\n", new_data.head())


print("\nShape of the dataset:", new_data.shape)


print("\nData Types:\n", new_data.dtypes)


missing_values = new_data.isnull().sum()
print("\nMissing Values:\n", missing_values)


new_data.info()
