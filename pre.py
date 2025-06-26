import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna(subset=["TotalCharges"])


object_columns = data.select_dtypes(include="object").columns
le = LabelEncoder()
for col in object_columns:
    if col != "customerID":
        data[col] = le.fit_transform(data[col])


data_model = data.drop(columns=["customerID"])


X = data_model.drop(columns=["Churn"])
y = data_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n✅ Model Evaluation Results")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)


results = pd.DataFrame({
    "Churn_Probability": y_pred,
    "Actual_Churn": y_test.reset_index(drop=True)
})
results["Risk_Segment"] = np.where(results["Churn_Probability"] == 1, "At Risk", "Loyal")

X_test_reset = X_test.reset_index(drop=True)

dormant_mask = (X_test_reset["tenure"] < 3) & (X_test_reset["MonthlyCharges"] < 60)
results.loc[dormant_mask, "Risk_Segment"] = "Dormant"


print("\n✅ Final Segment Distribution:\n", results["Risk_Segment"].value_counts())


results[results["Risk_Segment"] == "Loyal"].to_csv("loyal_customers.csv", index=False)
results[results["Risk_Segment"] == "At Risk"].to_csv("at_risk_customers.csv", index=False)
results[results["Risk_Segment"] == "Dormant"].to_csv("dormant_customers.csv", index=False)


segment_counts = results["Risk_Segment"].value_counts()
plt.figure(figsize=(8, 5))
segment_counts.plot(kind="bar", color=["green", "red", "gray"])
plt.title("Customer Segmentation Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Risk Segment", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("risk_segment_distribution.png", dpi=300)
plt.show()

print("\n Exported Loyal, At Risk, and Dormant customer files.")
print(" Exported Risk Segment chart as 'risk_segment_distribution.png'.")
