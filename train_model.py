import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\atpt1\coustomer churn\Churn_Modelling.csv")  # or "Churn_Modelling_EXTRA CREDIT.csv"
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Define features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train-test split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Decision Tree model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save the model properly using joblib
joblib.dump(model, "churn_model.pkl")

print("✅ Model trained and saved as churn_model.pkl")
print("Training features:", list(X.columns))
