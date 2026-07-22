import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("data/Customer_Churn.csv")

# Remove Customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to Numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill Missing Values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Label Encoding
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Features & Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("Accuracy :", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save Files
joblib.dump(model, "models/customer_churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("Model Saved Successfully!")

# Load Files
model = joblib.load("models/customer_churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

print("Model Loaded Successfully!")