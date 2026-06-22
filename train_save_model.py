import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv("loan_approval_data.csv")

# 2. Impute missing values
print("Imputing missing values...")
categoricalcol = ["Employment_Status", "Marital_Status", "Employer_Category", "Gender", "Property_Area", "Loan_Purpose"]
numericalcol = [col for col in df.columns if col not in categoricalcol + ["Education_Level", "Loan_Approved"]]

numimputer = SimpleImputer(strategy='mean')
df[numericalcol] = numimputer.fit_transform(df[numericalcol])

catgimputer = SimpleImputer(strategy='most_frequent')
df[categoricalcol] = catgimputer.fit_transform(df[categoricalcol])

# Fills missing values in target and Education_Level as well
df["Education_Level"] = df["Education_Level"].fillna("Graduate")
df["Loan_Approved"] = df["Loan_Approved"].fillna("No")

# 3. Label encode target and Education_Level
print("Label encoding Education_Level and Loan_Approved...")
le_edu = LabelEncoder()
df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

le_approved = LabelEncoder()
df["Loan_Approved"] = le_approved.fit_transform(df["Loan_Approved"])

# 4. One-hot encode other categorical features
print("One-hot encoding categoricals...")
ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore')
encoded = ohe.fit_transform(df[categoricalcol])
encodeddf = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categoricalcol), index=df.index)
df = pd.concat([df.drop(columns=categoricalcol), encodeddf], axis=1)

# 5. Define features X and target y
X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]

# 6. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train the models
print("Training models...")

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)
print("\n--- Logistic Regression ---")
print("Accuracy: ", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:   ", recall_score(y_test, y_pred_log))
print("F1-score: ", f1_score(y_test, y_pred_log))

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
print("\n--- KNN ---")
print("Accuracy: ", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn))
print("Recall:   ", recall_score(y_test, y_pred_knn))
print("F1-score: ", f1_score(y_test, y_pred_knn))

# Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)
y_pred_gnb = gnb_model.predict(X_test_scaled)
print("\n--- Naive Bayes ---")
print("Accuracy: ", accuracy_score(y_test, y_pred_gnb))
print("Precision:", precision_score(y_test, y_pred_gnb))
print("Recall:   ", recall_score(y_test, y_pred_gnb))
print("F1-score: ", f1_score(y_test, y_pred_gnb))

# 9. Save all artifacts in a single dict
print("\nSaving assets to loan_model_assets.pkl...")
assets = {
    "num_imputer": numimputer,
    "cat_imputer": catgimputer,
    "le_edu": le_edu,
    "le_approved": le_approved,
    "ohe": ohe,
    "scaler": scaler,
    "models": {
        "Logistic Regression": log_model,
        "KNN": knn_model,
        "Naive Bayes": gnb_model
    },
    "model_metrics": {
        "Logistic Regression": {
            "accuracy": accuracy_score(y_test, y_pred_log),
            "precision": precision_score(y_test, y_pred_log),
            "recall": recall_score(y_test, y_pred_log),
            "f1": f1_score(y_test, y_pred_log)
        },
        "KNN": {
            "accuracy": accuracy_score(y_test, y_pred_knn),
            "precision": precision_score(y_test, y_pred_knn),
            "recall": recall_score(y_test, y_pred_knn),
            "f1": f1_score(y_test, y_pred_knn)
        },
        "Naive Bayes": {
            "accuracy": accuracy_score(y_test, y_pred_gnb),
            "precision": precision_score(y_test, y_pred_gnb),
            "recall": recall_score(y_test, y_pred_gnb),
            "f1": f1_score(y_test, y_pred_gnb)
        }
    },
    "feature_columns": list(X.columns),
    "categorical_columns": categoricalcol,
    "numerical_columns": numericalcol
}

with open("loan_model_assets.pkl", "wb") as f:
    pickle.dump(assets, f)

print("Done! Model assets saved successfully.")
