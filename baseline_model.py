"""
Baseline Model - Simple Logistic Regression
Use this as your starting point
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
train = pd.read_csv('ecommerce_returns_train.csv')
test = pd.read_csv('ecommerce_returns_test.csv')

# Display basic information about the datasets
print("Training data shape:", train.shape)
print("Test data shape:", test.shape)
print("\nTraining data columns:", train.columns.tolist())
print("\nFirst few rows of training data:")
print(train.head())

# Identify the target column (assuming it's named 'returned' or similar)
# Common names: 'returned', 'return', 'is_returned', 'target', 'label'
potential_target_cols = ['returned', 'return', 'is_returned', 'target', 'label']
target_col = None
for col in potential_target_cols:
    if col in train.columns:
        target_col = col
        break

if target_col is None:
    # If standard names not found, assume last column is target
    target_col = train.columns[-1]
    print(f"\nAssuming '{target_col}' is the target column")
else:
    print(f"\nTarget column identified: '{target_col}'")

# Separate features and target
X_train = train.drop(columns=[target_col])
y_train = train[target_col]
X_test = test.drop(columns=[target_col])
y_test = test[target_col]

print(f"\nTarget distribution in training set:")
print(y_train.value_counts())

# Handle missing values
print("\nMissing values in training data:")
print(X_train.isnull().sum())

# Fill numeric columns with median (compute on training data only)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
median_values = {}
for col in numeric_cols:
    median_values[col] = X_train[col].median()
    X_train[col].fillna(median_values[col], inplace=True)
    X_test[col].fillna(median_values[col], inplace=True)

# Fill categorical columns with mode (compute on training data only)
categorical_cols = X_train.select_dtypes(include=['object']).columns
mode_values = {}
for col in categorical_cols:
    mode_values[col] = X_train[col].mode()[0] if not X_train[col].mode().empty else 'Unknown'
    X_train[col].fillna(mode_values[col], inplace=True)
    X_test[col].fillna(mode_values[col], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    # Handle unseen categories in test data by mapping to a consistent value
    X_test[col] = X_test[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([mode_values[col]])[0])
    label_encoders[col] = le

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_test_pred))

# Save the model and preprocessing artifacts
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump({'median': median_values, 'mode': mode_values}, 'preprocessing_values.pkl')

print("\nModel saved as 'logistic_regression_model.pkl'")
print("Scaler saved as 'scaler.pkl'")
print("Label encoders saved as 'label_encoders.pkl'")
print("Preprocessing values saved as 'preprocessing_values.pkl'")

# Feature importance (coefficients)
print("\nTop 10 Most Important Features (by absolute coefficient):")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_[0]
})
feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
print(feature_importance.head(10))
