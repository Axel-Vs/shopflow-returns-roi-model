"""
Prediction Script - Use trained model for inference
Load the saved model and make predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
import sys

def load_model_artifacts():
    """Load all saved model artifacts"""
    try:
        model = joblib.load('logistic_regression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        preprocessing_values = joblib.load('preprocessing_values.pkl')
        return model, scaler, label_encoders, preprocessing_values
    except FileNotFoundError as e:
        print(f"Error: Model file not found. Please train the model first using baseline_model.py")
        print(f"Missing file: {e.filename}")
        sys.exit(1)

def preprocess_data(df, label_encoders, preprocessing_values):
    """
    Preprocess input data using saved preprocessing parameters
    
    Args:
        df: Input DataFrame
        label_encoders: Dictionary of label encoders for categorical columns
        preprocessing_values: Dictionary with 'median' and 'mode' values
    
    Returns:
        Preprocessed DataFrame ready for scaling
    """
    X = df.copy()
    
    # Handle missing values using saved statistics
    median_values = preprocessing_values['median']
    mode_values = preprocessing_values['mode']
    
    # Fill numeric columns
    for col, median in median_values.items():
        if col in X.columns:
            X[col].fillna(median, inplace=True)
    
    # Fill categorical columns
    for col, mode in mode_values.items():
        if col in X.columns:
            X[col].fillna(mode, inplace=True)
    
    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in X.columns:
            default_value = le.transform([mode_values[col]])[0]
            mapping = {label: le.transform([label])[0] if label in le.classes_ else default_value 
                      for label in X[col].astype(str).unique()}
            X[col] = X[col].astype(str).map(mapping)
    
    return X

def predict(input_file, output_file=None):
    """
    Make predictions on new data
    
    Args:
        input_file: Path to CSV file with new data
        output_file: Optional path to save predictions (default: predictions.csv)
    """
    # Load model artifacts
    print("Loading model artifacts...")
    model, scaler, label_encoders, preprocessing_values = load_model_artifacts()
    
    # Load new data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Data shape: {df.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    X = preprocess_data(df, label_encoders, preprocessing_values)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_scaled)
    prediction_probabilities = model.predict_proba(X_scaled)
    
    # Add predictions to original dataframe
    df['predicted_return'] = predictions
    df['return_probability'] = prediction_probabilities[:, 1]
    
    # Save predictions
    if output_file is None:
        output_file = 'predictions.csv'
    
    df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total samples: {len(predictions)}")
    print(f"Predicted returns: {sum(predictions)}")
    print(f"Predicted non-returns: {len(predictions) - sum(predictions)}")
    print(f"Return rate: {sum(predictions) / len(predictions) * 100:.2f}%")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_file> [output_file]")
        print("Example: python predict.py new_data.csv predictions.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    predict(input_file, output_file)
