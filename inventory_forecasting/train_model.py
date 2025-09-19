# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv('data/cleaned_data.csv')
    
    # Convert dates
    df['date'] = pd.to_datetime(df['date'])
    df['delivery_time'] = pd.to_datetime(df['delivery_time'])
    
    # Create features
    df['date_ordinal'] = df['date'].map(lambda x: x.toordinal())
    df['delivery_ordinal'] = df['delivery_time'].map(lambda x: x.toordinal())
    df['delivery_days'] = df['delivery_ordinal'] - df['date_ordinal']
    
    # One-hot encode items
    df = pd.get_dummies(df, columns=['item'], prefix='item')
    
    return df

# Prepare features and target
def prepare_features(df):
    # Exclude non-feature columns
    exclude_cols = ['date', 'delivery_time', 'stock']  # 'stock' is our target
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['stock']
    
    return X, y, feature_cols

# Train model
def train():
    df = load_and_preprocess_data()
    X, y, feature_cols = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained with MSE: {mse:.2f}, R²: {r2:.2f}")
    
    # Save model and feature columns
    os.makedirs('models', exist_ok=True)
    with open('models/demand_forecast.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("Model saved successfully")

if __name__ == '__main__':
    train()