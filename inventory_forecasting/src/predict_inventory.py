# src/predict_inventory.py
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

MODEL_PATH = os.path.join("models", "demand_forecast.pkl")
FEATS_PATH = os.path.join("models", "feature_columns.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATS_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

# Load once at import (don’t reload per request)
_model, _feature_columns = load_model()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input data for prediction."""
    processed = df.copy()

    # Ordinals (vectorized)
    processed["date"] = pd.to_datetime(processed["date"], errors="coerce")
    processed["delivery_time"] = pd.to_datetime(processed["delivery_time"], errors="coerce")
    processed["date_ordinal"] = processed["date"].map(datetime.toordinal)
    processed["delivery_ordinal"] = processed["delivery_time"].map(datetime.toordinal)

    processed["delivery_days"] = processed["delivery_ordinal"] - processed["date_ordinal"]

    # One-hot for item
    processed = pd.concat(
        [processed, pd.get_dummies(processed["item"], prefix="item")],
        axis=1,
    )

    return processed

def predict_inventory_levels(input_data: pd.DataFrame) -> np.ndarray:
    """Predict inventory levels for input data."""
    processed = preprocess_data(input_data)

    # Build feature matrix in one shot (no column-by-column inserts)
    X = processed.reindex(columns=_feature_columns, fill_value=0)

    return _model.predict(X)
