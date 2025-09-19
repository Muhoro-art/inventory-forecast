# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime
from src.predict_inventory import predict_inventory_levels
from src.inventory_analyzer import InventoryAnalyzer
import numpy as np

app = Flask(__name__)
analyzer = InventoryAnalyzer()

# Load the cleaned data for display
def load_data():
    try:
        df = pd.read_csv('data/cleaned_data.csv')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'delivery_time' in df.columns:
            df['delivery_time'] = pd.to_datetime(df['delivery_time'], errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    df = load_data()
    summary = None
    alerts = []
    predictions = []

    if not df.empty:
        summary = {
            'total_items': int(df['item'].nunique()),
            'total_records': int(len(df)),
            'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            'avg_sold': round(float(df['sold'].mean()), 2),
            'avg_stock': round(float(df['stock'].mean()), 2)
        }

        # Generate alerts
        alerts_df = analyzer.generate_alerts()
        if not alerts_df.empty:
            # make JSON-safe
            alerts_df = alerts_df.copy()
            alerts_df['current_stock'] = alerts_df['current_stock'].astype(float)
            alerts_df['days_until_stockout'] = alerts_df['days_until_stockout'].astype(float)
            alerts_df['recommended_order'] = alerts_df['recommended_order'].astype(float)
            alerts = alerts_df.to_dict('records')

        # Predictions for the chart (per item, days_ahead=7)
        preds_df = analyzer.predict_future_inventory(days_ahead=7)
        if not preds_df.empty:
            preds_df = preds_df.copy()
            preds_df['item'] = preds_df['item'].astype(str)
            preds_df['predicted_stock'] = preds_df['predicted_stock'].astype(float)
            preds_df['current_stock'] = preds_df['current_stock'].astype(float)
            preds_df['change'] = preds_df['change'].astype(float)
            # normalize date to string
            preds_df['date'] = preds_df['date'].astype(str)
            predictions = preds_df.to_dict('records')

        # 🔥 Normalize df dates to strings for the table
        df = df.copy()
        df["date"] = df["date"].apply(lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "")
        if "delivery_time" in df.columns:
            df["delivery_time"] = df["delivery_time"].apply(
                lambda x: x.strftime("%Y-%m-%d") if (pd.notna(x) and hasattr(x, "strftime")) else ("" if pd.isna(x) else str(x))
            )

    return render_template(
        'dashboard.html',
        data=df.to_dict('records'),
        summary=summary,
        alerts=alerts,
        predictions=predictions
    )

@app.route('/alerts')
def alerts():
    alerts_df = analyzer.generate_alerts()
    return render_template('alerts.html', alerts=alerts_df.to_dict('records'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json() or {}
        # Convert to DataFrame
        input_df = pd.DataFrame([payload])
        input_df['date'] = pd.to_datetime(input_df['date'], errors='coerce')
        input_df['delivery_time'] = pd.to_datetime(input_df['delivery_time'], errors='coerce')

        # Make prediction
        yhat = predict_inventory_levels(input_df)
        # Ensure JSON-safe primitive
        pred_value = float(yhat[0])

        return jsonify({
            'success': True,
            'prediction': pred_value,
            'message': 'Prediction successful'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error making prediction: {str(e)}'
        }), 400

@app.route('/place_order', methods=['POST'])
def place_order():
    try:
        order_data = request.get_json() or {}
        print(f"Order placed: {order_data}")
        return jsonify({
            'success': True,
            'message': f"Order for {order_data.get('quantity')} units of {order_data.get('item')} placed successfully"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error placing order: {str(e)}'
        }), 400

@app.route('/accuracy')
def accuracy():
    accuracy_info = {
        'mse': 0.85,
        'r2': 0.92,
        'improvements': [
            'Try more complex models like Random Forest or Gradient Boosting',
            'Include additional features like seasonality indicators',
            'Incorporate external data like holidays or promotions',
            'Use time-series specific models like ARIMA or Prophet',
            'Implement cross-validation for more robust evaluation'
        ]
    }
    return render_template('accuracy.html', accuracy=accuracy_info)

if __name__ == '__main__':
    app.run(debug=True)
