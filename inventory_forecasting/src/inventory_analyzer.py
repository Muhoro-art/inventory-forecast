# src/inventory_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.predict_inventory import predict_inventory_levels

class InventoryAnalyzer:
    def __init__(self, data_path='data/cleaned_data.csv'):
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['delivery_time'] = pd.to_datetime(self.df['delivery_time'])
        
    def get_current_inventory(self):
        """Get the most recent inventory for each item"""
        latest_dates = self.df.groupby('item')['date'].max()
        current_inventory = []
        
        for item, date in latest_dates.items():
            item_data = self.df[(self.df['item'] == item) & (self.df['date'] == date)]
            if not item_data.empty:
                current_inventory.append({
                    'item': item,
                    'stock': item_data['stock'].values[0],
                    'last_updated': date
                })
                
        return pd.DataFrame(current_inventory)
    
    def generate_alerts(self, threshold=0.2):
        """Generate alerts for items that need restocking"""
        current_inv = self.get_current_inventory()
        alerts = []
        
        for _, row in current_inv.iterrows():
            item = row['item']
            current_stock = row['stock']
            
            # Get historical sales data for this item
            item_data = self.df[self.df['item'] == item]
            avg_daily_sales = item_data['sold'].mean()
            
            # Calculate days until stockout
            if avg_daily_sales > 0:
                days_until_stockout = current_stock / avg_daily_sales
            else:
                days_until_stockout = float('inf')
            
            # Check if alert is needed
            if days_until_stockout < 7:  # Less than a week of inventory
                # Calculate recommended order quantity (2 weeks of sales)
                recommended_order = max(avg_daily_sales * 14, 10)
                
                alerts.append({
                    'item': item,
                    'current_stock': current_stock,
                    'avg_daily_sales': round(avg_daily_sales, 2),
                    'days_until_stockout': round(days_until_stockout, 2),
                    'recommended_order': round(recommended_order),
                    'urgency': 'High' if days_until_stockout < 3 else 'Medium'
                })
        
        return pd.DataFrame(alerts).sort_values('days_until_stockout')
    
    def predict_future_inventory(self, days_ahead=7):
        """Predict inventory levels for all items days ahead"""
        predictions = []
        current_inv = self.get_current_inventory()
        
        for _, row in current_inv.iterrows():
            item = row['item']
            current_stock = row['stock']
            
            # Create future date for prediction
            future_date = datetime.now() + timedelta(days=days_ahead)
            
            # Prepare prediction data
            prediction_data = pd.DataFrame([{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'item': item,
                'sold': 0,  # This will be replaced with average sales
                'stock': current_stock,
                'delivery_time': future_date.strftime('%Y-%m-%d')
            }])
            
            # Get average sales for this item
            item_data = self.df[self.df['item'] == item]
            avg_sales = item_data['sold'].mean()
            
            # Adjust prediction data with average sales
            prediction_data['sold'] = avg_sales
            
            try:
                # Make prediction
                predicted_stock = predict_inventory_levels(prediction_data)[0]
                
                predictions.append({
                    'item': item,
                    'current_stock': current_stock,
                    'predicted_stock': max(0, predicted_stock),  # Ensure non-negative
                    'change': predicted_stock - current_stock,
                    'date': future_date.strftime('%Y-%m-%d')
                })
            except:
                # If prediction fails, use simple calculation
                predicted_stock = current_stock - (avg_sales * days_ahead)
                predictions.append({
                    'item': item,
                    'current_stock': current_stock,
                    'predicted_stock': max(0, predicted_stock),
                    'change': predicted_stock - current_stock,
                    'date': future_date.strftime('%Y-%m-%d')
                })
        
        return pd.DataFrame(predictions)