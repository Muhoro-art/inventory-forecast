# 📦 Inventory Forecasting Dashboard

A **Flask-based web application** for predicting inventory levels, generating restocking alerts, and providing real-time visibility into stock health.  
This project demonstrates how **AI-driven demand forecasting** and **data-driven decision support** can be applied to supply chain management in the context of **Industry 4.0**.

---

## 🌐 Overview

- **Data ingestion**: Uses historical sales and stock data (`cleaned_data.csv`).
- **Prediction engine**: Trained ML model (`demand_forecast.pkl`) with matching feature schema (`feature_columns.pkl`).
- **Analytics layer**: Forecasts future inventory and calculates days until stockout.
- **Alerts system**: Highlights items at risk of running out and recommends replenishment quantities.
- **Dashboard**:  
  - Data summary (records, items, averages).  
  - Restocking alerts with quick “order” buttons.  
  - Recent data view.  
  - Interactive Chart.js graph (Current Stock vs Sold vs Predicted).  
  - Prediction form for ad-hoc scenarios.  

---

## ⚙️ Tech Stack

- **Backend**: Python, Flask  
- **Data & ML**: Pandas, NumPy, scikit-learn  
- **Frontend**: Bootstrap, Chart.js, Jinja2 templates  
- **Model persistence**: Pickle  
- **Deployment ready**: Compatible with Docker / GitHub Actions  

---

## 🚀 Setup & Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
