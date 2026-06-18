# 🛒 Customer Churn Prediction Dashboard
**Joy Muthoni Wanjiru | SCT213-C002-0004/2022 | JKUAT BSc Data Science**

Live Streamlit dashboard for customer churn prediction using machine learning.

## Pages
1. 📊 Overview — KPIs, risk distribution, churn by season & category
2. 👥 Customer Segments — gender, discount, rating, frequency breakdowns
3. 🤖 Model Performance — LR vs RF vs XGBoost ROC curves + confusion matrix
4. 🎯 Retention Actions — Monte Carlo simulation + revenue at risk + top 20 table
5. 🔮 Live Predictor — enter any customer's details → instant churn probability gauge

## Stack
Python · Streamlit · XGBoost · scikit-learn · Pandas · Matplotlib · Seaborn

## Files Required
- `app.py` — main dashboard
- `complete_churn_predictions_all_3900_customers.csv` — pre-computed results
- `customer_shopping_behavior.csv` — raw dataset for live model training
- `requirements.txt` — dependencies
- `runtime.txt` — Python 3.11
- `.streamlit/config.toml` — theme
