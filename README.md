# 🛒 Customer Churn Prediction — A Business Intelligence Approach

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-XAI-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Joy Muthoni Wanjiru | SCT213-C002-0004/2022**  
**BSc Data Science — Jomo Kenyatta University of Agriculture and Technology (JKUAT)**  
**Supervisor: Mr. Isaac Mwangi Kega**

</div>

---

## 🚀 Live Demo

> **Click the button below to access the live interactive dashboard instantly — no installation required.**

<div align="center">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jysoniiee-Customer-Churn-Prediction-Dashboard.streamlit.app)

**🔗 https://joysoniiee-Customer-Churn-Prediction-dashboard.streamlit.app**

</div>

---

## 📌 Project Overview

This project develops a **machine learning pipeline** to predict customer churn in the retail and e-commerce sector. It combines predictive modelling, explainable AI, and business intelligence visualisation into a single deployable system.

The system:
- Trains and compares **Logistic Regression, Random Forest, and XGBoost** classifiers
- Applies **SHAP (SHapley Additive exPlanations)** to explain which features drive churn
- Segments all 3,900 customers into **four risk tiers** with prescriptive retention actions
- Quantifies business impact through **Monte Carlo simulation**
- Delivers insights through a **live interactive Streamlit dashboard**
- Exports results to **Power BI** for business stakeholder reporting

---

## 🎯 Research Objectives

| # | Objective | Status |
|---|-----------|--------|
| RO1 | Develop ML churn prediction models (LR, RF, XGBoost) | ✅ Complete |
| RO2 | Identify key churn drivers using SHAP explainability | ✅ Complete |
| RO3 | Visualise churn risk in Power BI dashboards | ✅ Complete |
| RO4 | Generate prescriptive retention recommendations | ✅ Complete |

---

## 📊 Model Performance Results

All models evaluated on a **30% stratified held-out test set** (random seed = 42).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7051 | 0.4842 | 0.9857 | 0.6493 | 0.8751 |
| Random Forest | 0.7812 | 0.5723 | 0.9286 | 0.7077 | 0.9088 |
| **XGBoost ★** | **0.8641** | **0.7308** | **0.9571** | **0.8308** | **0.9489** |

> ★ XGBoost selected as primary model — best across all metrics and compatible with SHAP TreeExplainer.

---

## 🔍 Top SHAP Feature Importances

| Rank | Feature | Mean \|SHAP Value\| |
|------|---------|-------------------|
| 1 | Frequency of Purchases | 0.2219 |
| 2 | Season | 0.2078 |
| 3 | Age | 0.2046 |
| 4 | Review Rating | 0.1853 |
| 5 | Previous Purchases | 0.1702 |
| — | Gender / Monetary | 0.0000 |

> Gender SHAP = 0.000 → The model is **gender-neutral**. Predictions are driven purely by behaviour. ✅

---

## 🗂️ Repository Structure

```
customer-churn-dashboard/
│
├── app.py                          # Main Streamlit dashboard (5 interactive pages)
├── requirements.txt                # Python dependencies (no version pins)
├── runtime.txt                     # Python 3.11 (forces stable runtime on Streamlit Cloud)
├── README.md                       # This file
│
├── .streamlit/
│   └── config.toml                 # Dashboard theme and server settings
│
├── data/
│   ├── customer_shopping_behavior.csv              # Raw dataset (Kaggle, 3,900 rows)
│   └── complete_churn_predictions_all_3900_customers.csv  # Model output with risk levels
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb      # Data cleaning and feature engineering
│   ├── 03_Modelling.ipynb          # Model training and evaluation
│   └── 04_SHAP_Analysis.ipynb      # SHAP explainability analysis
│
├── src/
│   └── churn_pipeline.py           # Complete ML pipeline (full annotated script)
│
├── artifacts/
│   ├── model_comparison.csv        # Metrics comparison table
│   └── churn_risk_summary.csv      # Risk level distribution summary
│
└── docs/
    └── project_report.docx         # Final year project report
```

---

## 🖥️ How to Access the Live Dashboard

### Option 1 — Access Online (Recommended, No Setup Required)

1. Click the link: **https://customer-churn-dashboard-pq6wsp6fwdwzmtesrtipo5.streamlit.app/**
2. The dashboard loads in your browser — no login, no installation
3. Use the **left sidebar** to navigate between the 5 pages
4. Use the **filters** (Season, Gender, Risk Level) to explore the data interactively
5. Go to **🔮 Live Predictor** to enter a customer's details and get an instant churn probability

---

### Option 2 — Run Locally on Your Computer

**Step 1 — Clone the repository**
```bash
git clone https://github.com/your-username/customer-churn-dashboard.git
cd customer-churn-dashboard
```

**Step 2 — Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Run the dashboard**
```bash
streamlit run app.py
```

**Step 5 — Open in browser**

The terminal will show:
```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Click the **Local URL** or it will open automatically.

---

## 📱 Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 **Overview** | 4 KPI cards · Risk donut chart · Churn by season and category · Probability histogram |
| 👥 **Customer Segments** | Churn by gender, discount, review rating, purchase frequency · Age density curves |
| 🤖 **Model Performance** | Metrics table · ROC curve comparison · XGBoost confusion matrix |
| 🎯 **Retention Actions** | Monte Carlo simulation · Revenue at risk · Top 20 highest-risk customers · CSV download |
| 🔮 **Live Predictor** | Enter any customer profile → instant churn probability gauge + all-model comparison |

---

## 🔮 Using the Live Predictor

1. Navigate to **🔮 Live Predictor** in the sidebar
2. Fill in the customer details form:
   - **Demographics** — Age, Gender, Location
   - **Purchase Behaviour** — Category, Item, Purchase Amount, Previous Purchases
   - **Engagement Signals** — Season, Review Rating, Discount Applied, Purchase Frequency, Subscription Status
3. Click **🔍 Predict Churn Risk**
4. The system returns:
   - A **gauge chart** showing churn probability (0–100%)
   - A **risk classification** (Low / Medium / High Risk)
   - A **recommended retention action**
   - A **bar chart** comparing the probability across all three models

**Example — High Risk Profile:**
```
Age: 25 | Gender: Female | Review Rating: 2.5
Discount Applied: Yes | Frequency: Annually | Subscription: No
→ Result: ~85% churn probability — Immediate Intervention Required
```

**Example — Low Risk Profile:**
```
Age: 35 | Gender: Male | Review Rating: 5.0
Discount Applied: No | Frequency: Weekly | Subscription: Yes
→ Result: ~12% churn probability — No Action Required
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **ML Libraries** | scikit-learn, XGBoost, SHAP |
| **Data** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **BI Tool** | Microsoft Power BI Desktop |
| **Deployment** | Streamlit Cloud (GitHub integration) |
| **Version Control** | Git, GitHub |

---

## 📦 Dataset

| Attribute | Value |
|-----------|-------|
| **Name** | Customer Shopping Behavior Dataset |
| **Source** | [Kaggle — Sourav Banerjee](https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset) |
| **Records** | 3,900 customers |
| **Features** | 19 original (14 retained after preprocessing) |
| **Domain** | Retail / E-Commerce |
| **Type** | Synthetic |

---

## ⚙️ Preprocessing Steps

1. **Missing value imputation** — median for numerical, mode for categorical columns
2. **Duplicate removal** — zero duplicates found
3. **Feature removal** — Size, Color, Shipping Type, Promo Code Used, Payment Method dropped
4. **Label encoding** — 8 categorical features encoded with `LabelEncoder`
5. **Standard scaling** — Age, Purchase Amount, Review Rating, Previous Purchases normalised: `z = (x − μ) / σ`
6. **RFM engineering** — Frequency (ordinal numeric), Monetary (= Purchase Amount)
7. **Churn label** — `Churn = 1` if Subscription Status = 'Yes', else `0`
8. **Leakage removal** — Recency caused perfect class separation and was removed

---

## 📈 Risk Segmentation

| Risk Level | Churn Probability | Count | Retention Decision |
|------------|-------------------|-------|--------------------|
| Low Risk | 0.0 – 0.3 | 2,705 | No action |
| Medium Risk | 0.3 – 0.6 | 84 | Monitor closely |
| High Risk | 0.6 – 0.8 | 1,111 | Target with retention offer |
| **Total** | — | **3,900** | — |

---

## 💰 Business Impact (Monte Carlo Simulation)

- **High risk customers targeted:** 1,111
- **Average purchase value:** $59.76
- **Assumed retention success rate:** 70%
- **Monte Carlo iterations:** 1,000
- **Expected churn reduction:** ~70% (95% CI: 65% – 75%)
- **Estimated revenue protected:** ~$46,500

---

## 🚀 Deployment Guide (GitHub → Streamlit Cloud)

1. **Fork or clone** this repository to your GitHub account
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
3. Click **New app**
4. Set:
   - Repository: `your-username/customer-churn-dashboard`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy** — live in approximately 3 minutes

> The `runtime.txt` file pins Python 3.11 to ensure all packages install correctly.

---

## 📚 Key References

1. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
2. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM KDD*, 785–794.
3. Hadden, J. et al. (2007). Computer assisted customer churn management. *Computers & Operations Research, 34*(10), 2902–2917.
4. Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS, 30*.
5. Burez, J. & Van den Poel, D. (2009). Handling class imbalance in customer churn prediction. *Expert Systems with Applications, 36*(3), 4626–4636.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Supervisor:** Mr. Isaac Mwangi Kega — JKUAT School of Computing
- **Dataset:** Sourav Banerjee via Kaggle
- **SHAP Library:** Lundberg & Lee (2017)
- **XGBoost:** Chen & Guestrin (2016)

---

<div align="center">

**Made with ❤️ by Joy Muthoni Wanjiru**  
*JKUAT BSc Data Science — Final Year Project 2024*

</div>
