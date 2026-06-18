import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
body { font-family: 'Calibri', sans-serif; }

/* ── KPI cards ── */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 12px;
    border-top: 4px solid #1E3A5F;
}
.kpi-val  { font-size: 2.2rem; font-weight: 800; margin: 0; color: #1E3A5F; }
.kpi-lbl  { font-size: 0.82rem; color: #6B7280; margin: 4px 0 0 0; }

/* ── Section headers ── */
.sec-hdr {
    background: #1E3A5F;
    color: white;
    padding: 9px 16px;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    margin: 18px 0 10px 0;
}

/* ── Prediction result box ── */
.pred-box {
    border-radius: 14px;
    padding: 24px 28px;
    text-align: center;
    font-size: 1.1rem;
}
.pred-low  { background:#D1FAE5; border: 2px solid #10B981; color:#065F46; }
.pred-med  { background:#FEF3C7; border: 2px solid #F59E0B; color:#92400E; }
.pred-high { background:#FEE2E2; border: 2px solid #EF4444; color:#991B1B; }
.pred-vhigh{ background:#7F1D1D; border: 2px solid #450A0A; color:#FFFFFF; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #1E3A5F !important; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio > label { color: white !important; }

/* ── Info banner ── */
.info-banner {
    background: #EFF6FF;
    border-left: 4px solid #2E86AB;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 0.9rem;
    color: #1E3A5F;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  COLOUR MAPS
# ─────────────────────────────────────────────
RISK_CLR = {
    'Low Risk'      : '#10B981',
    'Medium Risk'   : '#F59E0B',
    'High Risk'     : '#EF4444',
}

# ─────────────────────────────────────────────
#  LOAD PRE-COMPUTED RESULTS  (CSV from project)
# ─────────────────────────────────────────────
@st.cache_data
def load_results():
    df = pd.read_csv("complete_churn_predictions_all_3900_customers.csv")
    return df

# ─────────────────────────────────────────────
#  TRAIN LIVE MODELS  (for predictor + perf page)
# ─────────────────────────────────────────────
@st.cache_resource
def train_models():
    raw = pd.read_csv("customer_shopping_behavior.csv")
    raw.columns = raw.columns.str.strip().str.replace('\ufeff','')
    raw.rename(columns={'Purchase Amount (USD)': 'Purchase Amount'}, inplace=True)

    # Drop irrelevant columns
    drop_c = ['Size','Color','Shipping Type','Promo Code Used','Payment Method']
    raw.drop(columns=[c for c in drop_c if c in raw.columns], inplace=True)

    # Clean Subscription Status
    raw['Subscription Status'] = (raw['Subscription Status']
                                  .astype(str).str.strip().str[:3])
    raw['Subscription Status'] = raw['Subscription Status'].apply(
        lambda x: 'Yes' if x.lower().startswith('yes')
        else ('No' if x.lower().startswith('no') else 'No'))

    # Fill missing values — objects first, then numbers
    for c in raw.select_dtypes('object').columns:
        raw[c] = raw[c].fillna(raw[c].mode()[0])
    for c in raw.select_dtypes('number').columns:
        raw[c] = raw[c].fillna(raw[c].median())

    raw.drop_duplicates(inplace=True)

    # Churn label: subscribers (Yes) = churn = 1
    raw['Churn'] = (raw['Subscription Status'] == 'Yes').astype(int)

    # Add numeric Frequency feature from Frequency of Purchases (text → ordinal)
    freq_map = {
        'Weekly': 6, 'Bi-Weekly': 5, 'Fortnightly': 4,
        'Monthly': 3, 'Quarterly': 2, 'Every 3 Months': 2, 'Annually': 1
    }
    raw['Frequency'] = raw['Frequency of Purchases'].map(freq_map).fillna(3)
    raw['Monetary']  = raw['Purchase Amount']

    df_m = raw.copy()

    # Label-encode ALL categorical columns
    cat_cols = ['Gender','Item Purchased','Category','Location','Season',
                'Subscription Status','Discount Applied','Frequency of Purchases']
    enc = {}
    for c in cat_cols:
        if c in df_m.columns:
            le = LabelEncoder()
            df_m[c] = le.fit_transform(df_m[c].astype(str))
            enc[c] = le

    # Scale numerical columns
    num_cols = ['Age','Review Rating','Previous Purchases','Purchase Amount']
    num_cols = [c for c in num_cols if c in df_m.columns]
    sc = StandardScaler()
    df_m[num_cols] = sc.fit_transform(df_m[num_cols].astype(float))

    # Build feature matrix — drop target + identifier
    drop_x = ['Churn', 'Subscription Status', 'Customer ID']
    X = df_m.drop(columns=[c for c in drop_x if c in df_m.columns])
    y = df_m['Churn']

    # Final safety: fill any remaining NaN with 0
    X = X.fillna(0).astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    lr  = LogisticRegression(max_iter=1000, solver='liblinear',
                             class_weight='balanced', random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight='balanced', random_state=42)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        eval_metric='logloss', verbosity=0,
                        scale_pos_weight=float(
                            (len(y_tr)-int(y_tr.sum()))/max(int(y_tr.sum()),1)))

    lr.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)
    xgb.fit(X_tr, y_tr)

    return {'LR':lr, 'RF':rf, 'XGB':xgb}, X_te, y_te, enc, sc, list(X.columns)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Churn Dashboard")
    st.markdown("**Joy Muthoni Wanjiru**")
    st.markdown("SCT213-C002-0004/2022")
    st.markdown("JKUAT · BSc Data Science")
    st.markdown("---")

    page = st.radio("📌 Navigate", [
        "📊 Overview",
        "👥 Customer Segments",
        "🤖 Model Performance",
        "🎯 Retention Actions",
        "🔮 Live Predictor",
    ])

    st.markdown("---")
    st.markdown("**Quick Stats**")

df   = load_results()

with st.sidebar:
    st.markdown(f"Total customers: **{len(df):,}**")
    st.markdown(f"High risk: **{(df['Risk_Level']=='High Risk').sum():,}**")
    st.markdown(f"Avg churn prob: **{df['Churn_Probability'].mean():.2f}**")
    st.markdown("---")
    st.markdown("**Global Filters**")
    f_season = st.multiselect("Season", sorted(df['Season'].unique()),
                               default=sorted(df['Season'].unique()))
    f_gender = st.multiselect("Gender", sorted(df['Gender'].unique()),
                               default=sorted(df['Gender'].unique()))
    f_risk   = st.multiselect("Risk Level",
                               ['Low Risk','Medium Risk','High Risk'],
                               default=['Low Risk','Medium Risk','High Risk'])

dff = df[
    df['Season'].isin(f_season) &
    df['Gender'].isin(f_gender) &
    df['Risk_Level'].isin(f_risk)
].copy()

# ══════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Customer Churn Overview")
    st.markdown("**Customer Churn Prediction — Retail E-Commerce · JKUAT Final Year Project**")
    st.markdown("---")

    # ── KPIs ──────────────────────────────────
    k1,k2,k3,k4 = st.columns(4)
    total       = len(dff)
    churn_rate  = (dff['Actual_Subscription_Status']=='Yes').mean()
    avg_prob    = dff['Churn_Probability'].mean()
    high_n      = (dff['Risk_Level']=='High Risk').sum()

    with k1:
        st.markdown(f'<div class="kpi-card"><p class="kpi-val">{total:,}</p>'
                    f'<p class="kpi-lbl">Total Customers</p></div>',
                    unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#EF4444">'
                    f'{churn_rate:.1%}</p><p class="kpi-lbl">Actual Churn Rate</p></div>',
                    unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#F59E0B">'
                    f'{avg_prob:.3f}</p><p class="kpi-lbl">Avg Churn Probability</p></div>',
                    unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#7F1D1D">'
                    f'{high_n:,}</p><p class="kpi-lbl">High Risk Customers</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    # Risk donut
    with c1:
        st.markdown('<div class="sec-hdr">Risk Level Distribution</div>',
                    unsafe_allow_html=True)
        rc = dff['Risk_Level'].value_counts().reindex(
            ['Low Risk','Medium Risk','High Risk'], fill_value=0)
        fig, ax = plt.subplots(figsize=(5,4))
        wedges, _, auts = ax.pie(
            rc.values,
            labels=rc.index,
            colors=[RISK_CLR[r] for r in rc.index],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2},
            textprops={'fontsize':9})
        for a in auts: a.set_fontweight('bold'); a.set_color('white')
        ax.set_title('Customers by Risk Level', fontsize=11, fontweight='bold', pad=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Churn probability histogram
    with c2:
        st.markdown('<div class="sec-hdr">Churn Probability Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(dff['Churn_Probability'], bins=35,
                color='#2E86AB', edgecolor='white', alpha=0.88)
        ax.axvline(0.6, color='#EF4444', lw=2, ls='--', label='Threshold 0.6')
        ax.set_xlabel('Churn Probability'); ax.set_ylabel('Customers')
        ax.set_title('Distribution of Churn Probabilities', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)

    # Churn by season
    with c3:
        st.markdown('<div class="sec-hdr">Churn Rate by Season</div>',
                    unsafe_allow_html=True)
        sc2 = (dff.groupby('Season')['Actual_Subscription_Status']
               .apply(lambda x: (x=='Yes').mean()*100)
               .sort_values(ascending=False))
        fig, ax = plt.subplots(figsize=(5,3.5))
        bars = ax.bar(sc2.index, sc2.values,
                      color=['#1E3A5F','#2E86AB','#48CAE4','#90E0EF'],
                      edgecolor='white')
        ax.bar_label(bars, fmt='%.1f%%', fontsize=8, padding=3)
        ax.set_ylabel('Churn Rate (%)'); ax.set_ylim(0, sc2.max()+15)
        ax.set_title('Churn Rate by Season', fontsize=11, fontweight='bold')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Churn by category
    with c4:
        st.markdown('<div class="sec-hdr">Churn Rate by Category</div>',
                    unsafe_allow_html=True)
        cc = (dff.groupby('Category')['Actual_Subscription_Status']
              .apply(lambda x: (x=='Yes').mean()*100)
              .sort_values())
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.barh(cc.index, cc.values, color='#2E86AB', edgecolor='white', alpha=0.9)
        ax.set_xlabel('Churn Rate (%)')
        ax.set_title('Churn Rate by Product Category', fontsize=11, fontweight='bold')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Summary table
    st.markdown('<div class="sec-hdr">Risk Level Summary Table</div>',
                unsafe_allow_html=True)
    summary = (dff.groupby('Risk_Level')
               .agg(Count=('Customer_ID','count'),
                    Avg_Churn_Prob=('Churn_Probability','mean'),
                    Avg_Purchase=('Purchase Amount','mean'),
                    Avg_Age=('Age','mean'))
               .reset_index())
    summary['Avg_Churn_Prob'] = summary['Avg_Churn_Prob'].round(3)
    summary['Avg_Purchase']   = summary['Avg_Purchase'].round(2)
    summary['Avg_Age']        = summary['Avg_Age'].round(1)
    st.dataframe(summary, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 2 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation Analysis")
    st.markdown("Breakdown of churn risk across demographic and behavioural segments")
    st.markdown("---")

    c1, c2 = st.columns(2)

    # Gender
    with c1:
        st.markdown('<div class="sec-hdr">Churn Rate by Gender</div>',
                    unsafe_allow_html=True)
        gc = (dff.groupby('Gender')['Actual_Subscription_Status']
              .apply(lambda x: (x=='Yes').mean()*100))
        fig, ax = plt.subplots(figsize=(5,3.5))
        bars = ax.bar(gc.index, gc.values,
                      color=['#2E86AB','#F59E0B'], edgecolor='white', width=0.4)
        ax.bar_label(bars, fmt='%.1f%%', fontsize=10, padding=4)
        ax.set_ylabel('Churn Rate (%)'); ax.set_ylim(0, gc.max()+15)
        ax.set_title('Churn Rate by Gender', fontsize=11, fontweight='bold')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Discount
    with c2:
        st.markdown('<div class="sec-hdr">Discount Applied vs Churn</div>',
                    unsafe_allow_html=True)
        dc = (dff.groupby('Discount Applied')['Actual_Subscription_Status']
              .apply(lambda x: (x=='Yes').mean()*100))
        fig, ax = plt.subplots(figsize=(5,3.5))
        bars = ax.bar(dc.index, dc.values,
                      color=['#EF4444','#10B981'], edgecolor='white', width=0.4)
        ax.bar_label(bars, fmt='%.1f%%', fontsize=10, padding=4)
        ax.set_ylabel('Churn Rate (%)'); ax.set_ylim(0, dc.max()+15)
        ax.set_title('Discount Applied vs Churn Rate', fontsize=11, fontweight='bold')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)

    # Review rating vs churn prob
    with c3:
        st.markdown('<div class="sec-hdr">Review Rating vs Churn Probability</div>',
                    unsafe_allow_html=True)
        rp = dff.groupby('Review Rating')['Churn_Probability'].mean()
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.plot(rp.index, rp.values, 'o-', color='#EF4444', lw=2.5, ms=7)
        ax.axhline(0.6, color='#F59E0B', ls='--', lw=1.5, label='Threshold 0.6')
        ax.set_xlabel('Review Rating'); ax.set_ylabel('Avg Churn Probability')
        ax.set_title('Rating vs Churn Risk', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Frequency vs churn
    with c4:
        st.markdown('<div class="sec-hdr">Purchase Frequency vs Churn</div>',
                    unsafe_allow_html=True)
        freq_order = ['Weekly','Bi-Weekly','Fortnightly','Monthly',
                      'Quarterly','Every 3 Months','Annually']
        freq_order = [f for f in freq_order
                      if f in dff['Frequency of Purchases'].unique()]
        fc = (dff.groupby('Frequency of Purchases')['Churn_Probability']
              .mean().reindex(freq_order, fill_value=np.nan).dropna())
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.barh(fc.index, fc.values, color='#1E3A5F', edgecolor='white', alpha=0.9)
        ax.set_xlabel('Avg Churn Probability')
        ax.set_title('Churn Risk by Purchase Frequency', fontsize=11, fontweight='bold')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Age distribution
    st.markdown('<div class="sec-hdr">Age Distribution by Risk Level</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10,3.8))
    for risk, clr in RISK_CLR.items():
        sub = dff[dff['Risk_Level']==risk]['Age']
        if len(sub)>5:
            sub.plot(kind='kde', ax=ax, color=clr, label=risk, lw=2.5)
    ax.set_xlabel('Customer Age'); ax.set_ylabel('Density')
    ax.set_title('Age Distribution by Churn Risk Level', fontsize=11, fontweight='bold')
    ax.legend()
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Subscription status cross-tab
    st.markdown('<div class="sec-hdr">Subscription Status vs Risk Level</div>',
                unsafe_allow_html=True)
    ct = pd.crosstab(dff['Subscription Status'], dff['Risk_Level'],
                     normalize='index') * 100
    fig, ax = plt.subplots(figsize=(7,3.5))
    ct.plot(kind='bar', ax=ax,
            color=[RISK_CLR.get(c,'#999') for c in ct.columns],
            edgecolor='white', width=0.5)
    ax.set_xlabel('Subscription Status'); ax.set_ylabel('Percentage (%)')
    ax.set_title('Risk Distribution by Subscription Status', fontsize=11, fontweight='bold')
    ax.legend(title='Risk Level', bbox_to_anchor=(1,1))
    plt.xticks(rotation=0)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Comparison")
    st.markdown("Logistic Regression  ·  Random Forest  ·  XGBoost — evaluated on 30% held-out test set")
    st.markdown("---")

    with st.spinner("Training models on your dataset…"):
        models, X_te, y_te, enc, sc, feat_names = train_models()

    # ── Metrics table ─────────────────────────
    st.markdown('<div class="sec-hdr">Performance Metrics — All Models</div>',
                unsafe_allow_html=True)

    rows = []
    preds = {}
    for label, key in [('Logistic Regression','LR'),
                        ('Random Forest','RF'),
                        ('XGBoost','XGB')]:
        m   = models[key]
        yp  = m.predict(X_te)
        ypr = m.predict_proba(X_te)[:,1]
        preds[label] = (yp, ypr)
        rows.append({
            'Model'    : label,
            'Accuracy' : round(accuracy_score(y_te, yp),4),
            'Precision': round(precision_score(y_te, yp, zero_division=0),4),
            'Recall'   : round(recall_score(y_te, yp, zero_division=0),4),
            'F1-Score' : round(f1_score(y_te, yp, zero_division=0),4),
            'ROC-AUC'  : round(roc_auc_score(y_te, ypr),4),
        })

    met_df = pd.DataFrame(rows).set_index('Model')

    def hi(s):
        return ['background-color:#D1FAE5;font-weight:bold'
                if v==s.max() else '' for v in s]

    st.dataframe(met_df.style.apply(hi).format("{:.4f}"),
                 use_container_width=True)
    st.caption("Green = best per metric · XGBoost is the selected primary model")

    c1, c2 = st.columns(2)

    # ROC curves
    with c1:
        st.markdown('<div class="sec-hdr">ROC Curve Comparison</div>',
                    unsafe_allow_html=True)
        clrs = {'Logistic Regression':'#2E86AB',
                'Random Forest':'#F59E0B',
                'XGBoost':'#10B981'}
        fig, ax = plt.subplots(figsize=(6,5))
        for name,(yp,ypr) in preds.items():
            fpr,tpr,ths = roc_curve(y_te, ypr)
            ra = auc(fpr,tpr)
            ax.plot(fpr, tpr, color=clrs[name], lw=2.2,
                    label=f'{name} (AUC={ra:.3f})')
            # mark threshold 0.6
            idx = next((i for i,t in enumerate(ths) if t<=0.6), -1)
            if idx>=0:
                ax.scatter(fpr[idx], tpr[idx], s=90, color=clrs[name], zorder=5)
        ax.plot([0,1],[0,1],'k--',lw=1,label='Random (AUC=0.500)')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves  (● = threshold 0.6)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Confusion matrix XGBoost
    with c2:
        st.markdown('<div class="sec-hdr">XGBoost — Confusion Matrix</div>',
                    unsafe_allow_html=True)
        yp_xgb,_ = preds['XGBoost']
        cm = confusion_matrix(y_te, yp_xgb)
        fig, ax = plt.subplots(figsize=(5,4.2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred: Retained','Pred: Churned'],
                    yticklabels=['Actual: Retained','Actual: Churned'],
                    annot_kws={'size':14})
        ax.set_title('XGBoost Confusion Matrix', fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # F1 bar
    st.markdown('<div class="sec-hdr">F1-Score Comparison</div>',
                unsafe_allow_html=True)
    f1s = {r['Model']: r['F1-Score'] for r in rows}
    fig, ax = plt.subplots(figsize=(8,3))
    bars = ax.bar(f1s.keys(), f1s.values(),
                  color=['#2E86AB','#F59E0B','#10B981'],
                  edgecolor='white', width=0.4)
    ax.bar_label(bars, fmt='%.4f', fontsize=10, padding=4)
    ax.set_ylabel('F1-Score'); ax.set_ylim(0, max(f1s.values())+0.12)
    ax.set_title('F1-Score by Model', fontsize=11, fontweight='bold')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════
#  PAGE 4 — RETENTION ACTIONS
# ══════════════════════════════════════════════
elif page == "🎯 Retention Actions":
    st.title("🎯 Retention Action Centre")
    st.markdown("Prescriptive recommendations and business impact quantification")
    st.markdown("---")

    hr_df = dff[dff['Risk_Level']=='High Risk']
    rev_risk = (hr_df['Purchase Amount'] * hr_df['Churn_Probability']).sum()

    # Monte Carlo
    mc = []
    for _ in range(1000):
        r = np.random.binomial(1, 0.7, size=max(len(hr_df),1))
        mc.append(r.sum()/max(len(hr_df),1))
    mc_mean = np.mean(mc)
    mc_lo   = np.percentile(mc,2.5)
    mc_hi   = np.percentile(mc,97.5)
    avg_val = dff['Purchase Amount'].mean()
    exp_sav = len(hr_df) * avg_val * mc_mean

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#EF4444">'
                    f'{len(hr_df):,}</p><p class="kpi-lbl">High Risk Customers</p></div>',
                    unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#7F1D1D">'
                    f'${rev_risk:,.0f}</p><p class="kpi-lbl">Revenue at Risk</p></div>',
                    unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#10B981">'
                    f'{mc_mean:.0%}</p><p class="kpi-lbl">Expected Churn Reduction</p></div>',
                    unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#1E3A5F">'
                    f'${exp_sav:,.0f}</p><p class="kpi-lbl">Estimated Revenue Saved</p></div>',
                    unsafe_allow_html=True)

    st.markdown(f'<div class="info-banner">📈 Monte Carlo (1,000 iterations, 70% success rate) · '
                f'95% CI: {mc_lo:.0%} – {mc_hi:.0%}</div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2 = st.columns(2)

    # Retention decision pie
    with c1:
        st.markdown('<div class="sec-hdr">Retention Decision Breakdown</div>',
                    unsafe_allow_html=True)
        rd = dff['Retention_Decision'].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        wedges,_,auts = ax.pie(
            rd.values, labels=rd.index,
            colors=['#10B981','#EF4444'],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2},
            textprops={'fontsize':9})
        for a in auts: a.set_fontweight('bold'); a.set_color('white')
        ax.set_title('Retention Decision Distribution', fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Revenue at risk by season
    with c2:
        st.markdown('<div class="sec-hdr">Revenue at Risk by Season</div>',
                    unsafe_allow_html=True)
        rs = (dff.groupby('Season')
              .apply(lambda x: (x['Purchase Amount']*x['Churn_Probability']).sum())
              .sort_values())
        fig, ax = plt.subplots(figsize=(5,4))
        ax.barh(rs.index, rs.values,
                color='#EF4444', edgecolor='white', alpha=0.85)
        ax.set_xlabel('Revenue at Risk ($)')
        ax.set_title('Revenue at Risk by Season', fontsize=11, fontweight='bold')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Monte Carlo chart
    st.markdown('<div class="sec-hdr">Monte Carlo Simulation — Retention Impact</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.hist(mc, bins=35, color='#2E86AB', edgecolor='white', alpha=0.88)
    ax.axvline(mc_mean, color='#EF4444', lw=2.5, ls='--',
               label=f'Mean = {mc_mean:.1%}')
    ax.axvline(mc_lo, color='#F59E0B', lw=1.5, ls=':',
               label=f'95% CI lower = {mc_lo:.1%}')
    ax.axvline(mc_hi, color='#F59E0B', lw=1.5, ls=':',
               label=f'95% CI upper = {mc_hi:.1%}')
    ax.set_xlabel('Churn Reduction Rate'); ax.set_ylabel('Frequency')
    ax.set_title('Monte Carlo Simulation — 1,000 Iterations', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Top 20 highest risk
    st.markdown('<div class="sec-hdr">Top 20 Highest Risk Customers</div>',
                unsafe_allow_html=True)
    cols_show = ['Customer_ID','Gender','Age','Category','Season',
                 'Purchase Amount','Review Rating','Churn_Probability',
                 'Risk_Level','Retention_Decision']
    top20 = dff.nlargest(20,'Churn_Probability')[cols_show].copy()
    top20['Churn_Probability'] = top20['Churn_Probability'].round(4)
    st.dataframe(top20, use_container_width=True)

    # Download button
    csv_dl = dff.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Results as CSV",
        data=csv_dl,
        file_name="churn_predictions_filtered.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════
#  PAGE 5 — LIVE PREDICTOR  (INTERACTIVE)
# ══════════════════════════════════════════════
elif page == "🔮 Live Predictor":
    st.title("🔮 Live Customer Churn Predictor")
    st.markdown("Enter a customer's details below and get an **instant churn risk assessment**.")
    st.markdown('<div class="info-banner">ℹ️ This predictor uses the trained XGBoost model '
                'from the project pipeline. All fields are required.</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Loading trained model…"):
        models, X_te, y_te, enc, sc, feat_names = train_models()

    # ── INPUT FORM ────────────────────────────
    with st.form("predict_form"):
        st.markdown("### 👤 Customer Details")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            age      = st.slider("Age", 18, 70, 35,
                                 help="Customer age in years")
            gender   = st.selectbox("Gender", ["Male","Female"])
            location = st.selectbox("Location", sorted([
                'Kentucky','Maine','Massachusetts','Rhode Island','Oregon',
                'Wyoming','Montana','Louisiana','West Virginia','Missouri',
                'Illinois','California','Texas','New York','Florida',
                'Ohio','Georgia','Michigan','North Carolina','Pennsylvania',
                'Virginia','Washington','Arizona','Indiana','Tennessee',
                'Minnesota','Colorado','Nevada','Alabama','South Carolina',
                'Maryland','Connecticut','Oklahoma','Iowa','Mississippi',
                'Arkansas','Utah','Kansas','Nevada','New Mexico','Nebraska',
                'Idaho','New Hampshire','Hawaii','Maine','Delaware',
                'Vermont','Alaska','North Dakota','South Dakota','Wyoming'
            ]))

        with c2:
            st.markdown("**Purchase Behaviour**")
            category    = st.selectbox("Product Category",
                                       ['Clothing','Footwear','Outerwear','Accessories'])
            item        = st.selectbox("Item Purchased", sorted([
                'Blouse','Sweater','Jeans','Sandals','Sneakers','Shirt',
                'Shorts','Coat','Handbag','Shoes','Dress','Jacket',
                'Scarf','Hat','Boots','Gloves','Socks','Belt','Sunglasses',
                'Watch','Backpack','Wallet','Skirt','Leggings','Hoodie'
            ]))
            purch_amt   = st.slider("Purchase Amount ($)", 20, 100, 60,
                                    help="Transaction value in USD")
            prev_purch  = st.slider("Previous Purchases", 1, 50, 10,
                                    help="Number of past transactions")

        with c3:
            st.markdown("**Engagement Signals**")
            season      = st.selectbox("Season",
                                       ['Spring','Summer','Fall','Winter'])
            review      = st.slider("Review Rating", 2.5, 5.0, 3.5, 0.1,
                                    help="Customer satisfaction (1–5)")
            discount    = st.selectbox("Discount Applied", ["Yes","No"])
            freq        = st.selectbox("Purchase Frequency", [
                'Weekly','Bi-Weekly','Fortnightly','Monthly',
                'Quarterly','Every 3 Months','Annually'
            ])
            subscription= st.selectbox("Subscription Status", ["Yes","No"],
                                       help="Does the customer have an active subscription?")

        submitted = st.form_submit_button(
            "🔍 Predict Churn Risk", use_container_width=True, type="primary")

    # ── PREDICTION ────────────────────────────
    if submitted:
        st.markdown("---")
        st.markdown("### 📊 Prediction Result")

        # ── Re-use the SAME encoders and scaler from train_models ──
        # (they are returned from the cached function so schema is identical)
        xgb_m = models['XGB']
        lr_m  = models['LR']
        rf_m  = models['RF']

        def safe_encode(le, val):
            """Encode a single value; return 0 if unseen."""
            v = str(val)
            if v in le.classes_:
                return int(le.transform([v])[0])
            return 0

        # Numeric Frequency mapping — same as train_models
        freq_map_pred = {
            'Weekly': 6, 'Bi-Weekly': 5, 'Fortnightly': 4,
            'Monthly': 3, 'Quarterly': 2, 'Every 3 Months': 2, 'Annually': 1
        }
        freq_num = freq_map_pred.get(freq, 3)

        # Build raw row with the EXACT columns the model was trained on
        row_dict = {}
        for col in feat_names:
            if col == 'Age':
                row_dict[col] = float(age)
            elif col == 'Review Rating':
                row_dict[col] = float(review)
            elif col == 'Previous Purchases':
                row_dict[col] = float(prev_purch)
            elif col == 'Purchase Amount':
                row_dict[col] = float(purch_amt)
            elif col == 'Frequency':
                row_dict[col] = float(freq_num)
            elif col == 'Monetary':
                row_dict[col] = float(purch_amt)
            elif col in enc:
                row_dict[col] = float(safe_encode(enc[col],
                    subscription if col == 'Subscription Status'
                    else gender   if col == 'Gender'
                    else item     if col == 'Item Purchased'
                    else category if col == 'Category'
                    else location if col == 'Location'
                    else season   if col == 'Season'
                    else discount if col == 'Discount Applied'
                    else freq     if col == 'Frequency of Purchases'
                    else ''))
            else:
                row_dict[col] = 0.0

        row = pd.DataFrame([row_dict], columns=feat_names)

        # Scale the same numeric columns the scaler was fitted on
        num_cols_sc = ['Age','Review Rating','Previous Purchases','Purchase Amount']
        num_cols_sc = [c for c in num_cols_sc if c in row.columns]
        row[num_cols_sc] = sc.transform(row[num_cols_sc].astype(float))

        # Final safety
        row = row.fillna(0).astype(float)
        row_fin = row[feat_names]

        prob_xgb = xgb_m.predict_proba(row_fin)[0][1]
        prob_lr  = lr_m.predict_proba(row_fin)[0][1]
        prob_rf  = rf_m.predict_proba(row_fin)[0][1]

        # Risk assignment
        if prob_xgb < 0.3:
            risk='Low Risk'; cls='pred-low'
            action='✅ No action required — customer is loyal.'
            emoji='🟢'
        elif prob_xgb < 0.6:
            risk='Medium Risk'; cls='pred-med'
            action='⚠️ Monitor closely — consider a loyalty reward.'
            emoji='🟡'
        else:
            risk='High Risk'; cls='pred-high'
            action='🚨 Target with a personalised retention offer immediately.'
            emoji='🔴'

        # ── Layout ──
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="kpi-card"><p class="kpi-val" '
                        f'style="color:{"#EF4444" if prob_xgb>0.6 else "#10B981"}">'
                        f'{prob_xgb:.1%}</p>'
                        f'<p class="kpi-lbl">XGBoost Churn Probability</p></div>',
                        unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="kpi-card"><p class="kpi-val" '
                        f'style="font-size:1.4rem">{emoji} {risk}</p>'
                        f'<p class="kpi-lbl">Risk Classification</p></div>',
                        unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="kpi-card"><p class="kpi-lbl" '
                        f'style="font-size:1rem;color:#1E3A5F;font-weight:600">{action}</p>'
                        f'<p class="kpi-lbl">Recommended Action</p></div>',
                        unsafe_allow_html=True)

        # ── Gauge ──
        fig, ax = plt.subplots(figsize=(5,3),
                               subplot_kw=dict(aspect='equal'))
        c_gauge = '#EF4444' if prob_xgb>0.6 else '#F59E0B' if prob_xgb>0.3 else '#10B981'
        theta2  = 180 - prob_xgb * 180
        ax.add_patch(mpatches.Wedge((0,0),1, 0,180, color='#E5E7EB'))
        ax.add_patch(mpatches.Wedge((0,0),1, theta2,180, color=c_gauge))
        ax.add_patch(mpatches.Wedge((0,0),0.62, 0,180, color='white'))
        ax.text(0, 0.05, f'{prob_xgb:.1%}', ha='center', va='center',
                fontsize=22, fontweight='bold', color=c_gauge)
        ax.text(0,-0.28,'Churn Probability', ha='center', va='center',
                fontsize=10, color='#6B7280')
        ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.6,1.2)
        ax.axis('off')
        plt.tight_layout()
        g_col, _ = st.columns([1,1])
        with g_col:
            st.pyplot(fig); plt.close()

        # ── Model comparison bar for this customer ──
        st.markdown('<div class="sec-hdr">Churn Probability Across All Models</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7,3))
        m_names = ['Logistic Regression','Random Forest','XGBoost']
        m_probs = [prob_lr, prob_rf, prob_xgb]
        bars = ax.bar(m_names, m_probs,
                      color=['#2E86AB','#F59E0B','#10B981'],
                      edgecolor='white', width=0.45)
        ax.bar_label(bars, fmt='%.3f', fontsize=11, padding=4)
        ax.axhline(0.6, color='#EF4444', ls='--', lw=1.5, label='Threshold 0.6')
        ax.set_ylabel('Churn Probability')
        ax.set_ylim(0,1.05)
        ax.set_title('Churn Probability by Model for This Customer',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Input summary ──
        st.markdown('<div class="sec-hdr">Customer Profile Summary</div>',
                    unsafe_allow_html=True)
        summary_df = pd.DataFrame({
            'Field': ['Age','Gender','Location','Category','Item','Purchase Amount',
                      'Previous Purchases','Season','Review Rating',
                      'Discount Applied','Purchase Frequency','Subscription'],
            'Value': [age, gender, location, category, item, f'${purch_amt}',
                      prev_purch, season, review, discount, freq, subscription]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Customer Churn Prediction Dashboard · Joy Muthoni Wanjiru · "
    "SCT213-C002-0004/2022 · JKUAT BSc Data Science · 2024</small></center>",
    unsafe_allow_html=True)
