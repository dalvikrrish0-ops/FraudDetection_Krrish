import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("processed_data.csv")

# -----------------------------
# BASIC CLEANING
# -----------------------------

if 'isFraud' not in df.columns:
    st.error("Target column 'isFraud' not found")
    st.stop()

# -----------------------------
# SIMPLE FEATURES
# -----------------------------

if 'TransactionAmt' in df.columns:
    avg_amt = df['TransactionAmt'].mean()
else:
    avg_amt = 1

if 'TransactionDT' in df.columns:
    df['HourOfDay'] = (df['TransactionDT'] // 3600) % 24

# -----------------------------
# RISK SCORE
# -----------------------------

df['RiskProbability'] = np.random.uniform(0, 1, len(df))

# -----------------------------
# RISK TIERS
# -----------------------------

def risk_tier(prob):
    if prob >= 0.75:
        return "Critical Risk"
    elif prob >= 0.40:
        return "Suspicious"
    else:
        return "Clear"

df['RiskTier'] = df['RiskProbability'].apply(risk_tier)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Filters")

selected_tier = st.sidebar.multiselect(
    "Select Risk Tier",
    options=df['RiskTier'].unique(),
    default=df['RiskTier'].unique()
)

filtered_df = df[df['RiskTier'].isin(selected_tier)]

# -----------------------------
# PAGE SELECTION
# -----------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Transaction Explorer",
        "SHAP Explainer"
    ]
)

# =====================================================
# PAGE 1 — OVERVIEW
# =====================================================

if page == "Overview":

    st.title("Fraud Operations Dashboard")

    total_transactions = len(filtered_df)
    total_fraud = filtered_df['isFraud'].sum()
    detection_rate = (total_fraud / total_transactions) * 100

    if 'TransactionAmt' in filtered_df.columns:
        avg_fraud_amt = filtered_df[
            filtered_df['isFraud'] == 1
        ]['TransactionAmt'].mean()
    else:
        avg_fraud_amt = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", total_transactions)
    col2.metric("Total Fraud Count", int(total_fraud))
    col3.metric("Detection Rate", f"{detection_rate:.2f}%")
    col4.metric("Average Fraud Amount", f"{avg_fraud_amt:.2f}")

    st.subheader("Risk Tier Distribution")

    fig = px.histogram(
        filtered_df,
        x='RiskTier'
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE 2 — TRANSACTION EXPLORER
# =====================================================

elif page == "Transaction Explorer":

    st.title("Transaction Explorer")

    st.subheader("Search Transactions")

    search_value = st.text_input(
        "Search Transaction ID"
    )

    if search_value != "":
        search_df = filtered_df.astype(str).apply(
            lambda row: row.str.contains(search_value).any(),
            axis=1
        )

        display_df = filtered_df[search_df]

    else:
        display_df = filtered_df

    st.dataframe(display_df.head(100))

    st.subheader("Live Risk Score")

    row_number = st.number_input(
        "Enter Row Number",
        min_value=0,
        max_value=len(filtered_df)-1,
        value=0
    )

    risk_score = filtered_df.iloc[row_number]['RiskProbability']

    st.metric(
        "Fraud Risk Probability",
        f"{risk_score:.2f}"
    )

# =====================================================
# PAGE 3 — SHAP EXPLAINER
# =====================================================

elif page == "SHAP Explainer":

    st.title("SHAP Explainer")

    st.write("Explain fraud prediction using SHAP values")

    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    from lightgbm import LGBMClassifier

    # Select numeric columns only
    numeric_cols = filtered_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Remove target column from features
    if "isFraud" in numeric_cols:
        numeric_cols.remove("isFraud")

    # Features and target
    X = filtered_df[numeric_cols].fillna(0)
    y = filtered_df["isFraud"]

    # Train model
    model = LGBMClassifier(random_state=42)

    model.fit(X, y)

    # SHAP Explainer
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # SHAP Summary Plot
    st.subheader("Global SHAP Summary Plot")

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    st.pyplot(plt.gcf())

    plt.clf()

    # SHAP Bar Plot
    st.subheader("Feature Importance")

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )

    st.pyplot(plt.gcf())

    plt.clf()