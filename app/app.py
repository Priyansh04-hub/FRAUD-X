import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.investigation import FraudInvestigator


st.set_page_config(
    page_title="FRAUD-X",
    page_icon="🔐",
    layout="wide"
)


@st.cache_resource
def load_investigator():
    return FraudInvestigator()


@st.cache_data
def load_model_results():
    path = BASE_DIR / "reports" / "model_comparison.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def calculate_risk_sample(investigator, data, sample_size=2000):
    if len(data) > sample_size:
        sample = data.sample(sample_size, random_state=42)
    else:
        sample = data.copy()

    X_scaled, _ = investigator.prepare_transaction_batch(sample)

    fraud_probabilities = investigator.model.predict_proba(X_scaled)[:, 1]

    anomaly_scores = -investigator.anomaly_detector.score_samples(X_scaled)

    normalized_anomaly = (
        anomaly_scores - investigator.anomaly_min
    ) / (
        investigator.anomaly_max - investigator.anomaly_min
    )

    normalized_anomaly = normalized_anomaly.clip(0, 1)

    risk_scores = (
        fraud_probabilities * investigator.risk_engine.fraud_weight
        + normalized_anomaly * investigator.risk_engine.anomaly_weight
    )

    risk_levels = pd.cut(
        risk_scores,
        bins=[-float("inf"), 0.40, 0.75, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    return pd.DataFrame({
        "risk_score": risk_scores,
        "risk_level": risk_levels
    })


st.title("FRAUD-X")
st.subheader("AI-Powered Financial Fraud Detection and Investigation Platform")

st.sidebar.title("FRAUD-X Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a transaction CSV to begin.")

    st.markdown(
        """
        ### FRAUD-X Modules

        **Data Analysis**  
        Transaction statistics, fraud distribution and transaction patterns.

        **Machine Learning**  
        Supervised fraud classification using trained machine learning models.

        **Anomaly Detection**  
        Isolation Forest identifies unusual transaction behaviour.

        **Risk Engine**  
        Combines fraud probability and anomaly score into a unified risk score.

        **Investigation Engine**  
        Generates transaction-level investigation reports.

        **Explainability**  
        Displays important model features associated with the prediction.
        """
    )

    st.stop()


data = pd.read_csv(uploaded_file)

investigator = load_investigator()
model_results = load_model_results()

st.sidebar.success(f"{len(data):,} transactions loaded")

st.header("Transaction Analytics")

total_transactions = len(data)

if "is_fraud" in data.columns:
    fraud_transactions = int(data["is_fraud"].sum())
    legitimate_transactions = total_transactions - fraud_transactions
    fraud_rate = fraud_transactions / total_transactions
else:
    fraud_transactions = 0
    legitimate_transactions = total_transactions
    fraud_rate = 0

if "amt" in data.columns:
    total_amount = data["amt"].sum()
else:
    total_amount = 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

with col3:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2%}"
    )

with col4:
    st.metric(
        "Transaction Volume",
        f"${total_amount:,.2f}"
    )


st.divider()

col1, col2 = st.columns(2)

with col1:
    if "is_fraud" in data.columns:
        fraud_chart_data = pd.DataFrame({
            "Class": ["Legitimate", "Fraud"],
            "Transactions": [
                legitimate_transactions,
                fraud_transactions
            ]
        })

        fig = px.bar(
            fraud_chart_data,
            x="Class",
            y="Transactions",
            title="Fraud vs Legitimate Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

with col2:
    if "amt" in data.columns:
        fig = px.histogram(
            data,
            x="amt",
            nbins=50,
            title="Transaction Amount Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


if "category" in data.columns and "is_fraud" in data.columns:
    st.subheader("Fraud by Transaction Category")

    category_data = (
        data.groupby("category")["is_fraud"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    category_data.columns = [
        "Category",
        "Fraud Transactions"
    ]

    fig = px.bar(
        category_data,
        x="Category",
        y="Fraud Transactions",
        title="Fraud Transactions by Category"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()

st.header("Model Performance")

if not model_results.empty:
    display_results = model_results.copy()

    numeric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]

    for column in numeric_columns:
        if column in display_results.columns:
            display_results[column] = display_results[column].round(4)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    metric_chart = model_results.melt(
        id_vars=["Model"],
        value_vars=[
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        metric_chart,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Performance Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()

st.header("Risk Distribution")

try:
    risk_sample = calculate_risk_sample(
        investigator,
        data,
        sample_size=2000
    )

    risk_counts = (
        risk_sample["risk_level"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"])
        .fillna(0)
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Transactions"
    ]

    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Transactions",
        title="Risk Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Risk distribution is calculated on a sample of up to 2,000 uploaded transactions."
    )

except Exception:
    st.info(
        "Risk distribution is unavailable for this uploaded dataset."
    )


st.divider()

st.header("Transaction Investigation")

st.dataframe(
    data.head(100),
    use_container_width=True,
    hide_index=True
)

max_index = len(data) - 1

transaction_index = st.number_input(
    "Transaction index",
    min_value=0,
    max_value=max_index,
    value=0,
    step=1
)

if st.button(
    "Investigate Transaction",
    type="primary"
):
    transaction = data.iloc[int(transaction_index)]

    investigation = investigator.investigate(
        transaction
    )

    risk = investigation["risk"]

    st.subheader("Investigation Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{risk['fraud_probability']:.2%}"
        )

    with col2:
        st.metric(
            "Anomaly Score",
            f"{risk['anomaly_score']:.4f}"
        )

    with col3:
        st.metric(
            "Risk Score",
            f"{risk['risk_score']:.2%}"
        )

    with col4:
        st.metric(
            "Risk Level",
            risk["risk_level"]
        )

    st.subheader("Investigation Report")

    report_text = investigator.format_investigation(
        investigation
    )

    st.text(report_text)