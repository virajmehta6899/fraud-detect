"""
Fraud Detection Dashboard — Streamlit
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Real-Time Fraud Detection System")
st.caption("XGBoost + LLM Explainability | ROC-AUC: 0.9813 | Recall: 87.76%")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Info")
    try:
        info = requests.get(f"{API_URL}/model-info").json()
        st.metric("Best Model", info["best_model"])
        st.metric("ROC-AUC", info["metrics"]["roc_auc"])
        st.metric("Recall", info["metrics"]["recall"])
        st.metric("Precision", info["metrics"]["precision"])
        st.metric("Features Used", info["features_used"])
        st.metric("Training Data", f"{info['total_transactions']:,} transactions")
        st.metric("Fraud Rate", info["fraud_rate"])
    except:
        st.error("API not reachable")

    st.divider()
    st.header("🧪 Load Sample")
    if st.button("Load Fraud Sample", type="primary"):
        sample = requests.get(f"{API_URL}/sample-fraud").json()
        st.session_state.sample = sample
        st.success("Fraud sample loaded!")

# ── Main ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Predict Transaction", "📊 Model Performance"])

with tab1:
    st.subheader("Enter Transaction Details")

    # Load sample if available
    sample = st.session_state.get("sample", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Transaction Info**")
        amount = st.number_input("Amount (€)", value=float(sample.get("amount", 100.0)), min_value=0.0)
        time = st.number_input("Time (seconds)", value=float(sample.get("time", 0.0)))

    with col2:
        st.markdown("**Features V1-V14**")
        v1  = st.number_input("V1",  value=float(sample.get("v1", 0.0)))
        v2  = st.number_input("V2",  value=float(sample.get("v2", 0.0)))
        v3  = st.number_input("V3",  value=float(sample.get("v3", 0.0)))
        v4  = st.number_input("V4",  value=float(sample.get("v4", 0.0)))
        v5  = st.number_input("V5",  value=float(sample.get("v5", 0.0)))
        v6  = st.number_input("V6",  value=float(sample.get("v6", 0.0)))
        v7  = st.number_input("V7",  value=float(sample.get("v7", 0.0)))
        v8  = st.number_input("V8",  value=float(sample.get("v8", 0.0)))
        v9  = st.number_input("V9",  value=float(sample.get("v9", 0.0)))
        v10 = st.number_input("V10", value=float(sample.get("v10", 0.0)))
        v11 = st.number_input("V11", value=float(sample.get("v11", 0.0)))
        v12 = st.number_input("V12", value=float(sample.get("v12", 0.0)))
        v13 = st.number_input("V13", value=float(sample.get("v13", 0.0)))
        v14 = st.number_input("V14", value=float(sample.get("v14", 0.0)))

    with col3:
        st.markdown("**Features V15-V28**")
        v15 = st.number_input("V15", value=float(sample.get("v15", 0.0)))
        v16 = st.number_input("V16", value=float(sample.get("v16", 0.0)))
        v17 = st.number_input("V17", value=float(sample.get("v17", 0.0)))
        v18 = st.number_input("V18", value=float(sample.get("v18", 0.0)))
        v19 = st.number_input("V19", value=float(sample.get("v19", 0.0)))
        v20 = st.number_input("V20", value=float(sample.get("v20", 0.0)))
        v21 = st.number_input("V21", value=float(sample.get("v21", 0.0)))
        v22 = st.number_input("V22", value=float(sample.get("v22", 0.0)))
        v23 = st.number_input("V23", value=float(sample.get("v23", 0.0)))
        v24 = st.number_input("V24", value=float(sample.get("v24", 0.0)))
        v25 = st.number_input("V25", value=float(sample.get("v25", 0.0)))
        v26 = st.number_input("V26", value=float(sample.get("v26", 0.0)))
        v27 = st.number_input("V27", value=float(sample.get("v27", 0.0)))
        v28 = st.number_input("V28", value=float(sample.get("v28", 0.0)))

    st.divider()

    col_pred, col_explain = st.columns(2)

    with col_pred:
        if st.button("🔍 Predict Only", type="secondary", use_container_width=True):
            payload = {
                "amount": amount, "time": time,
                "v1": v1, "v2": v2, "v3": v3, "v4": v4,
                "v5": v5, "v6": v6, "v7": v7, "v8": v8,
                "v9": v9, "v10": v10, "v11": v11, "v12": v12,
                "v13": v13, "v14": v14, "v15": v15, "v16": v16,
                "v17": v17, "v18": v18, "v19": v19, "v20": v20,
                "v21": v21, "v22": v22, "v23": v23, "v24": v24,
                "v25": v25, "v26": v26, "v27": v27, "v28": v28,
            }
            with st.spinner("Analyzing..."):
                resp = requests.post(f"{API_URL}/predict", json=payload)
            if resp.status_code == 200:
                st.session_state.result = resp.json()
            else:
                st.error(f"Error: {resp.json().get('detail')}")

    with col_explain:
        if st.button("🤖 Predict + LLM Explain", type="primary", use_container_width=True):
            payload = {
                "amount": amount, "time": time,
                "v1": v1, "v2": v2, "v3": v3, "v4": v4,
                "v5": v5, "v6": v6, "v7": v7, "v8": v8,
                "v9": v9, "v10": v10, "v11": v11, "v12": v12,
                "v13": v13, "v14": v14, "v15": v15, "v16": v16,
                "v17": v17, "v18": v18, "v19": v19, "v20": v20,
                "v21": v21, "v22": v22, "v23": v23, "v24": v24,
                "v25": v25, "v26": v26, "v27": v27, "v28": v28,
            }
            with st.spinner("Analyzing + generating explanation..."):
                resp = requests.post(f"{API_URL}/explain", json=payload)
            if resp.status_code == 200:
                st.session_state.result = resp.json()
            else:
                st.error(f"Error: {resp.json().get('detail')}")

    # ── Results ───────────────────────────────────────────────────────────────
    if "result" in st.session_state:
        result = st.session_state.result
        st.divider()
        st.subheader("📊 Results")

        # Verdict
        is_fraud = result["prediction"] == "FRAUD"
        col_v, col_s, col_c = st.columns(3)

        with col_v:
            if is_fraud:
                st.error(f"🚨 {result['prediction']}")
            else:
                st.success(f"✅ {result['prediction']}")

        with col_s:
            st.metric("Risk Score", f"{result['risk_score']}/100")

        with col_c:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["risk_score"],
            title={"text": "Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if is_fraud else "#2ecc71"},
                "steps": [
                    {"range": [0, 30], "color": "#d5f5e3"},
                    {"range": [30, 70], "color": "#fdebd0"},
                    {"range": [70, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        if result.get("top_features"):
            st.subheader("🔑 Top Contributing Features")
            feat_df = pd.DataFrame(result["top_features"])
            st.bar_chart(feat_df.set_index("feature")["importance"])

        # LLM Explanation
        if result.get("explanation"):
            st.subheader("🤖 LLM Explanation")
            st.info(result["explanation"])

with tab2:
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Comparison")
        comparison_data = {
            "Model": ["XGBoost", "Random Forest", "Logistic Regression", "LightGBM"],
            "ROC-AUC": [0.9813, 0.9777, 0.9705, 0.9391],
            "F1-Score": [0.4943, 0.4521, 0.0955, 0.4538],
            "Recall": [0.8776, 0.8673, 0.9184, 0.8265],
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    with col2:
        st.markdown("### Key Insights")
        st.markdown("""
        **Why XGBoost was selected:**
        - Highest ROC-AUC (0.9813) — best fraud/normal separation
        - Strong Recall (87.76%) — catches most fraud cases
        - Balanced performance across all metrics

        **Why Recall matters more than Precision:**
        - Missing real fraud = financial loss for bank
        - False alarm = minor inconvenience for customer
        - Banks optimize for high recall in fraud detection

        **Dataset Challenge:**
        - 284,807 transactions, only 492 fraud (0.17%)
        - Used SMOTE to balance training data
        - PCA-transformed features (V1-V28) protect privacy
        """)

    st.divider()
    st.markdown("### 📈 Model Performance Visualization")
    metrics_df = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "LightGBM", "Logistic Regression"],
        "ROC-AUC": [0.9813, 0.9777, 0.9391, 0.9705],
        "Recall": [0.8776, 0.8673, 0.8265, 0.9184],
        "F1-Score": [0.4943, 0.4521, 0.4538, 0.0955],
    })
    st.bar_chart(metrics_df.set_index("Model"))