import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_logistic_regression_pipeline.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

st.markdown(
    '''
    <style>
    .main-title {font-size: 38px; font-weight: 800; margin-bottom: 0;}
    .subtitle {font-size: 16px; opacity: .75; margin-bottom: 25px;}
    .risk {padding: 14px 18px; border-radius: 12px; font-size: 20px; font-weight: 700;}
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📉 ChurnGuard AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Customer churn prediction powered by Logistic Regression</div>',
    unsafe_allow_html=True
)

total_customers = len(df)
churn_rate = (df["Churn"].eq("Yes").mean() * 100)
avg_monthly = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{total_customers:,}")
c2.metric("Observed Churn", f"{churn_rate:.1f}%")
c3.metric("Avg. Monthly Charges", f"${avg_monthly:,.2f}")
c4.metric("Avg. Tenure", f"{avg_tenure:.1f} months")

st.divider()

tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "📊 Business Insights", "📋 Dataset"])

with tab1:
    st.subheader("Customer Risk Assessment")
    st.write("Enter customer details to estimate the probability of churn.")

    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique()))
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", sorted(df["Partner"].dropna().unique()))
        dependents = st.selectbox("Dependents", sorted(df["Dependents"].dropna().unique()))
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone = st.selectbox("Phone Service", sorted(df["PhoneService"].dropna().unique()))
        multiple = st.selectbox("Multiple Lines", sorted(df["MultipleLines"].dropna().unique()))
        internet = st.selectbox("Internet Service", sorted(df["InternetService"].dropna().unique()))

    with right:
        security = st.selectbox("Online Security", sorted(df["OnlineSecurity"].dropna().unique()))
        backup = st.selectbox("Online Backup", sorted(df["OnlineBackup"].dropna().unique()))
        device = st.selectbox("Device Protection", sorted(df["DeviceProtection"].dropna().unique()))
        tech = st.selectbox("Tech Support", sorted(df["TechSupport"].dropna().unique()))
        tv = st.selectbox("Streaming TV", sorted(df["StreamingTV"].dropna().unique()))
        movies = st.selectbox("Streaming Movies", sorted(df["StreamingMovies"].dropna().unique()))
        contract = st.selectbox("Contract", sorted(df["Contract"].dropna().unique()))
        paperless = st.selectbox("Paperless Billing", sorted(df["PaperlessBilling"].dropna().unique()))
        payment = st.selectbox("Payment Method", sorted(df["PaymentMethod"].dropna().unique()))
        monthly = st.number_input("Monthly Charges", min_value=0.0, value=float(df["MonthlyCharges"].median()))
        total = st.number_input("Total Charges", min_value=0.0, value=float(pd.to_numeric(df["TotalCharges"], errors="coerce").median()))

    if st.button("🚀 Predict Churn Risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(model.predict(input_df)[0])

        st.subheader("Prediction Result")
        r1, r2 = st.columns(2)
        r1.metric("Churn Probability", f"{probability * 100:.1f}%")
        r2.metric("Prediction", "Likely to Churn" if prediction else "Likely to Stay")

        if probability >= 0.70:
            st.error("🔴 High Risk — proactive retention action is recommended.")
        elif probability >= 0.40:
            st.warning("🟠 Medium Risk — monitor the customer and consider engagement.")
        else:
            st.success("🟢 Low Risk — customer is currently less likely to churn.")

        st.progress(probability)

with tab2:
    st.subheader("Churn Analysis")

    col1, col2 = st.columns(2)

    with col1:
        contract_churn = pd.crosstab(
            df["Contract"], df["Churn"], normalize="index"
        ).mul(100).round(2)
        st.write("**Churn Rate by Contract**")
        st.bar_chart(contract_churn.get("Yes", pd.Series(dtype=float)))

    with col2:
        internet_churn = pd.crosstab(
            df["InternetService"], df["Churn"], normalize="index"
        ).mul(100).round(2)
        st.write("**Churn Rate by Internet Service**")
        st.bar_chart(internet_churn.get("Yes", pd.Series(dtype=float)))

    st.write("**Churn by Tenure**")
    tenure_bins = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72, 100],
        labels=["0–6", "7–12", "13–24", "25–48", "49–72", "73+"]
    )
    tenure_churn = pd.crosstab(
        tenure_bins, df["Churn"], normalize="index"
    ).mul(100).round(2)
    st.bar_chart(tenure_churn.get("Yes", pd.Series(dtype=float)))

with tab3:
    st.subheader("Customer Dataset")
    st.dataframe(df, use_container_width=True, height=450)
    st.download_button(
        "⬇️ Download Churn Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="Churn.csv",
        mime="text/csv"
    )

st.divider()
st.caption("ChurnGuard AI • Logistic Regression • Scikit-learn • Streamlit")
