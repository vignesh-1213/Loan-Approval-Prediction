import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Loan Intelligence | AI Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM AESTHETICS ---
st.markdown("""
<style>
    :root {
        --primary: #4F46E5;
        --secondary: #10B981;
        --background: #0F172A;
        --surface: #1E293B;
        --text: #F8FAFC;
    }
    
    body {
        color: var(--text);
        background-color: var(--background);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }

    /* Glassmorphism containers */
    .glass-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.6);
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10B981;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('models/logistic_regression.pkl')
    # Patch for scikit-learn version mismatch (newer model loaded in older environment)
    if not hasattr(model, 'multi_class'):
        model.multi_class = 'ovr' # or 'auto'
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_assets()
except Exception as e:
    st.error("Error loading models. Please run `model_pipeline.py` first to generate models.")
    st.stop()

# --- HEADER ---
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem;'>🏦 Loan Intelligence Engine</h1>
    <p style='color: #94A3B8; font-size: 1.2rem; font-weight: 300;'>AI-Powered Credit Risk Assessment & Decision Automation</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: USER INPUTS ---
with st.sidebar:
    st.markdown("### 📋 Applicant Profile")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    
    st.markdown("### 💰 Financials")
    app_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500)
    coapp_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=0, value=150, step=10)
    loan_term = st.selectbox("Loan Amount Term (Months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
    
    st.markdown("### 📊 Credit Profiling")
    credit_history = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    predict_btn = st.button("Analyze Application 🚀")

# --- MAIN CONTENT ---
col1, col2 = st.columns([1, 1])

# Data Formatting Function
def prepare_input_data():
    data = {
        'ApplicantIncome': app_income,
        'CoapplicantIncome': coapp_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': 1.0 if "Good" in credit_history else 0.0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Gender_Male': 1 if gender == "Male" else 0,
        'Married_Yes': 1 if married == "Yes" else 0,
        'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
        'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }
    
    # Ensure all features match training data
    df_input = pd.DataFrame(columns=feature_names)
    df_input.loc[0] = 0 # initialize with 0
    
    for key, value in data.items():
        if key in feature_names:
            df_input.at[0, key] = value
            
    return df_input

if predict_btn:
    # 1. Prepare Data
    input_df = prepare_input_data()
    
    # 2. Scale Data
    input_scaled = scaler.transform(input_df)
    
    # 3. Predict
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob >= 0.55 else 0 # Using 0.55 conservative threshold as recommended
    
    # Calculate derived metrics for reasoning
    total_income = app_income + coapp_income
    emi = (loan_amount * 1000) / loan_term
    dti = (emi / total_income) * 100 if total_income > 0 else 100
    
    # --- RENDER RESULTS ---
    with st.container():
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.markdown("## 🧠 AI Assessment Decision")
        
        c1, c2, c3 = st.columns(3)
        
        if prediction == 1:
            decision_color = "#10B981"
            decision_text = "APPROVED"
            icon = "✅"
            risk_level = "Low Risk"
        else:
            decision_color = "#EF4444"
            decision_text = "REJECTED"
            icon = "❌"
            risk_level = "High Risk"
            
        with c1:
            st.markdown(f"""
            <div class='metric-card' style='border-top: 4px solid {decision_color};'>
                <div class='metric-label'>Final Decision</div>
                <div class='metric-value' style='color: {decision_color}; font-size: 2rem;'>{icon} {decision_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Approval Probability</div>
                <div class='metric-value' style='color: #3B82F6;'>{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Risk Assessment</div>
                <div class='metric-value' style='color: {decision_color}; font-size: 1.8rem;'>{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display explicit rejection reasoning
        if prediction == 0:
            reasons = []
            if "Bad" in credit_history:
                reasons.append("Poor Credit History (0.0)")
            if dti > 35:
                reasons.append(f"Unsafe Debt-to-Income Ratio ({dti:.1f}% > 35% safe limit)")
            if loan_amount > 500:
                reasons.append(f"Excessive Loan Amount requested (₹{loan_amount*1000:,})")
            
            if not reasons:
                reasons.append("Overall low confidence score based on historical demographic and financial patterns")
                
            st.error(f"**⚠️ Primary Reason(s) for Rejection:** {', '.join(reasons)}")
        
    # --- INTERPRETATION SECTION ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.markdown("### 📈 Probability Breakdown")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Score", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#818cf8"},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.2)",
                'steps': [
                    {'range': [0, 55], 'color': 'rgba(239, 68, 68, 0.3)'},
                    {'range': [55, 100], 'color': 'rgba(16, 185, 129, 0.3)'}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 55}
            }
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Note: Deployment threshold set at 55% for conservative risk management.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_b:
        st.markdown("<div class='glass-container' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("### 💡 Business Interpretation")
        
        if "Good" in credit_history:
            st.success("✔️ Positive Credit History heavily influences approval chances.")
        else:
            st.error("⚠️ Lack of Credit History is the primary bottleneck for this application.")
            
        total_income = app_income + coapp_income
        emi = (loan_amount * 1000) / loan_term
        dti = (emi / total_income) * 100 if total_income > 0 else 100
        
        st.markdown(f"**Total Household Income**: ₹{total_income:,.2f}")
        st.markdown(f"**Estimated Monthly Installment (EMI)**: ₹{emi:,.2f}")
        st.markdown(f"**Debt-to-Income Ratio (DTI)**: {dti:.1f}%")
        
        if dti > 30:
            st.warning("⚠️ The DTI ratio is quite high (>30%), indicating potential difficulty in repayment despite other positive factors.")
        else:
            st.info("ℹ️ The DTI ratio is healthy, supporting repayment capability.")
            
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Default landing screen
    st.markdown("""
    <div style='margin-top: 50px; text-align: center; color: #94A3B8;'>
        <h2>👈 Enter applicant details in the sidebar to generate an AI assessment.</h2>
        <p>The model analyzes financial and demographic data to predict default risk with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
