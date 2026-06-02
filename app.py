import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

model = load_model()

st.title("💳 Credit Card Fraud Detection")
st.markdown("### Real-time Fraud Detection using Random Forest (99.92% Accuracy)")

st.divider()

st.subheader("🔍 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Transaction Amount (₹)", 
                              min_value=0.0, value=100.0)

with col2:
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)

with col3:
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)
    v6 = st.number_input("V6", value=0.0)

if st.button("🔍 Predict", type="primary"):
    features = np.array([[v1, v2, v3, v4, v5, v6, 
                          0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,amount]])
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    if prediction[0] == 1:
        st.error(f"🚨 FRAUDULENT Transaction Detected! "
                 f"Confidence: {probability[0][1]*100:.2f}%")
    else:
        st.success(f"✅ Legitimate Transaction! "
                   f"Confidence: {probability[0][0]*100:.2f}%")

st.divider()
st.markdown("""
### 📊 Model Performance
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 99.82% | 0.72 |
| **Random Forest ✅** | **99.92%** | **0.88** |

**Developed by Shikha Sahu | B.Tech ECE-IoT | MMMUT 2025**
""")
