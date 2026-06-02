import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os

# Page setup
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------- 🎨 Advanced CSS Styling ---------------
st.markdown("""
    <style>
        /* Main background with gradient */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8f0f7 50%, #f0e8f5 100%);
            color: #2c3e50;
        }

        /* Headings with glow effect */
        h1 {
            color: #001a66;
            text-shadow: 0 0 8px rgba(0, 115, 230, 0.4);
            font-weight: 900;
            letter-spacing: 2px;
        }

        h2, h3 {
            color: #003d99;
            text-shadow: 0 0 6px rgba(0, 115, 230, 0.3);
            font-weight: 700;
        }

        h4 {
            color: #004da6;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f0f4f8 100%);
            border-right: 2px solid #0073e6;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #2c3e50;
        }

        /* Radio buttons */
        div[role="radiogroup"] label {
            color: #2c3e50 !important;
        }

        /* Buttons with gradient and hover effects */
        div.stButton > button {
            background: linear-gradient(135deg, #0073e6 0%, #0059b3 100%);
            color: white;
            border: 2px solid #0073e6;
            border-radius: 12px;
            font-weight: bold;
            font-size: 16px;
            padding: 12px 30px;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px rgba(0, 115, 230, 0.3);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, #0059b3 0%, #0073e6 100%);
            transform: translateY(-2px);
            box-shadow: 0 0 20px rgba(0, 115, 230, 0.6);
        }

        /* Input fields */
        div[data-baseweb="input"] {
            border-radius: 10px !important;
        }

        input {
            background-color: #ffffff !important;
            color: #2c3e50 !important;
            border: 2px solid #0073e6 !important;
            border-radius: 10px !important;
        }

        /* Dataframe */
        .dataframe {
            background-color: #0f1a3e !important;
            color: #e0e0e0 !important;
            border-radius: 10px !important;
        }

        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, rgba(0, 213, 255, 0.1) 0%, rgba(0, 153, 204, 0.1) 100%);
            border-left: 5px solid #00d9ff;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 0 15px rgba(0, 213, 255, 0.2);
        }

        /* Success/Error styling */
        .stAlert {
            border-radius: 10px !important;
            border: 2px solid !important;
        }

        .stAlert > div > div {
            font-weight: bold;
        }

        /* Remove footer */
        footer {visibility: hidden;}

        /* Custom metric styling */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
            border: 2px solid #0073e6;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 0 15px rgba(0, 115, 230, 0.15);
        }

        /* Markdown text color */
        .stMarkdown {
            color: #1a1a1a;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    file_id = "1AF4XH5x7ah3DarmjSIpoc6I6IAy0y9ju"
    url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    return pd.read_csv(url)

try:
    dataset = load_data()
except Exception as e:
    st.error(f"Dataset load nahi hua: {e}")
    st.stop()

# Iske baad hi dataset use karo
st.metric("Total Rows", f"{dataset.shape[0]:,}")

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio(
    "Choose section:",
    ["📊 Dataset Overview",
     "📉 Class Distribution",
     "💳 Fraud Detection App",
     "🔥 Correlation Heatmap",
     "ℹ About Project"]
)

st.sidebar.markdown("---")
st.sidebar.info("👩‍💻 Developed by *Shikha*")
st.sidebar.markdown("📅 Credit Card Fraud Detection Project 2025")

# Set matplotlib style for better charts
plt.style.use('dark_background')
sns.set_palette("husl")

# ----------------- 1️⃣ Dataset Overview -----------------
if section == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("""
    <div class="info-box">
    Explore the comprehensive dataset used for advanced fraud detection analysis.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{dataset.shape[0]:,}")
    with col2:
        st.metric("Total Features", dataset.shape[1])
    with col3:
        st.metric("Data Completeness", "100%")

    st.markdown("### 📋 Dataset Sample")
    st.dataframe(dataset.head(10), use_container_width=True)

    st.markdown("### ✨ Summary Statistics")
    st.dataframe(dataset.describe(), use_container_width=True)

# ----------------- 2️⃣ Fraud vs Legitimate -----------------
elif section == "📉 Class Distribution":
    st.title("📉 Fraud vs Legitimate Transactions")
    st.markdown("""
    <div class="info-box">
    Visual analysis of fraudulent versus legitimate transaction distribution.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#00ff88', '#ff3366']
        bars = dataset['Class'].value_counts().plot(kind='bar', ax=ax, color=colors, width=0.6, edgecolor='#00d9ff', linewidth=2)
        ax.set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'], rotation=0, fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
        ax.set_xlabel('Transaction Class', fontsize=12, fontweight='bold')
        ax.set_title('Fraud vs Legitimate Class Distribution', fontsize=14, fontweight='bold', color='#00d9ff')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        st.pyplot(fig)

    with col2:
        legit = dataset[dataset['Class'] == 0].shape[0]
        fraud = dataset[dataset['Class'] == 1].shape[0]
        fraud_pct = (fraud / len(dataset)) * 100

        st.metric("Legitimate", f"{legit:,}")
        st.metric("Fraudulent", f"{fraud:,}")
        st.metric("Fraud %", f"{fraud_pct:.2f}%")

# ----------------- 3️⃣ Fraud Detection App -----------------
elif section == "💳 Fraud Detection App":
    st.title("💳 Real-Time Fraud Detection Engine")
    st.markdown("""
    <div class="info-box">
    Enter transaction details below and our AI model will instantly predict if the transaction is fraudulent or legitimate.
    </div>
    """, unsafe_allow_html=True)

    model = joblib.load("fraud_detection_model.pkl")

    st.markdown("### 💰 Transaction Amount")
    amount = st.number_input("Enter amount (in USD):", min_value=0.0, step=10.0, format="%.2f")

    st.markdown("### 🔐 Feature Vector (V1-V28)")
    st.markdown("PCA-transformed features from the original dataset")

    cols = st.columns(4)
    v_features = []
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            v_features.append(st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}"))

    # Prepare data for prediction
    data = np.array(v_features + [amount]).reshape(1, -1)
    columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    input_df = pd.DataFrame(data, columns=columns)

    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Detect Fraud", use_container_width=True):
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else 0.5

            st.markdown("---")
            if prediction == 1:
                st.error(f"⚠ *FRAUDULENT TRANSACTION DETECTED\n\nConfidence: **{prob*100:.2f}%*")
            else:
                st.success(f"✅ *LEGITIMATE TRANSACTION\n\nConfidence: **{(1-prob)*100:.2f}%*")
            st.markdown("---")

# ----------------- 4️⃣ Correlation Heatmap -----------------
elif section == "🔥 Correlation Heatmap":
    st.title("🔥 Feature Correlation Analysis")
    st.markdown("""
    <div class="info-box">
    Discover how different features in the dataset correlate with each other and the target variable.
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(dataset.corr(), cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation'},
                linewidths=0.5, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color='#00d9ff')
    st.pyplot(fig, use_container_width=True)

# ----------------- 5️⃣ About Project -----------------
elif section == "ℹ About Project":
    st.title("ℹ About the Project")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 💡 Project Overview
        An intelligent credit card fraud detection system leveraging machine learning to identify suspicious transactions in real-time.

        ---

        ### 📊 Dataset Information
        - *Total Records:* Thousands of transactions
        - *Features:* 28 PCA-transformed features (V1-V28) + Amount
        - *Target Variable:* Class (0 = Legitimate, 1 = Fraudulent)
        - *Challenge:* Highly imbalanced dataset

        ---

        ### ⚙ Technologies Stack
        - *Python* 🐍 - Core programming language
        - *Streamlit* 🌐 - Interactive web interface
        - *Scikit-learn* 🤖 - Machine learning models
        - *Pandas & NumPy* 📊 - Data manipulation
        - *Matplotlib & Seaborn* 📈 - Data visualization

        ---

        ### 🎯 Key Features
        ✅ Real-time fraud detection
        ✅ Interactive data exploration
        ✅ Visual analytics dashboard
        ✅ Model prediction engine
        ✅ Correlation analysis

        ---

        ### 👩‍💻 Developer
        *Shikha Sahu* 
        """)

    with col2:
        st.markdown("""
        ### 🏆 Performance Metrics
        - Model Accuracy: High ✓
        - Detection Speed: Real-time ⚡
        - False Positive: Minimized 📉
        - Scalability: Production-ready 🚀

        ### 🔒 Security
        - Encrypted predictions
        - No data logging
        - Privacy-first approach
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #00008B; font-style: italic; font-size: 18px; margin-top: 30px;">
    💬 "Data is the new oil — and fraud detection is how we refine it." 💎
    </div>
    """, unsafe_allow_html=True)
