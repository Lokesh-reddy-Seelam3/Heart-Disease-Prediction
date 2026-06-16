import streamlit as st
import numpy as np
import pickle

# Load model files
model = pickle.load(open("heart_disease.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# --- REFINED HIGH-VISIBILITY GLASSMORPHISM INTERFACE ---
anime_background_url = "https://images.alphacoders.com/936/936173.png"

st.markdown(f"""
    <style>
    /* Main Background */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(rgba(16, 16, 24, 0.65), rgba(16, 16, 24, 0.75)), 
                    url("{anime_background_url}") no-repeat center center fixed;
        background-size: cover;
    }}
    
    [data-testid="stHeader"] {{
        background: transparent;
    }}
    
    /* Brighter Container Box for high visibility */
    .main .block-container {{
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 20px;
        padding: 40px !important;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4);
        margin-top: 30px;
        margin-bottom: 30px;
    }}
    
    /* Headings */
    h1 {{
        font-family: 'Segoe UI', sans-serif !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        text-shadow: 0 2px 10px rgba(255, 75, 107, 0.4);
        text-align: center;
        margin-bottom: 5px;
    }}
    
    .sub-title-text {{
        text-align: center;
        color: #f0f2f6;
        font-size: 1.1rem;
        margin-bottom: 35px;
        font-weight: 500;
    }}
    
    /* White Labels for Input Fields */
    label p {{
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }}
    
    /* HIGH CONTRAST INPUT BOXES (Black text on crisp light inputs) */
    div[data-baseweb="input"], div[data-baseweb="select"] {{
        background-color: rgba(255, 255, 255, 0.85) !important;
        border: 2px solid rgba(255, 255, 255, 1) !important;
        border-radius: 10px !important;
    }}
    
    /* Force input text & dropdown options to be solid black */
    input, div[data-testid="stMarkdownContainer"] p, .stSelectbox div {{
        color: #000000 !important;
        font-weight: 600 !important;
    }}
    
    /* Maintain white color ONLY for application structural elements, not form values */
    .sub-title-text, h1, h2, h3, h4 {{
        color: #ffffff !important;
    }}

    /* Predict Button Styling */
    div.stButton > button:first-child {{
        background: linear-gradient(45deg, #ff4b6b, #ff8f9d) !important;
        color: white !important;
        border: none !important;
        padding: 14px 30px !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(255, 75, 107, 0.5) !important;
    }}
    
    div.stButton > button:first-child:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(255, 75, 107, 0.7) !important;
    }}

    /* Custom classes for results area */
    .result-card-positive {{
        background: rgba(235, 64, 52, 0.25);
        border: 2px solid #eb4034;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }}
    
    .result-card-negative {{
        background: rgba(46, 204, 113, 0.25);
        border: 2px solid #2ecc71;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }}

    .dummy-button {{
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #ffffff;
        border: 1px dashed rgba(255, 255, 255, 0.4);
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 15px;
        cursor: not-allowed;
        opacity: 0.8;
    }}
    </style>
    """, unsafe_allow_html=True)

# App Content Headers
st.title("❤️ Cardiovascular Risk Assessment")
st.markdown('<p class="sub-title-text">Machine Learning Diagnostic Interface</p>', unsafe_allow_html=True)

# Grid Layout for Fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    
    sex = st.selectbox(
        "Sex", 
        [0, 1], 
        format_func=lambda x: "Female" if x == 0 else "Male"
    )
    
    cp = st.selectbox(
        "Chest Pain Type", 
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }.get(x)
    )
    
    bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=200)
    
    blood_suger = st.selectbox(
        "Fasting Blood Sugar", 
        [0, 1], 
        format_func=lambda x: "Normal (≤ 120 mg/dl)" if x == 0 else "High (> 120 mg/dl)"
    )

with col2:
    ecg = st.selectbox(
        "Resting ECG Results", 
        [0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }.get(x)
    )
    
    max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    
    exang = st.selectbox(
        "Exercise Induced Angina", 
        [0, 1], 
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    
    depression = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", 
        [0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }.get(x)
    )
    
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])

# Thalassemia mapping
thal = st.selectbox(
    "Thalassemia (Thal)", 
    [0, 1, 2, 3],
    format_func=lambda x: {
        0: "Normal",
        1: "Fixed Defect",
        2: "Reversible Defect",
        3: "Unknown / Other"
    }.get(x)
)

st.write(" ")
st.write(" ")

# Prediction Execution
if st.button("Generate ML Prediction"):
    features = np.array([[
        age, sex, cp, bp, chol, blood_suger, ecg, max_heart_rate, exang, depression, slope, ca, thal
    ]])

    # Data Pipeline transformations
    features_pca = pca.transform(features)
    features_scaled = scaler.transform(features_pca)

    # ML Model outputs
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)

    st.write("---")
    
    # Large format results rendered directly below the form
    if prediction[0] == 1:
        st.markdown(f"""
            <div class="result-card-positive">
                <h2 style="color: #ff4b4b !important; margin-bottom: 10px;">⚠️ Heart Disease Detected</h2>
                <p style="color: #ffffff !important; font-size: 1.2rem; font-weight: 500;">
                    The model identified significant clinical indicator patterns matching heart disease profiles.
                </p>
                <p style="color: #ff8f9d !important; font-size: 1.3rem; font-weight: 700; margin-top: 10px;">
                    Calculated Risk Probability: {probability[0][1]*100:.2f}%
                </p>
                            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card-negative">
                <h2 style="color: #2ecc71 !important; margin-bottom: 10px;">✅ No Heart Disease Detected</h2>
                <p style="color: #ffffff !important; font-size: 1.2rem; font-weight: 500;">
                    Patient parameters fall within expected standard reference baseline ranges.
                </p>
                <p style="color: #a3e4d7 !important; font-size: 1.3rem; font-weight: 700; margin-top: 10px;">
                    Calculated Risk Probability: {probability[0][1]*100:.2f}%
                </p>
            </div>
        """, unsafe_allow_html=True)