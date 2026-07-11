import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Lung Cancer Risk Predictor", 
    page_icon="🫁", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Premium Clean Theme UI Elements)
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.5);
        color: #ffffff !important;
        border: none !important;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .success-alert {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #10b981;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
    }
    .danger-alert {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🫁 Lung Cancer Prediction Dashboard")
st.markdown("Predict the likelihood of lung cancer based on demographic, behavioral, and clinical features using trained machine learning models.")
st.markdown("---")

# Setup model paths
model_dir = "models/ui_models"
model_files = {
    'Gradient Boosting Classifier': 'GradientBoosting_ui.pkl',
    'XGBoost Classifier': 'XGBoosting_ui.pkl',
    'Random Forest Classifier': 'RandomForest_ui.pkl',
    'Decision Tree Classifier': 'DecisionTree_ui.pkl',
    'Logistic Regression': 'LogisticRegression_ui.pkl'
}

# Loader functions cached
@st.cache_resource
def load_ui_model(model_name):
    model_path = os.path.join(model_dir, model_files[model_name])
    with open(model_path, 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_encoders():
    encoder_path = os.path.join(model_dir, 'ui_label_encoders.pkl')
    with open(encoder_path, 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_feature_list():
    feature_path = os.path.join(model_dir, 'ui_features.pkl')
    with open(feature_path, 'rb') as file:
        return pickle.load(file)

# Sidebar - model selection and description
st.sidebar.header("🔧 Settings & Model Selection")
selected_model_name = st.sidebar.selectbox("Select ML Classifier:", list(model_files.keys()))
st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Info**: The model predicts the probability of a **Positive (Yes)** or **Negative (No)** lung cancer prediction.
- **Gradient Boosting** and **XGBoost** generally offer the highest accuracy and F1 scores.
""")

# Load resources
try:
    model = load_ui_model(selected_model_name)
    encoders = load_encoders()
    ui_features = load_feature_list()
    resources_loaded = True

except Exception as e:
    st.error(f"❌ Error loading resources:\n\n{e}")
    resources_loaded = False

if resources_loaded:
    st.header("📋 Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics & Basic Info")
        age = st.slider("Patient Age", 18, 100, 50)
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        
    with col2:
        st.subheader("Exposures & Habits")
        smoking_status = st.selectbox("Smoking Status", encoders['Smoking_Status'].classes_)
        air_pollution = st.selectbox("Air Pollution Exposure", encoders['Air_Pollution_Exposure'].classes_)
        family_history = st.selectbox("Family History of Cancer", encoders['Family_History'].classes_)
        occupation_exposure = st.selectbox("Occupational Hazard Exposure", encoders['Occupation_Exposure'].classes_)

    with col3:
        st.subheader("Clinical Metrics & Access")
        healthcare_access = st.selectbox("Healthcare Access Quality", encoders['Healthcare_Access'].classes_)
        stage_at_diagnosis = st.selectbox("Stage at Diagnosis", encoders['Stage_at_Diagnosis'].classes_)
        cancer_type = st.selectbox("Cancer Type Subtype", encoders['Cancer_Type'].classes_)
        mortality_risk = st.slider("Mortality Risk (Computed Score)", 0.0, 1.0, 0.3)
        survival_prob = st.slider("5-Year Survival Probability", 0.0, 1.0, 0.7)

    # Compile and transform input features
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Smoking_Status': smoking_status,
        'Air_Pollution_Exposure': air_pollution,
        'Family_History': family_history,
        'Occupation_Exposure': occupation_exposure,
        'Healthcare_Access': healthcare_access,
        'Stage_at_Diagnosis': stage_at_diagnosis,
        'Cancer_Type': cancer_type,
        'Mortality_Risk': mortality_risk,
        '5_Year_Survival_Probability': survival_prob
    }])

    # Preprocess categorical features using saved LabelEncoders
    encoded_input = input_data.copy()
    for col, encoder in encoders.items():
        if col in encoded_input.columns:
            value = str(encoded_input.loc[0, col])

            if value not in encoder.classes_:
                st.error(f"Unknown value '{value}' for column '{col}'")
                st.stop()

        encoded_input[col] = encoder.transform([value])[0]

    # Reorder columns to match model training signature
    encoded_input = encoded_input[ui_features]

    st.markdown("---")

    # Predict Button
    if st.button("🔮 Run Lung Cancer Risk Prediction", use_container_width=True):
        # Obtain prediction and probability
        prediction = model.predict(encoded_input)[0]
        
        # Some models (like SVM/Ridge) might not support predict_proba natively, 
        # but our UI models are classifier models that support it
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(encoded_input)[0]
            confidence = proba[prediction] * 100
        else:
            proba = None
            confidence = 100.0

        st.subheader("Prediction Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.markdown(
            """
            <div class="danger-alert">
                🚨 <b>Result:</b> Positive for Lung Cancer (Yes)
            </div>
            """,
            unsafe_allow_html=True,
        )
            else:
                st.markdown(
            """
            <div class="success-alert">
                ✅ <b>Result:</b> Negative for Lung Cancer (No)
            </div>
            """,
            unsafe_allow_html=True,
        )
        with res_col2:
            st.metric(
                label="Prediction Confidence / Probability",
                value=f"{confidence:.2f}%",
                delta="Elevated Risk" if prediction == 1 else "Normal Range"
            )

        # Risk breakdown visualizer helper
        st.markdown("#### Patient Risk Summary")
        
        risk_indicators = []
        if smoking_status in ['Current Smoker', 'Former Smoker']:
            risk_indicators.append("• Smoking History (Active or Past)")
        if air_pollution == 'High':
            risk_indicators.append("• High Air Pollution Exposure")
        if family_history == 'Yes':
            risk_indicators.append("• Family History of Cancer")
        if occupation_exposure == 'Yes':
            risk_indicators.append("• Workplace Hazard Exposure")
        if mortality_risk > 0.5:
            risk_indicators.append("• High Mortality Risk Score")
        if survival_prob < 0.4:
            risk_indicators.append("• Low 5-Year Survival Expectancy")

        if len(risk_indicators) > 0:
            st.warning(f"**Identified Risk Indicators:**\n" + "\n".join(risk_indicators))
        else:
            st.info("No major clinical risk indicators identified for this profile.")
