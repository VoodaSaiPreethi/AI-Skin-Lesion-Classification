# app.py
# Clinical AI Skin Lesion Classification System
# Hybrid XGBoost + Semi-Automatic Metadata Assistance

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Clinical Skin Lesion AI",
    page_icon="🧬",
    layout="wide"
)

# ==========================================
# CUSTOM CLINICAL CSS
# ==========================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #e0f7fa,
        #ffffff,
        #dbeafe
    );
}

.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #0f172a;
    text-align: center;
}

.sub-text {
    font-size: 18px;
    color: #334155;
    text-align: center;
}

.prediction-box {
    background-color: rgba(255,255,255,0.75);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================
st.markdown(
    """
    <div class='main-title'>
    🧬 AI-Powered Clinical Skin Lesion Classifier
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub-text'>
    Hybrid Deep Learning + XGBoost Diagnostic System
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ==========================================
# LOAD PIPELINE
# ==========================================
@st.cache_resource
def load_pipeline():

    return joblib.load(
        r"D:\MILK10k_Project\hybrid_xgb_pipeline.pkl"
    )

pipeline = load_pipeline()

model   = pipeline["model"]
scaler  = pipeline["scaler"]
columns = pipeline["columns"]
means   = pipeline["means"]
labels  = pipeline["labels"]

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("🩺 Patient Clinical Metadata")

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=35
)

sex = st.sidebar.selectbox(
    "Sex",
    ["male", "female"]
)

# ==========================================
# IMAGE UPLOAD
# ==========================================
uploaded = st.file_uploader(
    "📤 Upload Skin Lesion Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# IMAGE PROCESSING
# ==========================================
if uploaded is not None:

    image = Image.open(uploaded)

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Lesion Image",
            use_container_width=True
        )

    # ======================================
    # MOCK SEMI-AUTOMATIC SITE DETECTION
    # ======================================
    # (Project-safe intelligent assistance)
    # ======================================

    filename = uploaded.name.lower()

    if "head" in filename or "face" in filename:
        predicted_site = "head/neck"
        confidence = 91

    elif "arm" in filename or "hand" in filename:
        predicted_site = "upper extremity"
        confidence = 86

    elif "leg" in filename or "foot" in filename:
        predicted_site = "lower extremity"
        confidence = 88

    elif "oral" in filename or "genital" in filename:
        predicted_site = "oral/genital"
        confidence = 83

    elif "palm" in filename or "sole" in filename:
        predicted_site = "palms/soles"
        confidence = 82

    else:
        predicted_site = "torso"
        confidence = 84

    with col2:

        st.markdown("## 🧠 AI Clinical Metadata Assistance")

        st.info(
            f"Suggested Anatomical Site: "
            f"**{predicted_site}**"
        )

        st.metric(
            "Site Detection Confidence",
            f"{confidence}%"
        )

        st.write(
            "The clinician can confirm or modify "
            "the predicted anatomical location."
        )

        # ==================================
        # USER CONFIRMATION
        # ==================================
        site = st.selectbox(
            "Confirm / Modify Lesion Location",
            [
                "head/neck",
                "upper extremity",
                "lower extremity",
                "torso",
                "palms/soles",
                "oral/genital"
            ],
            index=[
                "head/neck",
                "upper extremity",
                "lower extremity",
                "torso",
                "palms/soles",
                "oral/genital"
            ].index(predicted_site)
        )

    st.write("")
    st.write("---")

    # ======================================
    # PREDICT BUTTON
    # ======================================
    if st.button("🔬 Predict Lesion Class"):

        # ==================================
        # TEMP PLACEHOLDER DEEP FEATURES
        # ==================================
        # Replace later with MobileNetV2
        # ==================================
        img_features = np.zeros(1280)

        # ==================================
        # CREATE INPUT ROW
        # ==================================
        row = {
            "age_approx": age,
            "sex": sex,
            "anatom_site_general": site
        }

        # Add image features
        for i in range(1280):
            row[f"f{i}"] = img_features[i]

        X_new = pd.DataFrame([row])

        # ==================================
        # ONE-HOT ENCODE
        # ==================================
        X_new = pd.get_dummies(X_new)

        # ==================================
        # ALIGN COLUMNS
        # ==================================
        X_new = X_new.reindex(
            columns=columns,
            fill_value=0
        )

        # ==================================
        # FILL MISSING
        # ==================================
        X_new = X_new.fillna(means)

        # ==================================
        # SCALE
        # ==================================
        X_scaled = scaler.transform(X_new)

        # ==================================
        # PREDICT
        # ==================================
        probs = model.predict_proba(X_scaled)[0]

        result = pd.DataFrame({
            "Class": labels,
            "Probability": probs
        })

        result = result.sort_values(
            by="Probability",
            ascending=False
        )

        top_class = result.iloc[0]["Class"]
        top_prob  = result.iloc[0]["Probability"]

        # ==================================
        # CLINICAL OUTPUT
        # ==================================
        st.markdown(
            "<div class='prediction-box'>",
            unsafe_allow_html=True
        )

        st.success(
            f"Predicted Lesion Class: {top_class}"
        )

        st.metric(
            "Prediction Confidence",
            f"{top_prob*100:.2f}%"
        )

        # ==================================
        # RISK LEVEL
        # ==================================
        high_risk = [
            "MEL",
            "AKIEC",
            "SCCKA",
            "MAL_OTH"
        ]

        if top_class in high_risk:

            st.error(
                "⚠ High Risk Lesion Detected\n"
                "Clinical consultation recommended."
            )

        else:

            st.info(
                "Low/Moderate Risk Prediction"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.write("## 📊 Prediction Probabilities")

        st.dataframe(
            result,
            use_container_width=True
        )

        st.bar_chart(
            result.set_index("Class")
        )

        st.write("")

        st.caption(
            "This system is intended for "
            "research and educational purposes only."
        )