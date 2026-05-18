import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Page config
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")

# Custom CSS for colorful modern look
st.markdown(
    """
    <style>
        .main {
            background-color: #0f0f1a;
        }
        .stApp {
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        }
        .title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #a0a0b0;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        .result-box {
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            margin-top: 1.5rem;
        }
        .pneumonia-box {
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
            box-shadow: 0 8px 32px rgba(255, 65, 108, 0.4);
        }
        .normal-box {
            background: linear-gradient(135deg, #00b09b, #96c93d);
            box-shadow: 0 8px 32px rgba(0, 176, 155, 0.4);
        }
        .result-label {
            font-size: 2.5rem;
            font-weight: 800;
            color: white;
        }
        .result-sub {
            font-size: 1rem;
            color: rgba(255,255,255,0.85);
            margin-top: 0.5rem;
        }
        .confidence-text {
            font-size: 1.2rem;
            font-weight: 600;
            color: white;
            margin-top: 0.8rem;
        }
        .info-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-top: 1rem;
            color: #c0c0d0;
            font-size: 0.9rem;
        }
        .upload-label {
            color: #a0a0b0;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# Load model and config
@st.cache_resource
def load_model_and_config():
    model = tf.keras.models.load_model("pneumonia_densenet121_model8.h5")
    with open("model_config.json", "r") as f:
        config = json.load(f)
    return model, config


model, config = load_model_and_config()
THRESHOLD = config["threshold"]


# Preprocess image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Header
st.markdown('<div class="title">🫁 Pneumonia Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a chest X-ray to detect pneumonia using deep learning</div>',
    unsafe_allow_html=True,
)

# Upload
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    with st.spinner("Analyzing X-Ray..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]
        is_pneumonia = prediction > THRESHOLD
        confidence = prediction if is_pneumonia else 1 - prediction
        confidence_pct = round(confidence * 100, 2)

    # Result
    if is_pneumonia:
        st.markdown(
            f"""
            <div class="result-box pneumonia-box">
                <div class="result-label">⚠️ PNEUMONIA DETECTED</div>
                <div class="result-sub">This X-ray shows signs of pneumonia</div>
                <div class="confidence-text">Confidence: {confidence_pct:.2f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-box normal-box">
                <div class="result-label">✅ NORMAL</div>
                <div class="result-sub">No signs of pneumonia detected</div>
                <div class="confidence-text">Confidence: {confidence_pct:.2f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Info card
    st.markdown(
        f"""
        <div class="info-card">
            📊 <b>Model:</b> DenseNet121 &nbsp;|&nbsp;
            🎯 <b>Threshold:</b> {THRESHOLD} &nbsp;|&nbsp;
            🔬 <b>Dataset:</b> Kaggle Chest X-Ray Pneumonia &nbsp;|&nbsp;
            ⚕️ <b>False Negatives:</b> 0
        </div>
    """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <div class="info-card" style="text-align:center; padding: 2rem;">
            👆 Upload a chest X-ray image above to get started
        </div>
    """,
        unsafe_allow_html=True,
    )

# Disclaimer
st.markdown(
    """
    <div class="info-card" style="margin-top: 2rem; text-align: center;">
        ⚠️ <b>Disclaimer:</b> This tool is for educational and portfolio purposes only.
        It is not intended for real clinical diagnosis.
    </div>
""",
    unsafe_allow_html=True,
)
