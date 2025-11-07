# app.py

import streamlit as st
import tempfile
from classify_utils import extract_green_signal_from_video, classify_signal
import numpy as np

st.set_page_config(page_title="PulseCheckAI - Deepfake Detection", layout="wide")
st.title(" PulseCheck AI - Deepfake Detection from Green Signals")
st.write("Upload a facial video, and we'll detect if it's real or deepfake using facial blood flow signals (PPG).")

# Upload section
uploaded_file = st.file_uploader("📤 Upload a face video (MP4 format)", type=["mp4"])

# Process
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
        st.video(video_path)

        # Extract and classify
        with st.spinner("🕵️ Extracting signal & analyzing..."):
            signal = extract_green_signal_from_video(video_path)

            if len(signal) < 30:
                st.error(" Couldn't extract a valid signal. Try a clearer video.")
            else:
                prediction = classify_signal(signal)
                st.success(f" Prediction: **{prediction.upper()}** video.")
