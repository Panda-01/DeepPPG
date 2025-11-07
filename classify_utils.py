# classify_utils.py

import numpy as np
import cv2
import mediapipe as mp
import joblib
from scipy.signal import butter, filtfilt
import os

# Load trained model and scaler
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Bandpass filter for PPG (0.7–4 Hz ~ 42–240 BPM)
def bandpass_filter(signal, fs=30, low=0.7, high=4.0, order=4):
    nyquist = 0.5 * fs
    low_cut = low / nyquist
    high_cut = high / nyquist
    b, a = butter(order, [low_cut, high_cut], btype='band')
    return filtfilt(b, a, signal)

# Green signal extraction (matching training pipeline)
def extract_green_signal_from_video(video_path, sample_rate=2, max_frames=300, fs=30):
    cap = cv2.VideoCapture(video_path)
    green_signal = []
    frame_count = 0
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while cap.isOpened() and len(green_signal) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    pt = results.multi_face_landmarks[0].landmark[234]
                    x, y = int(pt.x * w), int(pt.y * h)
                    x1, x2 = max(x - 5, 0), min(x + 5, w)
                    y1, y2 = max(y - 5, 0), min(y + 5, h)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size != 0:
                        green = roi[:, :, 1].mean()
                        green_signal.append(green)

            frame_count += 1

    cap.release()
    green_signal = np.array(green_signal)

    # Apply bandpass filter if enough frames
    if len(green_signal) > fs * 2:
        green_signal = bandpass_filter(green_signal, fs=fs)

    return green_signal

# Feature extraction (same as in train_classifier.py)
def extract_features(signal, fs=30):
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fs)
    magnitudes = np.abs(fft)[:len(signal)//2]
    peak_freq = freqs[np.argmax(magnitudes)] * 60
    peak_mag = np.max(magnitudes)
    return [
        peak_freq,
        peak_mag,
        np.var(signal),
        np.mean(signal),
        np.std(signal)
    ]

# Classification
def classify_signal(signal, fs=30):
    features = extract_features(signal, fs)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return "Fake" if prediction == 1 else "Real"
