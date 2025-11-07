import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import sys

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

# Path where extracted signals are stored
folder = "../data/signals"

# Check if folder exists and contains npy files
if not os.path.exists(folder):
    print(f"❌ Signals folder not found: {os.path.abspath(folder)}")
    sys.exit(1)

files = [f for f in os.listdir(folder) if f.endswith(".npy")]
if not files:
    print(f"⚠ No .npy signal files found in {os.path.abspath(folder)}")
    print("Run extract_green_signals.py first to generate signals.")
    sys.exit(1)

X, y = [], []
for filename in files:
    signal = np.load(os.path.join(folder, filename))
    label = 1 if "fake" in filename.lower() else 0
    X.append(extract_features(signal))
    y.append(label)

X, y = np.array(X), np.array(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(model, "../models/xgb_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("\n Model and scaler saved in ../models/")
