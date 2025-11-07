import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh

def extract_green_signal_from_video(video_path, sample_rate=2, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    green_signal = []
    frame_count = 0

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
    return np.array(green_signal)

# === MAIN BATCH PROCESSING ===
real_dir = "../data/real"
fake_dir = "../data/fake"
output_dir = "../data/signals"

# Show where we're looking
print(f"📂 Looking for REAL videos in: {os.path.abspath(real_dir)}")
print(f"📂 Looking for FAKE videos in: {os.path.abspath(fake_dir)}")
print(f"💾 Signals will be saved to: {os.path.abspath(output_dir)}\n")

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Check if folders exist and contain videos
for label, folder in [("REAL", real_dir), ("FAKE", fake_dir)]:
    if not os.path.exists(folder):
        print(f"❌ {label} folder not found!")
    elif not any(fname.endswith(".mp4") for fname in os.listdir(folder)):
        print(f"⚠ No MP4 videos found in {label} folder.")

# Process videos
for cls, folder in [("real", real_dir), ("fake", fake_dir)]:
    for filename in os.listdir(folder):
        if filename.endswith(".mp4"):
            path = os.path.join(folder, filename)
            print(f"🔍 Extracting green signal from: {path}")
            signal = extract_green_signal_from_video(path)
            if len(signal) > 10:
                out_name = f"{cls}_{os.path.splitext(filename)[0]}.npy"
                np.save(os.path.join(output_dir, out_name), signal)
                print(f"✅ Saved signal: {out_name} ({len(signal)} samples)\n")
            else:
                print(f"⚠ Skipped {filename} (signal too short)\n")
