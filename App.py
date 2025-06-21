import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import shap
import streamlit as st
from PIL import Image
import tempfile
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features  # Assuming this is a separate module for feature extraction

# Initialize MediaPipe Pose and Hands module

mp_hands = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return [], [], 0, []
    
    keypoints_list = []
    fingertap_list = []
    fingertap_count = 0
    prev_touch = False
    cooldown = 0  # Prevents rapid duplicate counts
    frame_placeholder = st.empty()
    tap_timestamps = []
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
                touching = distance < 0.035 and abs(thumb_tip.y - index_tip.y) < 0.03  # Improved condition
                
                if touching and not prev_touch and cooldown == 0:
                    fingertap_count += 1  # Count only new taps
                    cooldown = 5  # Set cooldown to 5 frames
                    tap_timestamps.append(frame_number)
                prev_touch = touching
        
        if cooldown > 0:
            cooldown -= 1  # Reduce cooldown over time
        
        # Overlay fingertap count on video
        cv2.putText(frame_rgb, f'Taps: {fingertap_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to PIL image and display in Streamlit
        img = Image.fromarray(frame_rgb)
        frame_placeholder.image(img, caption="Processed Video", use_column_width=True)
        
        frame_number += 1
    
    cap.release()
    return np.array(keypoints_list), fingertap_count, tap_timestamps

def calculate_fingertap_variability(tap_timestamps):
    if len(tap_timestamps) < 2:
        return 0  # No variability with fewer than two taps
    
    intervals = np.diff(tap_timestamps)  # Time intervals between consecutive taps
    variability = np.std(intervals)  # Standard deviation of intervals
    return variability



# Simulated dataset loading
def load_dataset():
    np.random.seed(42)
    X = np.random.rand(100, 4)  # 100 samples, 4 features (tremor index, avg velocity, fingertap count, tap variability)
    y = np.random.randint(0, 2, 100)  # Binary classification (0 = Normal, 1 = Parkinsonism)
    return X, y

# Train the ML Model
@st.experimental_memo
def train_model():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return clf

# Interpret the model with SHAP
def interpret_model(clf, X_test):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

# Streamlit UI
def run_ui():
    st.title("Parkinsonism Motor State Analysis")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
    if uploaded_file:
        st.write("Processing Video...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        
            features   = extract_features(temp_path, "data", "left", labels=(0,""))
            print("Extracted Features:")
            print(features)

def withoutUI():
    # This function can be used for testing without the UI
    video_path = "drreinhard1.mp4"  # Replace with your video path
    features   = extract_features(video_path, "data", "left", labels=(0,""))
    
    print("Extracted Features:")
    print(features)


if __name__ == "__main__":
    run_ui()
