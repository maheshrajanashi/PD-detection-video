import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize MediaPipe Pose and Hands module
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    fingertap_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        hand_results = hands.process(frame_rgb)
        
        if pose_results.pose_landmarks:
            keypoints = [(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark]
            keypoints_list.append(keypoints)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                fingertips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky
                fingertap_list.append([(lm.x, lm.y, lm.z) for lm in fingertips])
    
    cap.release()
    return np.array(keypoints_list), np.array(fingertap_list)

def extract_features(keypoints, fingertaps):
    keypoints = np.array(keypoints)
    fingertaps = np.array(fingertaps)
    velocities = np.diff(keypoints, axis=0)
    accelerations = np.diff(velocities, axis=0)
    
    tremor_index = np.std(accelerations)
    avg_velocity = np.mean(velocities)
    fingertap_variability = np.std(np.diff(fingertaps, axis=0)) if fingertaps.size > 0 else 0

    st.write("Tremor Index:", tremor_index)
    st.write("Average Velocity:", avg_velocity)
    st.write("Fingertap Variability:", fingertap_variability)

    
    return [tremor_index, avg_velocity, fingertap_variability]

# Simulated dataset loading
def load_dataset():
    np.random.seed(42)
    X = np.random.rand(100, 3)  # 100 samples, 3 features (tremor index, avg velocity, fingertap variability)
    y = np.random.randint(0, 2, 100)  # Binary classification (0 = Normal, 1 = Parkinsonism)
    return X, y

# Train the ML Model
def train_model():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100,
                                     max_depth=5,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return clf, X_test

# Interpret the model with SHAP
def interpret_model(clf, X_test):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

import tempfile

def run_ui():
    st.title("Parkinsonism Motor State Analysis")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmp_path = tmpfile.name

        st.video(uploaded_file)
        st.write("Processing Video...")

        keypoints, fingertaps = extract_keypoints(tmp_path)
        features = extract_features(keypoints, fingertaps)

        fingertap_variability = features[2] * 1000  # Scale for better visualization



        st.write("Predicted Motor State:", "Parkinsonism" if fingertap_variability > 100 else "Normal")


run_ui()
