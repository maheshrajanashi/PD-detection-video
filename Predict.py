import joblib
import pandas as pd
import numpy as onp
from feature_extraction import extract_features
import matplotlib.pyplot as plt
import seaborn as sns
import os

def drawFeatureImportance( data, full_dir_path):

    # Define feature groups with interpretation
    feature_groups = {
        "Wrist Movement (mm)": {
            "features": ["wrist_mvmnt_x_mean", "wrist_mvmnt_y_mean", "wrist_mvmnt_dist_mean"],
            "interpretation": "Movement range and variability (Bradykinesia/Rigidity)"
        },
        "Speed & Acceleration": {
            "features": ["speed_mean_denoised", "acceleration_mean_denoised"],
            "interpretation": "Slowness or jerky compensatory movement (Bradykinesia)"
        },
        "Amplitude Decline": {
            "features": ["amplitude_mean_denoised", "amplitude_decrement_slope_denoised"],
            "interpretation": "Progressive amplitude loss (Hypometria)"
        },
        "Tremor Frequency (Hz)": {
            "features": ["frequency_mean_denoised", "frequency_stdev_denoised"],
            "interpretation": "Tremor regularity and rate (Tremor Detection)"
        },
        "Period Variability": {
            "features": ["period_mean_denoised", "period_stdev_denoised", "periodEntropy_denoised"],
            "interpretation": "Tremor rhythm and regularity"
        },
        "Freezing Events": {
            "features": ["numFreeze_denoised", "numInterruptions_denoised"],
            "interpretation": "Freezing of movement"
        },
        "Signal Irregularity": {
            "features": ["aperiodicity_denoised", "periodVarianceNorm_denoised"],
            "interpretation": "Instability in timing or tremor patterns"
        }
    }

    # Plotting
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(len(feature_groups), 1, figsize=(10, 24))
    fig.suptitle("PD-Related Feature Analysis from Movement Data", fontsize=18, fontweight='bold')

    for idx, (title, group) in enumerate(feature_groups.items()):
        keys = group["features"]
        values = [data[k] for k in keys]
        labels = [k.replace("_denoised", "").replace("_mean", "").replace("_", " ").title() for k in keys]
        
        sns.barplot(x=values, y=labels, ax=axes[idx], palette="viridis")
        axes[idx].set_title(f"{title}: {group['interpretation']}", fontsize=12)
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(full_dir_path, "feature_importance.png"))



def plot_graph(predictions, full_dir_path):

    raw_score = predictions[0]  # Assuming the model returns a single score
    print("Raw scores:", raw_score)

    # Post-process predictions (e.g., rounding, clipping)
    predictions = onp.round(predictions)
    print("Rounded predictions:", predictions)
    predictions = onp.clip(predictions, 0, 4)
    

    # classify predictions into severity levels
    severity_levels = ["No Symptoms", "Mild", "Severe", "Critical"]
    predictions = [severity_levels[int(pred)] for pred in predictions]
    print(predictions)

    # Create a figure
    plt.figure(figsize=(12, 4))

    # Draw a horizontal line to represent the scale
    plt.hlines(1, 0, 1, colors='lightgray', linewidth=3)  # Line from 0 to 1

    # Add markers for severity levels (evenly spaced)
    positions = onp.linspace(0, 1, len(severity_levels))  # Evenly distribute points between 0 and 1
    for pos, label in zip(positions, severity_levels):
        plt.plot(pos, 1, 'o', color='lightblue')  # Markers for each severity level
        plt.text(pos, 1.005, label, ha='center', fontsize=10)  # Labels above markers

    # Highlight your score on the scale
    plt.plot(raw_score, 1, 'o', color='orange', markersize=10)  # Your score marker
    plt.text(raw_score, 1.15, f"Your Score: {raw_score} ({predictions[0]})", ha='center', color='orange', fontsize=12)

    # Formatting the plot
    plt.title("Severity Of Motor Performance")
    plt.xticks([])  # Remove x-axis ticks (only labels are shown)
    plt.yticks([])  # Remove y-axis ticks
    plt.xlim(-0.1, 1.1)  # Add some padding around the scale
    plt.tight_layout()

    plt.savefig(os.path.join(full_dir_path, "severity_prediction.png"))


def process_video(video_path, hand):
    """
    Process the video to extract features and save them.
    """

    # Define the video file and output directory
    output_path = "data"  # Replace with your desired output directory
    hand = "left"  # Specify the hand to analyze ("left" or "right")
    model_filename = f"super_model_{hand}.joblib"
    regressor = joblib.load(model_filename)
    print(f"Model loaded from {model_filename}")

    base_file = os.path.basename(video_path)
    base_file = base_file[0:base_file.find(".mp4")]

    print("Base file: %s"%(base_file))

    full_dir_path = os.path.join(output_path,base_file)
    if not os.path.exists(full_dir_path):
        os.mkdir(full_dir_path)

    # Extract features from the video
    print("Extracting features from the video...")
    features = extract_features(video_path, output_path, hand)
    drawFeatureImportance(features, full_dir_path)

    # Convert the features dictionary to a DataFrame for prediction
    features_df = pd.DataFrame([features])  # Convert to a single-row DataFrame
    print(features_df)

    # Ensure the feature columns match the model's training features
    # (Optional: Drop or reorder columns if necessary)

    # Make predictions
    print("Making predictions...")
    predictions = regressor.predict(features_df, predict_disable_shape_check=True)

    print("Predictions made.")
    # Print the raw predictions
    print("Raw predictions:", predictions)


    # plot graph of predictions
    plot_graph(predictions, full_dir_path)

    return full_dir_path


# run streamlit app

def Init():
    import streamlit as st
    import tempfile
    """
    Initialize the application.
    """
    st.title("Parkinsonism Motor State Analysis")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
    if uploaded_file:
        st.write("Processing Video...")

    # dropdown to select hand
        
    hand = st.selectbox("Select Hand", ["left", "right"])

    # Button to process the video
    if st.button("Process Video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
            hand = hand.lower()
            handCaps = hand.upper()
            print(f"Temporary file created at: {temp_path}")
            # spinner while processing
            with st.spinner(f"Processing {handCaps} hand video..."):
        
                outputPath = process_video(temp_path, hand)  # Process the video with the specified hand

                # display the output directory

                # display png images in the output directory
                st.write(f"Output saved to: {outputPath}")
                st.image(os.path.join(outputPath+"/"+handCaps, "finger_taps_vs_time_line.png"), caption="Finger Taps vs Time")
                st.image(os.path.join(outputPath, "feature_importance.png"), caption="Feature Importance")
                st.image(os.path.join(outputPath, "severity_prediction.png"), caption="Severity Prediction")
                
                vidPath = os.path.join(outputPath, handCaps, "output.mp4")
                st.success("Processed video!:"+vidPath)
                

                if os.path.exists(vidPath):
                    with open(vidPath, "rb") as video_file:
                        video_bytes = video_file.read()

                    st.download_button(
                        label="📥 Download Video",
                        data=video_bytes,
                        file_name="output.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("Video file not found at: " + vidPath)


if __name__ == "__main__":
    Init()  # Initialize the application
    # Uncomment the line below to run the Streamlit app directly
    # run_ui()  # Run the Streamlit UI