import pickle
import streamlit as st
import cv2
import numpy as np
import pickle

# Load your trained model
with open('enhanced_emotion_detection_v1.2.h5.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title("Real-Time Emotion Prediction")

# Streamlit sidebar for user options
st.sidebar.title("Settings")
show_webcam = st.sidebar.checkbox("Show Webcam Feed", value=True)

# Video capture
cap = cv2.VideoCapture(0)  # Use the default camera (index 0)


# Function to preprocess frames
def preprocess_frame(frame):
    # Example preprocessing: Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))  # Assuming 48x48 input size for the model
    normalized = resized / 255.0  # Normalize pixel values
    return normalized.reshape(1, 48, 48, 1)  # Reshape for model input


# Main loop
if show_webcam:
    st.sidebar.write("Press 'Stop' to exit.")
    st.write("Webcam Feed:")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video.")
            break

        # Display the frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Preprocess and predict
        preprocessed = preprocess_frame(frame)
        prediction = model.predict(preprocessed)
        emotion = np.argmax(prediction)  # Assuming output is a probability array

        # Display prediction
        st.write(f"Predicted Emotion: {emotion}")

        # Press 'q' to exit in the Streamlit interface
        if st.button("Stop"):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
