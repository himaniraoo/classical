import cv2
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, left_hand, right_hand])

# Path to videos (Update this to your actual paths)
VIDEO_DIR = "/Users/himanipraneshrao/Desktop/sig/try/"
SAVE_DIR = "dataset/"
VINIYOGA_CLASSES = ["Viniyoga1", "Viniyoga2", "Viniyoga3", "Viniyoga4", "Viniyoga5"]

os.makedirs(SAVE_DIR, exist_ok=True)

# Process each video
for viniyoga in VINIYOGA_CLASSES:
    os.makedirs(os.path.join(SAVE_DIR, viniyoga), exist_ok=True)
    video_path = os.path.join(VIDEO_DIR, f"{viniyoga}.mp4")
    cap = cv2.VideoCapture(video_path)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            
            # Save keypoints
            np.save(os.path.join(SAVE_DIR, viniyoga, f"keypoints_{frame_count}.npy"), keypoints)
            frame_count += 1
    
    cap.release()

print("Keypoints extracted and saved successfully!")

# Load dataset
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(VINIYOGA_CLASSES)}

for viniyoga in VINIYOGA_CLASSES:
    for file in os.listdir(os.path.join(SAVE_DIR, viniyoga)):
        keypoints = np.load(os.path.join(SAVE_DIR, viniyoga, file))
        sequences.append(keypoints)
        labels.append(label_map[viniyoga])

X = np.array(sequences)
y = tf.keras.utils.to_categorical(labels, num_classes=len(VINIYOGA_CLASSES))

# Reshape X to have 3 dimensions
X = X.reshape((X.shape[0], X.shape[1], 1))  # Add a third dimension for features
print(f"Reshaped X: {X.shape}")  # Debug the new shape

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(VINIYOGA_CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, batch_size=32)

# Save model
model.save("viniyoga_model.h5")
print("Model trained and saved successfully!")
