import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("viniyoga_model.h5")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, left_hand, right_hand])

# Viniyoga Classes
VINIYOGA_CLASSES = ["Viniyoga1", "Viniyoga2", "Viniyoga3", "Viniyoga4", "Viniyoga5"]

# Open Webcam (0 for default webcam, change if needed)
cap = cv2.VideoCapture(0)  # Webcam

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Unable to read from webcam.")
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process with MediaPipe
        results = holistic.process(image)
        image.flags.writeable = True

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints and predict
        keypoints = extract_keypoints(results)
        keypoints = np.expand_dims(keypoints, axis=0)  # Add batch dimension

        predictions = model.predict(keypoints)
        predicted_class = VINIYOGA_CLASSES[np.argmax(predictions)]

        # Display prediction
        cv2.putText(frame, f'Class: {predicted_class}', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show Webcam Output
        cv2.imshow('Viniyoga Classification - Webcam', frame)

        # Exit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("🔴 Exiting...")
            break

    # Release Resources
    cap.release()
    cv2.destroyAllWindows()
