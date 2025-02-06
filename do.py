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

# Classes
VINIYOGA_CLASSES = ["Viniyoga1", "Viniyoga2", "Viniyoga3", "Viniyoga4", "Viniyoga5"]

# Test with video file instead of webcam
cap = cv2.VideoCapture("Viniyoga3.mp4")  # Video file in the same directory

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video file or unable to read the video.")
            break

        # Convert frame to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process frame with MediaPipe
        results = holistic.process(image)
        image.flags.writeable = True

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints and prepare for prediction
        keypoints = extract_keypoints(results)
        keypoints = np.expand_dims(keypoints, axis=0)  # Add batch dimension

        # Predict the class
        predictions = model.predict(keypoints)
        predicted_class = VINIYOGA_CLASSES[np.argmax(predictions)]

        # Display prediction on the frame
        cv2.putText(frame, f'Class: {predicted_class}', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Viniyoga Testing - Video', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
