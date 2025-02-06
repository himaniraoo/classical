import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Load model
model = tf.keras.models.load_model("viniyoga_model.h5")

# Define dataset path
DATASET_DIR = "dataset/"

# Load keypoint files from subdirectories
X_test = []
y_test = []

# Class names in order
VINIYOGA_CLASSES = ["Viniyoga1", "Viniyoga2", "Viniyoga3", "Viniyoga4", "Viniyoga5"]

# Loop through each class folder and load keypoints
for class_idx, class_name in enumerate(VINIYOGA_CLASSES):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_path):
        print(f"Warning: {class_path} not found, skipping...")
        continue

    # Get all .npy files in the class directory
    for file in os.listdir(class_path):
        if file.endswith(".npy"):
            file_path = os.path.join(class_path, file)
            keypoints = np.load(file_path)
            X_test.append(keypoints)
            y_test.append(class_idx)  # Assign class index as label

# Convert to NumPy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Check if data was loaded
if X_test.shape[0] == 0:
    print("Error: No keypoint files found in the dataset directory.")
    exit()

# Predict
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
