import numpy as np

# Class labels should be in quotes (strings)
labels = ["Viniyoga1", "Viniyoga2", "Viniyoga3", "Viniyoga4", "Viniyoga5"]

# Save labels to a file
np.save("dataset/labels.npy", labels)

print("labels.npy file created successfully!")
