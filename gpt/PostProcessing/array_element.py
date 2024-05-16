import numpy as np

# Assuming your numpy array is named 'array_3d'
# Replace array_3d with the name of your actual numpy array

# Generate a random 3D numpy array for demonstration
python_array = [
    [0, 255, 0],
    [255, 0, 255],
    [0, 255, 0],
    [0, 255, 0],
    [255, 255, 255]
]

# Convert to numpy array
numpy_array = np.array(python_array)

# Count occurrences
count = np.sum(np.all(numpy_array == [255, 0, 255], axis=1))

print("Number of occurrences:", count)
