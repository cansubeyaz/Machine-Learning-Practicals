import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Function to calculate min, max, mean, and standard deviation of each channel in the training set
def calculate_statistics(images):
    # Concatenate images along the first axis to calculate statistics across all images
    images_concatenated = np.concatenate(images, axis=0)
    # Calculate statistics for each channel
    min_vals = np.min(images_concatenated, axis=(0, 1))
    max_vals = np.max(images_concatenated, axis=(0, 1))
    mean_vals = np.mean(images_concatenated, axis=(0, 1))
    std_vals = np.std(images_concatenated, axis=(0, 1))
    return min_vals, max_vals, mean_vals, std_vals

# Function to perform min-max normalization
def min_max_normalization(image, min_vals, max_vals):
    return (image - min_vals) / (max_vals - min_vals)

# Function to perform mean-standard deviation normalization
def mean_std_normalization(image, mean_vals, std_vals):
    return (image - mean_vals) / std_vals

# Read in color snippet images
image_dir = r'C:\Users\Lenovo\Desktop\machine_learning\PAML_2024\colour_snippets\red'
images = []
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(image_dir, filename))
        if image is not None:
            # Normalize image to range [0, 1]
            image = image / 255.0
            images.append(image)

# Split images into training and evaluation sets
train_images, eval_images = train_test_split(images, test_size=0.1, random_state=42)

# Calculate statistics for the training set
min_vals, max_vals, mean_vals, std_vals = calculate_statistics(train_images)

# Display calculated statistics
print("Minimum Values:", min_vals)
print("Maximum Values:", max_vals)
print("Mean Values:", mean_vals)
print("Standard Deviation Values:", std_vals)
