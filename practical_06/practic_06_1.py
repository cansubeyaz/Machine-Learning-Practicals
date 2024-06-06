import numpy as np
import matplotlib.pyplot as plt
import pickle

# Step 1: Load the Data
with open(r'C:\Users\Lenovo\Desktop\PAML_2024\practical_06\mvg_data11.pkl', 'rb') as file:
    data = pickle.load(file)

# Inspect the data structure
print(f"Type of data: {type(data)}")
if isinstance(data, dict):
    for key, value in data.items():
        print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape if isinstance(value, np.ndarray) else 'N/A'}")

# Assuming the key for the data we need is 'X'
data = data['X']

# Step 2: Remove NaN and Infinite Values
def remove_nan_and_inf(data):
    is_finite = np.isfinite(data).all(axis=1)
    return data[is_finite]

data_clean = remove_nan_and_inf(data)

# Step 3: Detect and Remove Outliers
def z_score_filter(data, threshold=2.5):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scores = (data - mean) / std
    filtered_data = data[(np.abs(z_scores) < threshold).all(axis=1)]
    return filtered_data

def modified_z_score_filter(data, threshold=3.5):
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    modified_z_scores = 0.6745 * (data - median) / mad
    filtered_data = data[(np.abs(modified_z_scores) < threshold).all(axis=1)]
    return filtered_data

data_z_score_filtered = z_score_filter(data_clean)
data_modified_z_score_filtered = modified_z_score_filter(data_clean)

# Step 4: Visualize the Data
def plot_data(original, z_filtered, mod_z_filtered):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].scatter(original[:, 0], original[:, 1], c='blue', label='Original')
    axes[0].set_title('Original Data')

    axes[1].scatter(z_filtered[:, 0], z_filtered[:, 1], c='green', label='Z-score Filtered')
    axes[1].set_title('Z-score Filtered Data')

    axes[2].scatter(mod_z_filtered[:, 0], mod_z_filtered[:, 1], c='red', label='Modified Z-score Filtered')
    axes[2].set_title('Modified Z-score Filtered Data')

    for ax in axes:
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.legend()

    plt.tight_layout()
    plt.show()

plot_data(data_clean, data_z_score_filtered, data_modified_z_score_filtered)
