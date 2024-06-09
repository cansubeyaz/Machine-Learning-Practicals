import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Load image in grayscale
image = cv2.imread(r"C:\Users\cansu\OneDrive\Desktop\PAML_2024\pepper.jpg", cv2.IMREAD_GRAYSCALE)

# Parameters for LBP
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(image, n_points, radius, method="uniform")

# Plot the original image and LBP image
fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))

ax[0].axis("off")
ax[0].set_title("Input Image")
ax[0].imshow(image, cmap = "gray")

ax[1].axis("off")
ax[1].set_title("Local Binary Pattern (LBP)")
ax[1].imshow(lbp, cmap = "gray")

plt.tight_layout()
plt.show()