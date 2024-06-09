import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog

## Load the image in grayscale
image = cv2.imread(r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\pepper.jpg', cv2.IMREAD_GRAYSCALE)

# Compute HOG features and the HOG image for visualization
features, hog_image = hog(image, orientations = 8, pixels_per_cell = (16,16), cells_per_block = (1,1), visualize = True, channel_axis = None) ## no channel for grayscale image

fig,ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].axis("off")
ax[0].imshow(image, cmap = "gray")
ax[0].set_title("Input Image")

ax[1].axis("off")
ax[1].imshow(hog_image, cmap = "gray")
ax[1].set_title("Histogram of Oriented Gradients (HOG)")

plt.tight_layout()
plt.show()