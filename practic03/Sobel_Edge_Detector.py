import cv2
import matplotlib.pyplot as plt
import numpy as np

## Load the image in the grayscale
image = cv2.imread(r"C:\Users\Lenovo\Desktop\PAML_2024\pepper.jpg", cv2.IMREAD_GRAYSCALE)

## Compute the gradients using Sobel Operator
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1,0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0,1, ksize=3)

gradient_magnitude = np.sqrt(gradient_y**2 + gradient_x**2)

threshold = 50
edges = gradient_magnitude > threshold

fig, ax = plt.subplots(1,2, sharey=True, sharex=True)
ax[0].axis("off")
ax[0].set_title("Input Image")
ax[0].imshow(image, cmap = "gray")

ax[1].axis("off")
ax[1].set_title("Sobel Edge Detector")
ax[1].imshow(edges, cmap = "gray")

plt.tight_layout()
plt.show()
