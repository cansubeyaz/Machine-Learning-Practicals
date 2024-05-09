import argparse
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
import matplotlib.pyplot as plt

# Setup argparse for command line options
parser = argparse.ArgumentParser(description='Sobel Edge Detection')
parser.add_argument('--image', action='store', default=r'C:\Users\Lenovo\Desktop\machine_learning\PAML_2024\texture_snippets\strip_h\AA0004.jpg')
flags = parser.parse_args()

# Load image and convert to grayscale
image = imread(flags.image)
gray_image = rgb2gray(image)


def mysobel(grayimage):
    rows, columns = grayimage.shape
    hor = np.zeros_like(grayimage)
    ver = np.zeros_like(grayimage)

    # Sobel kernels
    kh = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kv = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply kernels to each pixel
    for r in range(1, rows - 1):
        for c in range(1, columns - 1):
            snippet = grayimage[r - 1:r + 2, c - 1:c + 2]
            hor[r, c] = np.sum(kh * snippet)
            ver[r, c] = np.sum(kv * snippet)

    magnitude = np.sqrt(hor ** 2 + ver ** 2)
    return hor, ver, magnitude


# Calculate custom and skimage Sobel edges
horizontal_edges, vertical_edges, magnitude = mysobel(gray_image)
skimage_sobel = sobel(gray_image)

# Set up subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Display grayscale image
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Grayscale Image')
axes[0].axis('off')

# Display horizontal edges
axes[1].imshow(horizontal_edges, cmap='gray')
axes[1].set_title('Horizontal Edges')
axes[1].axis('off')

# Display vertical edges
axes[2].imshow(vertical_edges, cmap='gray')
axes[2].set_title('Vertical Edges')
axes[2].axis('off')

# Display magnitude
axes[3].imshow(magnitude, cmap='gray')
axes[3].set_title('Magnitude of Edges')
axes[3].axis('off')

# Display skimage sobel
axes[4].imshow(skimage_sobel, cmap='gray')
axes[4].set_title('Skimage Sobel')
axes[4].axis('off')

# Leave the last subplot empty
axes[5].axis('off')

plt.show()
