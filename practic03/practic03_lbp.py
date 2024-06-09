import argparse
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Setup argparse
parser = argparse.ArgumentParser(description='Local Binary Patterns (LBP) Feature Extraction')
parser.add_argument('--image', action='store', default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\texture_snippets\strip_h\AA0004.jpg')
parser.add_argument('--radius', type=int, default=1, help='Radius for LBP')
parser.add_argument('--points', type=int, default=8, help='Number of points in LBP')
parser.add_argument('--method', default='default', help='Method for LBP computation')
flags = parser.parse_args()

# Load and convert image
image = imread(flags.image)
gray_image = rgb2gray(image)

# Compute LBP features
lbp_features = local_binary_pattern(gray_image, P=flags.points, R=flags.radius, method=flags.method)

# Plotting the images
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(gray_image, cmap=plt.cm.gray)
ax1.set_title('Grayscale Image')

ax2.axis('off')
ax2.imshow(lbp_features, cmap='gray')
ax2.set_title('LBP Image')

# Plot histogram
ax3.hist(lbp_features.ravel(), bins=flags.points * 2, range=(0, flags.points * 2), facecolor='0.5')
ax3.set_title('LBP Histogram')

plt.show()

## python Practical_01.py --image 'path/to/other_image.jpg' --radius 2 --points 16 --method 'uniform'
