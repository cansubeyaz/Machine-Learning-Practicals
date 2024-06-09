import argparse
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
import matplotlib.pyplot as plt

# Set up argparse
parser = argparse.ArgumentParser(description='Histogram of Oriented Gradients (HOG) Feature Extraction')
parser.add_argument('--image', action='store', default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\texture_snippets\strip_h\AA0004.jpg')
parser.add_argument('--orient', type=int, default=8, help='Number of orientations')
parser.add_argument('--ppc', type=int, default=8, help='Pixels per cell')
parser.add_argument('--cpb', type=int, default=1, help='Cells per block')
flags = parser.parse_args()

# Load and convert image
image = imread(flags.image)
gray_image = rgb2gray(image)

# Compute HOG features
feat, hog_image = hog(gray_image, orientations=flags.orient, pixels_per_cell=(flags.ppc, flags.ppc),
                      cells_per_block=(flags.cpb, flags.cpb), visualize=True, feature_vector=False)

# Plotting the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(gray_image, cmap=plt.cm.gray)
ax1.set_title('Grayscale Image')

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('HOG Visualization')
plt.show()

##  python machine_learning\practic03_1.py --image 'machine_learning/PAML_2024/texture_snippets/strip_h/AA0004.jpg' --orient 8 --ppc 16 --cpb 2
