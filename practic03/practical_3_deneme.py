import argparse
import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.filters import sobel

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import cv2

from libs.sobel_edge import my_sobel

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--exr", action="store", required=True)
parser.add_argument("--image", action="store", default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\texture_snippets\strip_h\AA0004.jpg')
## Histogram of Oriented Gradients (HOG) parameters
parser.add_argument("--orientations", action="store", type=int, default=8) ## The number of orientation bins.
parser.add_argument("--pixels_per_cell", action="store", type=int, default=8) ## The size (in pixels) of a cell.
parser.add_argument("--cells_per_block", action="store", type=int, default=1) ## The number of cells in each block
## LOCAL BINARY PATTERNS - LBP Parameters
parser.add_argument("--radius", action = "store", type=int, default=1)
parser.add_argument("--points", action="store", type=int, default=8)
parser.add_argument("--method", action="store", default="default")

flags=parser.parse_args()

exr0 = True if flags.exr == "exr0" else False ## Histogram of Oriented Gradients
exr1 = True if flags.exr == "exr1" else False ## Local Binary Pattern
exr2 = True if flags.exr == "exr2" else False ## Sobel Edge Detection
exr3 = True if flags.exr == "exr3" else False ## Principal Component Analysis

if exr0:
    gray_image = cv2.imread(flags.image, cv2.IMREAD_GRAYSCALE) ##rgb2gray(imread(flags.image)
    orientation = flags.orientations
    ppc = [flags.pixels_per_cell, flags.pixels_per_cell] ## [8,8]
    cpb = [flags.cells_per_block, flags.cells_per_block] ## [1,1]
    ## Histogram of Oriented Gradients (HOG)
    feature, hog_map = hog(gray_image, orientations = orientation, pixels_per_cell = ppc, cells_per_block = cpb, visualize= True, feature_vector = False)
    ## feature vector - false : feature_vector=False: The HOG features are returned as a 3D array, feature_vector = True : 1D array

    ## Plot
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    ax[0].imshow(gray_image)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(hog_map, cmap = "gray")
    ax[1].set_title("Histogram of Orient Gradient (HOG)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

## TERMINAL
##  python practical_03\practical_3_deneme.py --exr exr0 --image C:\Users\Lenovo\Desktop\PAML_2024\data\texture_snippets\strip_h\AA0004.jpg --orientation 8 --pixels_per_cell 16 --cells_per_block 1

if exr1:
    gray_image = cv2.imread(flags.image, cv2.IMREAD_GRAYSCALE)
    radi = flags.radius
    point = flags.points
    ## LBP feature
    lbp_feature = local_binary_pattern(gray_image, P = point, R = radi, method=flags.method)
    print(lbp_feature.shape)

    ## Plot
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    ax[0].imshow(gray_image, cmap ="gray")
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(lbp_feature, cmap = "gray")
    ax[1].set_title("Local Binary Pattern (LBP)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

## TERMINAL
## python practical_03\practical_3_deneme.py --exr exr1 --image C:\Users\Lenovo\Desktop\PAML_2024\data\texture_snippets\strip_h\AA0004.jpg --rad 1 --points 8

if exr2:
    gray_image = cv2.imread(flags.image, cv2.IMREAD_GRAYSCALE)
    horizontal, vertical, magnitude = my_sobel(gray_image)

    sobel_opr = sobel(gray_image) ## skimage sobel

    ## Plot
    fig,ax = plt.subplots(2,3, sharex=True, sharey=True)
    ax[0,0].imshow(gray_image, cmap ="gray")
    ax[0,0].set_title("Input Image")
    ax[0,1].imshow(sobel_opr, cmap ="gray")
    ax[0,1].set_title("Skimage Sobel")
    ax[0,2].imshow(magnitude, cmap="gray")
    ax[0,2].set_title("My Sobel Edge Detector")
    ax[1,0].imshow(horizontal, cmap="gray")
    ax[1,0].set_title("Horizontal")
    ax[1,1].imshow(vertical, cmap="gray")
    ax[1,1].set_title("Vertical")

    # Remove the empty subplot
    fig.delaxes(ax[1, 2])

    plt.tight_layout()
    plt.show()

    print("Shape of h:", horizontal.shape) ## (140,140) - height width
    print("Shape of v:", vertical.shape) ## (140,140) - height width
    print("Shape of magnitude:", magnitude.shape) ## (140,140)

## TERMINAL
## python practical_03\practical_3_deneme.py --exr exr2 --image C:\Users\Lenovo\Desktop\PAML_2024\data\texture_snippets\strip_h\AA0004.jpg

if exr3:
    x, y = load_wine(return_X_y= True) ## return the features (x) and labels (y) separately.
    x = np.array(x)
    y = np.array(y)
    print("samples y :", len(y)) ## 178 samples
    print(" y :", y) ## 178 target samples, [0 1 2] label categories
    print("x shape :", x.shape) ## (178,13) - x : input, 178: samples, 13: features (dimensions)
    print("y shape :", y.shape) ## (178,) - y : output (features - labels), 178: samples

    ## PCA
    pca = PCA(n_components=2).fit_transform(x)
    print("PCA shape :", pca.shape) ## (178,2) -  PCA 178: samples, 2: dimensions (features)

    ## Plotting the PCA Results
    plt.figure()
    plt.scatter(pca[y==0,0], pca[y==0,1], color='red') ## Class 0 (Red Points)
    plt.scatter(pca[y==1,0], pca[y==1,1], color='green') ## Class 0 (Red Points)
    plt.scatter(pca[y==2,0], pca[y==2,1], color='blue') ## Class 0 (Red Points)

    plt.tight_layout()
    plt.show()