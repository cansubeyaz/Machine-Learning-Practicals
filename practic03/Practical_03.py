
import argparse
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.feature import local_binary_pattern as lbp
from skimage.filters import sobel

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

from libs.edge import my_sobel

parser = argparse.ArgumentParser( add_help=True )

parser.add_argument( '--exr', action='store', required=True )
parser.add_argument( '--image', action='store', default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\texture_snippets\strip_h\AA0004.jpg' )
parser.add_argument( '--orient', action='store', type=int, default=8 )
parser.add_argument( '--ppc', action='store', type=int, default=8 )
parser.add_argument( '--cpb', action='store', type=int, default=1 )
parser.add_argument( '--rad', action='store', type=int, default=1 )
parser.add_argument( '--points', action='store', type=int, default=8 )
parser.add_argument( '--method', action='store', default='default' )
flags = parser.parse_args()

exr0 = True if flags.exr == 'exr0' else False
exr1 = True if flags.exr == 'exr1' else False
exr2 = True if flags.exr == 'exr2' else False
exr3 = True if flags.exr == 'exr3' else False

if exr0:
    gry = rgb2gray( imread( flags.image ) )
    o = flags.orient
    ppc = [flags.ppc, flags.ppc]
    cpb = [flags.cpb, flags.cpb]
    # create hog
    feat, map = hog( gry, orientations=o, pixels_per_cell=ppc, cells_per_block=cpb, visualize=True, feature_vector=False )
    # plot
    fig, ax = plt.subplots( 1, 2 )
    ax[0].imshow( gry, cmap='gray' )
    ax[0].set_title( 'Gray scale image' )
    ax[1].imshow( map )
    ax[1].set_title( 'HoG map' )
    plt.show()

if exr1:
    gry = rgb2gray( imread( flags.image ) )
    r = flags.rad
    p = flags.points
    # create lbp feature
    feats = lbp( gry, R=r, P=p, method=flags.method )
    print( feats.shape )
    fig, ax = plt.subplots( 1, 2 )
    ax[0].imshow( gry, cmap='gray' )
    ax[1].imshow( feats )
    plt.show()

if exr2:
    # read in gray scale
    img = rgb2gray( imread( flags.image ) )
    # calculate my sobel operator
    h, v, m = my_sobel( img )
    # calculate skimage sobel operator
    sks = sobel( img )
    # plot them together
    fig, ax = plt.subplots( 2, 3 )
    ax[0,0].imshow( img )
    ax[0,1].imshow( sks )
    ax[0,1].set_title( 'skimage sobel' )
    ax[0,2].imshow( m )
    ax[0,2].set_title( 'my sobel' )
    ax[1,0].imshow( h )
    ax[1,1].imshow( v )
    plt.show()


if exr3:
    x, y = load_wine( return_X_y=True )
    x = np.array( x )
    y = np.array( y )
    print( y )
    print( x.shape )
    print( y.shape )
    pca = PCA( n_components=2 ).fit_transform( x )
    print( pca.shape )
    plt.figure()
    plt.scatter( pca[y==0,0], pca[y==0,1], color='red' )
    plt.scatter( pca[y==1,0], pca[y==1,1], color='green' )
    plt.scatter( pca[y==2,0], pca[y==2,1], color='blue' )
    plt.tight_layout()
    plt.show()