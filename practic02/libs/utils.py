import glob
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.color import rgb2lab

def read_to_vector( path ):
    filenames = glob.glob( path )
    first = True
    for f in filenames:
        img = imread( f )/255.
        cnv = rgb2lab( img )
        if first:
            rgb = img[:,:,0].reshape( 1, -1 )
            lab = cnv[:,:,1].reshape( 1, -1 )
            first = False
        else:
            rgb = np.hstack( (rgb, img[:,:,0].reshape( 1, -1 )) )
            lab = np.hstack( (lab, cnv[:,:,1].reshape( 1, -1 )) )
    return rgb.reshape( -1, ), lab.reshape( -1, )

def plot_stuff( bins, red, grn ):
    fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
    ax[0].hist(red, bins=bins)
    ax[1].hist(grn, bins=bins)
    plt.show()