"""
  Bag of Visual Words with HOG
  K-Nearest Neighbour Classification

"""

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys



from skimage.io import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve as prc

from libs.extraction import extract_train_eval_files
from libs.features import extract_train_hog_feature_vector, BoVW, extract_hog
from libs.utils import meanstd
from libs.nearest import KNN

parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--exr', action='store', required=True )
## parser.add_argument( '--dataset', action='store', required=True )
parser.add_argument('--dataset', action='store',
                    default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\texture_snippets')
parser.add_argument( '--classes', '-c', nargs="+", default=[] )
parser.add_argument( '--split', action='store', type=float, default=0.3 )
parser.add_argument( '--orient', action='store', type=int, default=8 )
parser.add_argument( '--ppc', action='store', type=int, default=8 )
parser.add_argument( '--cpb', action='store', type=int, default=1 )
parser.add_argument( '--nclusters', action='store', type=int, default=3 )
parser.add_argument( '--knn', action='store', type=int, default=1 )
parser.add_argument( '--verbosity', action='store_true' )
flags = parser.parse_args()

# preamble
exr0 = True if flags.exr == 'exr0' else False
exr1 = True if flags.exr == 'exr1' else False

if 'all' in flags.classes:
  flags.classes = ['irregular', 'plaid', 'plaid', 'plaid_d', 'plain', 'spots', 'strip_d', 'strip_h', 'strip_v']
print( flags.classes )

def display_things( files, labels, o, ppc, cpb,  bovw, ncls ):
    idx = labels.index( 1 )
    f0 = files[:10]
    f1 = files[idx:idx+10]
    fig, ax = plt.subplots( 4, 5, sharex=True, sharey=True )
    r,c = 0,0
    for f0_,f1_ in zip( f0,f1 ):
        f0_f = extract_hog( f0_, o, ppc, cpb )
        f1_f = extract_hog( f1_, o, ppc, cpb )
        h0 = bovw.predict( f0_f )
        h1 = bovw.predict( f1_f )
        ax[r,c].hist( h0, bins=ncls )
        ax[r+2,c].hist( h1, bins=ncls )
        c+=1
        if c == 5:
            c = 0
            r+=1
    plt.tight_layout()
    plt.show()

if exr0:
    tf, tl, ef, el = extract_train_eval_files( flags.dataset, flags.split, flags.classes )
    train_hog, train_hog_labels = extract_train_hog_feature_vector( tf, tl, flags.orient, [flags.ppc, flags.ppc], [flags.cpb, flags.cpb] )
    print( train_hog.shape )
    print( train_hog_labels.shape )
    # bovw
    bovw = BoVW( flags.nclusters )
    bovw.fit( train_hog )
    # display
    display_things( ef, el, flags.orient, [flags.ppc,flags.ppc], [flags.cpb,flags.cpb], bovw, flags.nclusters )


if exr1:
    first = True
    cls = 0
    for i, c in enumerate( sorted( os.listdir( flags.dataset ) ) ):
        if c in flags.classes:
            print( c )
            for f in sorted( glob.glob( os.path.join( flags.dataset, c, '*.png' ) ) ):
                rgb = imread( f )/255.
                rgb = meanstd( rgb )
                rgb = rgb.reshape( -1, 3 )
                if first:
                    V = rgb
                    l = np.ones( (rgb.shape[0],) )*cls
                    first = False
                else:
                    V = np.vstack( (V,rgb) )
                    l = np.concatenate( (l, np.ones( (rgb.shape[0], ) )*cls) )
            cls+=1
    Xt, Xe, yt, ye = train_test_split( V, l, test_size=flags.split )
    obj = KNN( K=flags.knn )
    obj.fit( Xt, yt )
    preds = obj.predict( Xe )
    cm = confusion_matrix( ye, preds, normalize='true' )
    print( cm )