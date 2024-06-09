"""
MVG
GMMS with sklearn
"""

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from skimage.io import imread

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from libs.Gaussian import MVG, MultiGMM

np.set_printoptions( precision=3 )

# parse the arguments
parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--exr', action='store', required=True )
parser.add_argument( '--data_name', action='store', default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\practical_05\mvg_data.pkl' )
parser.add_argument( '--split', action='store', type=float, default=0.3 )
parser.add_argument( '--dataset', action='store', default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\colour_snippets' )
parser.add_argument( '--n_mixtures', action='store', type=int, default=3 )
flags = parser.parse_args()

# preamble
exr0 = True if flags.exr == 'exr0' else False
exr1 = True if flags.exr == 'exr1' else False

if exr0:
    # load data
    with open( flags.data_name, 'rb' ) as fid:
        data = pickle.load( fid )
    print( data.keys() )
    lab = data['LX'].reshape( -1, )
    print( lab.shape )
    X0 = data['X'][lab==0]
    X1 = data['X'][lab==1]
    print( X0.shape, X1.shape )
    # create the MVG objects and fit them
    obj_0, obj_1 = MVG(), MVG()
    obj_0.fit( X0 )
    obj_1.fit( X1 )
    print( obj_0.mu )
    print( obj_0.sigma )
    print( obj_1.mu )
    print( obj_1.sigma )
    # evaluate our MVGs
    lab = data['LY_0_1']
    print( lab.shape )
    Y = data['Y_0_1']
    log_like = np.zeros( (lab.shape[0], 2) )
    log_like[:,0] = obj_0.log_likelihood( Y )
    log_like[:,1] = obj_1.log_likelihood( Y )
    # classify our results
    preds = np.argmax( log_like, axis=1 )
    # calculate the accuracy score and confusion matrix
    acc = accuracy_score( lab, preds )
    print( f'The accuracy is {acc}' )
    cm = confusion_matrix( lab, preds, normalize='true' )
    print( cm )
    acc_cm = cm.diagonal().sum()/cm.shape[0]
    print( f'cm diagonal accuracy {acc_cm}' )


if exr1:
    train = {}
    test_first = True
    for i, c in enumerate( sorted( os.listdir( flags.dataset ) ) ):
        print( c )
        first = True
        for f in sorted( glob.glob( os.path.join( flags.dataset, c, '*.png' ) ) ):
            # print( f )
            V = imread( f ).reshape( (-1,3) )
            if first:
                X = V
                first = False
            else:
                X = np.vstack( (X, V) )
        # split them up
        train[c], e = train_test_split( X, test_size=flags.split )
        if test_first:
            Y = e
            labels = np.zeros( (e.shape[0],) )
            test_first = False
        else:
            Y = np.vstack( (Y, e) )
            labels = np.hstack( (labels, np.ones( (e.shape[0],) )*i) )
    # create gmm
    gmm = MultiGMM( flags.n_mixtures )
    gmm.fit( train )
    preds, _ = gmm.predict( Y )
    # metrics
    acc = accuracy_score( labels, preds )
    print( f'The accuracy score {acc}' )
    cm = confusion_matrix( labels, preds, normalize='true' )
    print( cm )
    cm_acc = cm.diagonal().sum()/float( cm.shape[0] )
    print( cm_acc )