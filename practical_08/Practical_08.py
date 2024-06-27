"""
  Today we will look at
   * HOG-BoVW-SVM
   * SVM LBP for precision recall curves
"""

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


from sklearn.svm import SVC

from libs.extraction import extract_train_eval_files
from libs.features import extract_train_hog_feature_vector, BoVW, get_bovw_features, extract_lbp_feature_vector
from libs.metrics import svm_metrics

np.set_printoptions( precision=3 )

parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--exr', action='store', required=True )
parser.add_argument( '--dataset', action='store', required=True )
parser.add_argument( '--classes', '-c', nargs="+", default=[] )
parser.add_argument( '--split', action='store', type=float, default=0.3 )
parser.add_argument( '--orient', action='store', type=int, default=8 )
parser.add_argument( '--ppc', action='store', type=int, default=8 )
parser.add_argument( '--cpb', action='store', type=int, default=1 )
parser.add_argument( '--nclusters', action='store', type=int, default=16 )
parser.add_argument( '--C', action='store', type=float, default=1.0 )
parser.add_argument( '--radius', action='store', type=int, default=1 )
parser.add_argument( '--npoints', action='store', type=int, default=8 )
parser.add_argument( '--nbins', action='store', type=int, default=10 )
parser.add_argument( '--method', action='store', default='uniform' )
parser.add_argument( '--verbosity', action='store_true' )
flags = parser.parse_args()

# preamble
exr0 = True if flags.exr == 'exr0' else False
exr1 = True if flags.exr == 'exr1' else False

if 'all' in flags.classes:
  flags.classes = ['irregular', 'plaid', 'plaid_d', 'plain', 'spots', 'strip_d', 'strip_h', 'strip_v']
print( flags.classes )

if exr0:
  # get the training and eval image and labels
  tfiles, tlabs, efiles, elabs = extract_train_eval_files( flags.dataset, flags.split, flags.classes )
  # get the training hog feature vector
  train_hog, _ = extract_train_hog_feature_vector( tfiles, tlabs, flags.orient, [flags.ppc,flags.ppc], [flags.cpb, flags.cpb] )
  # bovw stuff
  bovw = BoVW( flags.nclusters )
  bovw.fit( train_hog )
  # get the bovw features from training and eval
  bovw_feats_train = get_bovw_features( tfiles, bovw, flags.orient, [flags.ppc,flags.ppc], [flags.cpb, flags.cpb] )
  bovw_feats_eval = get_bovw_features(efiles, bovw, flags.orient, [flags.ppc, flags.ppc], [flags.cpb, flags.cpb])
  print( 'training', bovw_feats_train.shape, len( tlabs) )
  print( 'evaluation', bovw_feats_eval.shape, len( elabs ) )
  # normalize
  mu = bovw_feats_train.mean( axis=0 )
  st = bovw_feats_train.std( axis=0 )
  bovw_feats_train = (bovw_feats_train-mu)/st
  bovw_feats_eval = (bovw_feats_eval - mu) / st
  # train linear svm
  linsvm = SVC( kernel='linear', C=flags.C )
  linsvm.fit( bovw_feats_train, tlabs )
  # train rbf svm
  rbfsvm = SVC( kernel='rbf', C=flags.C, gamma='scale')
  rbfsvm.fit( bovw_feats_train, tlabs )

  # report the metrics
  svm_metrics( bovw_feats_eval, linsvm, elabs, 'linear svm' )
  svm_metrics( bovw_feats_eval, rbfsvm, elabs, 'RBF svm' )

if exr1:
  # get the training and eval image and labels
  tfiles, tlabs, efiles, elabs = extract_train_eval_files(flags.dataset, flags.split, flags.classes)
  # get the lbp features
  train_lbp = extract_lbp_feature_vector( tfiles, radius=flags.radius, npoints=flags.npoints, method=flags.method, nbins=flags.nbins )
  eval_lbp = extract_lbp_feature_vector(efiles, radius=flags.radius, npoints=flags.npoints, method=flags.method,
                                         nbins=flags.nbins)
  # normalise features
  mu = train_lbp.mean( axis=0 )
  st = train_lbp.std( axis=0 )
  train_lbp = (train_lbp-mu)/st
  eval_lbp = (eval_lbp-mu)/st

  # train linear svm
  linsvm = SVC(kernel='linear', C=flags.C)
  linsvm.fit(train_lbp, tlabs)
  # train rbf svm
  rbfsvm = SVC(kernel='rbf', C=flags.C, gamma='scale')
  rbfsvm.fit(train_lbp, tlabs)

  # report the metrics
  svm_metrics(eval_lbp, linsvm, elabs, 'linear svm')
  svm_metrics(eval_lbp, rbfsvm, elabs, 'RBF svm')

