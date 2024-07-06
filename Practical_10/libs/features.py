import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.feature import local_binary_pattern as lbp

from sklearn.cluster import KMeans

from .extraction import extract_train_eval_files

def extract_hog( file, o, ppc, cpb ):
    img = rgb2gray( imread( file ) )
    feat = hog( img, orientations=o, pixels_per_cell=ppc, cells_per_block=cpb, visualize=False, feature_vector=True )
    return feat.reshape( -1, o )

def extract_train_hog_feature_vector( files, labels, o, ppc, cpb ):
    labs = []
    first = True
    for f, i in zip( files, labels ):
        fe = extract_hog( f, o, ppc, cpb )
        if first:
            feats = fe
            labs += [i]*fe.shape[0]
            first = False
        else:
            feats = np.vstack( (feats, fe) )
            labs += [i]*fe.shape[0]
    return feats, np.array( labs )

class BoVW():
    def __init__(self, num_clusters ):
        self.num_clusters = num_clusters

    def fit(self, X ):
        self.kmeans = KMeans( self.num_clusters )
        self.kmeans.fit( X )

    def predict(self, X ):
        fv = self.kmeans.predict( X )
        h, _ = np.histogram( fv, bins=self.num_clusters )
        return h

def get_bovw_features( files, bovw, o, ppc, cpb ):
    first = True
    for f in files:
        feat = extract_hog( f, o, ppc, cpb )
        h = bovw.predict( feat )
        if first:
            feats = h
            first = False
        else:
            feats = np.vstack( (feats, h) )
    return feats

def extract_lbp_feature_vector( files, radius=1, npoints=8, method='uniform', nbins=10 ):
    first = True
    for f in files:
        gry = rgb2gray( imread( f ) )
        fe = lbp( gry, R=radius, P=npoints, method=method )
        fe, _ = np.histogram( fe, bins=nbins )
        if first:
            feats = fe
            first = False
        else:
            feats = np.vstack( (feats, fe) )
    return feats

def extract_data_mlp( flags ):
    # get the files and labels for training and eval
    tf, tl, ef, el = extract_train_eval_files( flags.dataset, flags.split, flags.classes )
    # get the hog features
    train_hog, _ = extract_train_hog_feature_vector( tf, tl, flags.orient, [flags.ppc, flags.ppc], [flags.cpb,flags.cpb] )
    # train bovw
    bovw = BoVW( flags.nclusters )
    bovw.fit( train_hog )
    # get bovw features
    train_bovw = get_bovw_features( tf, bovw, flags.orient, [flags.ppc,flags.ppc], [flags.cpb, flags.cpb] )
    eval_bovw = get_bovw_features( ef, bovw, flags.orient, [flags.ppc,flags.ppc], [flags.cpb, flags.cpb] )
    # get lbp features
    train_lbp = extract_lbp_feature_vector( tf, radius=flags.radius, npoints=flags.npoints, method=flags.method, nbins=flags.nbins )
    eval_lbp = extract_lbp_feature_vector( ef, radius=flags.radius, npoints=flags.npoints, method=flags.method, nbins=flags.nbins )
    # print( train_bovw.shape, train_lbp.shape )
    # print( eval_bovw.shape, eval_lbp.shape )
    Xt = np.hstack( (train_bovw, train_lbp) )
    Xe = np.hstack( (eval_bovw, eval_lbp) )
    # normalizer
    mu = Xt.mean( axis=0 )
    st = Xt.std( axis=0 )
    Xt = (Xt-mu)/st
    Xe = (Xe-mu)/st
    return Xt, tl, Xe, el