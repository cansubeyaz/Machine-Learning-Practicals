"""
Library with Guassian based stuff
"""

import numpy as np

from sklearn.mixture import GaussianMixture
class MVG():
    # train step
    def fit(self, X ):
        m, n = X.shape
        # get the mean from X
        mu = np.mean( X, axis=0 )
        # residuals
        X_ = X-mu
        # covariance
        sigma = (X_.T@X_)/float( m )
        # members
        self.mu = mu
        self.sigma = sigma
        # calculate the constants
        self.precompute()

    def precompute(self ):
        n = len( self.mu )
        # calculate the inverse sigma
        self.inv_sigma = np.linalg.inv( self.sigma )
        log_two_pi = -n/2.*np.log( 2*np.pi )
        log_det = -0.5*np.linalg.slogdet( self.sigma )[1]
        self.constant = log_two_pi+log_det

    def log_likelihood(self, X ):
        m,n = X.shape
        # set up storage vector
        l_like = np.zeros( (m,) )
        # calculate resids
        resids = X-self.mu
        # iterate over resids and calculate the log like
        for i in range( m ):
            l_like[i] = -0.5*resids[i,:]@self.inv_sigma@resids[i,:].T+self.constant
        return l_like

class MultiGMM():
    def __init__(self, n_mixtures ):
        self.n_mixtures = n_mixtures
        self.gmms = {}

    def fit(self, d ):
        for k, v in d.items():
            self.gmms[k] = GaussianMixture( self.n_mixtures ).fit( v )

    def predict(self, X ):
        scores = np.zeros( (X.shape[0], len( self.gmms )) )
        for i, (k,g) in enumerate( self.gmms.items() ):
            scores[:,i] = g.score_samples( X )
        cls = np.argmax( scores, axis=1 )
        return cls, scores
