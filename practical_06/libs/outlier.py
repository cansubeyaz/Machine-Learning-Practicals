import numpy as np

def Z_score( X, t=2.5 ):
    mu = X.mean( axis=0 )
    st = X.std( axis=0 )
    z = (X-mu)/st
    b0 = np.logical_and( z[:,0]<t, z[:,0]>-t )
    b1 = np.logical_and( z[:,1]<t, z[:,1]>-t )
    b = np.logical_and( b0, b1 )
    return X[b,:]

def Z_score_mod( X, t=3.5 ):
    md = np.median( X, axis=0 )
    mad = np.median( np.abs( X-md ), axis=0 )
    m = 0.6745*(X-md)/mad
    b0 = np.logical_and(m[:, 0] < t, m[:, 0] > -t)
    b1 = np.logical_and(m[:, 1] < t, m[:, 1] > -t)
    b = np.logical_and(b0, b1)
    return X[b, :]