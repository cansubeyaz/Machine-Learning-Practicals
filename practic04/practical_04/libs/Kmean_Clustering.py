import numpy as np

class KMeans():
    def __init__(self, n_clusters, imax=100): ## imax = max iterations
        self.n_clusters = n_clusters
        self.imax = imax

    def euclid(self, X, c): ## c : centers
        sqrd = (X-c)**2
        sum = np.sum(sqrd, axis=1)
        return np.sqrt(sum)

    def fit(self, X ):
        # randomly start points
        center_start = np.random.randint( 0, X.shape[0], self.n_clusters )
        self.C = X[center_start,:]
        # iterate over imax
        for _ in range( self.imax ):
            dist = np.zeros( (X.shape[0], self.n_clusters) )
            for i in range( self.n_clusters ):
                dist[:,i] = self.euclid(X, self.C[i] )
            # assign to center
            X_assign = np.argmin( dist, axis=1 )
            # recalculate centers
            for i in range( self.n_clusters ):
                self.C[i,:] = np.mean( X[X_assign==i,:], axis=0 )

    def predict(self, X ):
        dist = np.zeros( (X.shape[0], self.n_clusters) )
        # compute the distances
        for i in range( self.n_clusters ):
            dist[:,i] = self.euclid(X, self.C[i])
        return np.argmin( dist, axis=1 )
