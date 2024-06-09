import numpy as np
class KMeans():
    def __init__(self, n_clusters, imax=100 ):
        self.n_clusters = n_clusters
        self.imax = imax

    def euclid(self, X, c ): ## Calculate the Euclidean distance between each data point and each center
        diff = X-c
        sqrd = diff**2
        smmd = np.sum( sqrd, axis=1 )
        return np.sqrt( smmd )

    def fit(self, X ):
        # # Randomly initialize centers from the data points
        cstart = np.random.randint( 0, X.shape[0], self.n_clusters )
        self.C = X[cstart,:]
        # iterate over imax
        for _ in range( self.imax ):
            dist = np.zeros( (X.shape[0], self.n_clusters) )
            for i in range( self.n_clusters ):
                dist[:,i] = self.euclid(X, self.C[i] )

            # assign to center # Assign each data point to the closest center
            X_assign = np.argmin( dist, axis=1 )

            # recalculate centers -  # Update centers to the mean of assigned points
            for i in range( self.n_clusters ):
                self.C[i,:] = np.mean( X[X_assign==i,:], axis=0 )

    def predict(self, X ):
        dist = np.zeros( (X.shape[0], self.n_clusters) )
        # compute the distances
        for i in range( self.n_clusters ):
            dist[:,i] = self.euclid(X, self.C[i])
        return np.argmin( dist, axis=1 )