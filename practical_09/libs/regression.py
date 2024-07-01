import numpy as np

class LogisticRegression():
    def __init__( self, lr=0.001, imax=100 ):
      self.lr = lr
      self.imax = imax
      self.w = None
      self.b = None

    def sigmoid( self, x ):
      return 1/(1+np.exp( -x ))

    def fit( self, X, y ):
      ns, nf = X.shape
      self.w = np.zeros( nf )
      self.b = 0

      for i in range( self.imax ):
        preds = X@self.w + self.b   # X.w+b
        preds = self.sigmoid( preds ) # sig( x )

        dw = (1/ns) * X.T@(preds-y) # \frac{1}{ns} \sum_i{ylabeli-ypredi}
        db = (1/ns) * np.sum( preds-y  ) # \frac{1}{ns} \sum_i{ylabeli-ypredi}

        self.w = self.w-self.lr*dw
        self.b = self.b-self.lr*db

    def predict( self, X ):
      preds = X@self.w + self.b
      probs = self.sigmoid( preds )

      cls = [0 if p<0.5 else 1 for p in probs]
      return probs, cls
