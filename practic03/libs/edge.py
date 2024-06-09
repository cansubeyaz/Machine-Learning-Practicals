import numpy as np

def my_sobel( image ):
    # kernels
    kh = np.array( [[-1,0,1],[-2,0,2],[-1,0,1]] )
    kv = np.array( [[-1,-2,-1],[0,0,0],[1,2,1]] )
    # storage
    h = np.zeros( image.shape )
    v = np.zeros( image.shape )
    # calculate
    for r in range( 1, image.shape[0]-1 ):
        for c in range( 1, image.shape[1]-1 ):
            snip = image[r-1:r+2,c-1:c+2]
            h[r,c] = np.sum( kh*snip )
            v[r,c] = np.sum( kv*snip )
    return h, v, np.sqrt( h**2+v**2 )