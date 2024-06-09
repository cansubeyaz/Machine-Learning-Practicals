import numpy as np

def my_sobel(image):
    ## Kernels
    kh = np.array([[-1,0,1],[-2,0,2], [-1,0,1]])
    kv = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    horizontal = np.zeros(image.shape) ## Create a 2D array of zeros with shape (140, 140)
    vertical = np.zeros(image.shape) ## Create a 2D array of zeros with shape (140, 140)

    for r in range( 1, image.shape[0]-1 ): ## terates over the rows of the image, starting from the second row (1) and ending at the second last row
        for c in range( 1, image.shape[1]-1 ): ## iterates over the columns of the image, starting from the second column (1) and ending at the second last column
            ## Extract 3x3 Neighborhood (snip)
            snip = image[r-1:r+2,c-1:c+2]
            ## Apply Sobel Kernels
            horizontal[r,c] = np.sum(kh*snip)
            vertical[r,c] = np.sum(kv*snip)
    return horizontal, vertical, np.sqrt(horizontal**2+vertical**2) ## gradient magnitude