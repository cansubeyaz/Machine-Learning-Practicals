"""
Practical 4
  * MVG data creation
  * KMeans clustering
  * spectral clustering
  * image normalisation

"""

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

from skimage.io import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import completeness_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import SpectralClustering

from libs.clustering import KMeans

# get the parsing
parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--exr', action='store', default='none' )
parser.add_argument( '--create_data', action='store_true' )
parser.add_argument( '--data_name', action='store', default='mvg_data.pkl' )
parser.add_argument( '--train_samp', action='store', type=int, default=100 )
parser.add_argument( '--test_samp', action='store', type=int, default=50 )
parser.add_argument( '--nclusters', action='store', type=int, default=2 )
flags = parser.parse_args()

# preamble
exr0 = True if flags.exr == 'exr0' else False ## Kmeans Clustering
exr1 = True if flags.exr == 'exr1' else False ## Spectral Clustering
exr2 = True if flags.exr == 'exr2' else False ## Image Normalisation

print( flags.create_data )

if flags.create_data:
    # means and covariance
    mu0 = np.array( [0,0] )
    sigma0 = np.array( [[1.,0],[0,1]] )
    mu1 = np.array( [4,1] )
    sigma1 = np.array( [[1.,0],[0,1]] )
    mu2 = np.array( [2,2] )
    sigma2 = np.array( [[0.5,0],[0,0.5]] )
    # create mvg data
    X0_train = np.random.multivariate_normal( mu0, sigma0, flags.train_samp )
    X0_test = np.random.multivariate_normal(mu0, sigma0, flags.test_samp)
    X1_train = np.random.multivariate_normal(mu1, sigma1, flags.train_samp)
    X1_test = np.random.multivariate_normal(mu1, sigma1, flags.test_samp)
    X2_test = np.random.multivariate_normal(mu2, sigma2, flags.test_samp)
    # plot the data
    plt.figure()
    plt.scatter( X0_train[:,0], X0_train[:,1], color='red', label='X0_train' )
    plt.scatter(X0_test[:, 0], X0_test[:, 1], color='blue', label='X0_test')
    plt.scatter(X1_train[:, 0], X1_train[:, 1], color='green', label='X1_train')
    plt.scatter(X1_test[:, 0], X1_test[:, 1], color='yellow', label='X1_test')
    plt.scatter(X2_test[:, 0], X2_test[:, 1], color='magenta', label='X2_test')
    plt.legend()
    plt.show()

    # dictionary of data
    data = {}
    data['X'] = np.vstack( (X0_train, X1_train) )
    data['LX'] = np.vstack( (np.zeros( (flags.train_samp, 1)), np.ones( (flags.train_samp, 1) )) )
    data['Y_0_1'] = np.vstack((X0_test, X1_test))
    data['Y_0_1_2'] = np.vstack((X0_test, X1_test, X2_test))
    data['LY_0_1'] = np.vstack((np.zeros((flags.test_samp, 1)), np.ones((flags.test_samp, 1))))
    data['LY_0_1_2'] = np.vstack((np.zeros((flags.test_samp, 1)), np.ones((flags.test_samp, 1)), np.ones((flags.test_samp, 1)) * 2))

    with open( flags.data_name, 'wb' ) as fid: ## Save data as pickle
        pickle.dump( data, fid )

## TERMINAL
## python practical_04\Practical_04.py --create_data

if exr0:
    with open( flags.data_name, 'rb') as fid:
        data = pickle.load(fid)
    print(data.keys())

    obj = KMeans(flags.nclusters, imax=10)
    obj.fit(data['X'] )
    preds = obj.predict(data['Y_0_1']) # Predict data['Y_0_1']

    # Scatter plot the predictions for Y_0_1
    plt.figure()
    for i in range(flags.nclusters):
        plt.scatter(data['Y_0_1'][preds == i, 0], data['Y_0_1'][preds == i, 1], label=f'Cluster {i}')
    plt.legend()
    plt.title('Clustering of Y_0_1')
    plt.show()

    # Calculate completeness_score and accuracy_score for Y_0_1
    print('Completeness Score for Y_0_1:', completeness_score(data['LY_0_1'].reshape(-1), preds)) ## for understanding the quality of the clustering.
    print('Accuracy Score for Y_0_1:', accuracy_score(data['LY_0_1'].reshape(-1), preds))

    preds = obj.predict(data['Y_0_1_2']) # Predict data['Y_0_1_2']
    plt.figure()

    # Scatter plot the predictions for Y_0_1_2
    for i in range(flags.nclusters):
        plt.scatter(data['Y_0_1_2'][preds==i,0], data['Y_0_1_2'][preds==i,1], label=f'centroid{i}' )
    plt.legend()
    plt.title('Clustering of Y_0_1_2')
    plt.show()

    print("shape X ", data["X"].shape) ## (200,2)
    print("shape LY : ", data['LY_0_1_2'].shape) ## (150,1)

    ## Use accuracy score for classification tasks, Use completeness score for clustering tasks
    print('Completeness Score for Y_0_1_2:', completeness_score(preds, data['LY_0_1_2'].reshape(-1,)))
    print('Accuracy Score for Y_0_1_2:', accuracy_score(preds, data['LY_0_1_2'].reshape(-1,)))

    print("shape LY : ", data['LY_0_1_2'].shape) ## (150,1)


if exr1:
    with open(flags.data_name, 'rb') as fid: ## read binary (rb) load the MVG data that we created earlier .
        data = pickle.load(fid)
    print(data.keys())

    # create the object
    spcl = SpectralClustering(n_clusters=flags.nclusters, assign_labels="discretize", )
    # fit and predict the data
    fpred = spcl.fit_predict(data['X'])
    # now based on this data we can find the centroid of each cluster
    centroids = np.zeros((flags.nclusters, data['X'].shape[1]))
    for i in range(flags.nclusters):
        centroids[i, :] = np.mean(data['X'][fpred == i], axis=0)
    # now predict the data
    dist = np.zeros((data['Y_0_1'].shape[0], flags.nclusters))
    for i in range(flags.nclusters):
        dist[:, i] = np.sqrt(np.sum((data['Y_0_1'] - centroids[i, :]) ** 2, axis=1))
    # get the assigned cluster
    pred = np.argmin(dist, axis=1)
    # now check the completeness and accuracy scores
    print("Completeness Score for Y_0_1: ", completeness_score(pred, data['LY_0_1'].reshape(-1, )))
    print("Accuracy Score for Y_0_1: ", accuracy_score(pred, data["LY_0_1"].reshape(-1, )))

    # Scatter plot the predictions for Y_0_1
    plt.figure()
    for i in range(flags.nclusters):
        plt.scatter(data['Y_0_1'][pred == i, 0], data['Y_0_1'][pred == i, 1], label=f'Cluster {i}')
    plt.legend()
    plt.title('Spectral Clustering of Y_0_1')
    plt.show()

    # Predict data['Y_0_1_2'] and calculate metrics
    distances_Y_0_1_2 = np.linalg.norm(data['Y_0_1_2'][:, np.newaxis] - centroids, axis=2)
    preds_Y_0_1_2 = np.argmin(distances_Y_0_1_2, axis=1)

    completeness_Y_0_1_2 = completeness_score(data['LY_0_1_2'].reshape(-1), preds_Y_0_1_2)
    accuracy_Y_0_1_2 = accuracy_score(data['LY_0_1_2'].reshape(-1), preds_Y_0_1_2)

    print(f'Completeness Score for Y_0_1_2: {completeness_Y_0_1_2}')
    print(f'Accuracy Score for Y_0_1_2: {accuracy_Y_0_1_2}')

    # Scatter plot the predictions for Y_0_1_2
    plt.figure()
    for i in range(flags.nclusters):
        plt.scatter(data['Y_0_1_2'][preds_Y_0_1_2 == i, 0], data['Y_0_1_2'][preds_Y_0_1_2 == i, 1],
                    label=f'Cluster {i}', marker="o")

    # Plotting the cluster centers with diamond markers
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='D', color='red', s=100, label='Cluster Centers')

    plt.legend()
    plt.title('Spectral Clustering of Y_0_1_2')
    plt.show()

## TERMINAL
##  python .\Practical_04.py --exr exr1

if exr2:
    files = glob.glob(r'C:\Users\Lenovo\Desktop\PAML_2024\data\colour_snippets\red\*.png') ## Load all the color snippet red images.
    train, test = train_test_split( files, test_size=0.1 ) ## Split the images into a training set (90%) and an evaluation set (10%).
    first = True
    for f in train:
        rgb = imread(f)/255. ## Divide the pixel values by 255 to rescale them to the range [0,1]
        if first:
            r = rgb[:,:,0].reshape(-1, 1)
            g = rgb[:,:,1].reshape(-1, 1)
            b = rgb[:,:,2].reshape(-1, 1)
            first = False
        else:
            r = np.vstack((r, rgb[:,:,0].reshape( -1, 1 )))
            g = np.vstack((g, rgb[:, :, 1].reshape(-1, 1)))
            b = np.vstack((b, rgb[:, :, 2].reshape(-1, 1)))
    print( r.max(), g.max(), b.max() ) ## shape : (8740, 1) (8740, 1) (8740, 1)

    # calculate statistics - For the training set, calculate the minimum, maximum, mean, and standard deviation for each channel (R, G, B).
    mn = np.array([r.min(), g.min(), b.min()]) ## minimum for each channel - train
    mx = np.array([r.max(), g.max(), b.max()]) ## maximum for each channel - train
    mu = np.array([r.mean(), g.mean(), b.mean()]) ## mean for each channel - train
    st = np.array([r.std(), g.std(), b.std()]) ## standard dev. for each channel - train

    # Normalise your eval set - load test image
    image = imread(test[0])/255.

    # Min-Max normalization
    def minmax(im, mn, mx):
        return (im-mn)/(mx-mn)

    ## Mean-Standard Deviation normalization.
    def meanstd(im, mu=0.5, st=0.5):
        return (im-mu)/st
    mmax = minmax( image, mn, mx ) ## minimum - maximum
    mustd_def = meanstd( image )
    mustd_int = meanstd( image, mu=mu, st=st ) ## Mean - standard dev

    print('Min-Max Normalization - Min values:', mmax.min(axis=(0, 1)))
    print('Min-Max Normalization - Max values:',  mmax.max(axis=(0, 1)))

    print('Mean-Std Normalization (Using Dataset Stats) - Min values:', mustd_int.min(axis=(0, 1)))
    print('Mean-Std Normalization (Using Dataset Stats) - Max values:', mustd_int.max(axis=(0, 1)))

    print('Mean-Std Normalization (mean, std : 0.5) - Min values:', mustd_def.min(axis=(0, 1)))
    print('Mean-Std Normalization (mean, std : 0.5) - Max values:', mustd_def.max(axis=(0, 1)))