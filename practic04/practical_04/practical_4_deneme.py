import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import completeness_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import spectral_clustering

from libs.Kmean_Clustering import KMeans

## ARGPARSE COMMAND
parser = argparse.ArgumentParser(description="Extracting command line arguments", add_help=True)
parser.add_argument("--exr", action="store", default="None")
parser.add_argument("--create_data", action="store_true")
parser.add_argument("--data_name", action="store", default="mvg_data_deneme.pkl")
parser.add_argument("--train_samples", action="store", type=int, default=100, help="Create Train samples")
parser.add_argument("--test_samples", action="store", type=int, default=50, help="Create Test Samples")
parser.add_argument("-numberlusters", action="store", type=int, default=2)

flag = parser.parse_args()

exr0 = True if flag.exr == "exr0" else False ## KMeans Clustering
exr1 = True if flag.exr == "exr1" else False ## Spectral Clustering
exr2 = True if flag.exr == "exr2" else False ## Image Normalisation

print(flag.create_data)

if flag.create_data:
    ## Means and Covariances
    mu_0 = np.array([0,0])
    cov_0 = np.array([[1,0],[0,1]])

    mu_1 = np.array([4,1])
    cov_1 = np.array([[1,0],[0,1]])

    mu_2 = np.array([2,2])
    cov_2 = np.array([[0.5,0],[0,0.5]])

    ## Create Multivariate Gaussian Data
    X0_train = np.random.multivariate_normal(mu_0, cov_0, flag.train_samples)
    X0_test = np.random.multivariate_normal(mu_0, cov_0, flag.test_samples)
    X1_train = np.random.multivariate_normal(mu_1, cov_1, flag.train_samples)
    X1_test = np.random.multivariate_normal(mu_1, cov_1, flag.test_samples)
    X2_test = np.random.multivariate_normal(mu_2, cov_2, flag.test_samples)

    print(X0_train)
    print(X0_train.shape) ##(100,2) 100 rows, 2 columns
    print("X0_test :", X0_test.shape) ## (50,2) 50 rows, 2 columns
    print("X1_train :", X1_train.shape) ## (100,2)

    plt.figure()
    plt.scatter(X0_train[:,0], X0_train[:,1], color = "red", label="X0_train")
    plt.scatter(X0_test[:,0], X0_test[:,1], color = "blue", label = "X0_test")
    plt.scatter(X1_train[:,0], X1_train[:,1], color = "green", label = "X1_train")
    plt.scatter(X1_test[:,0], X1_test[:,1], color = "yellow", label = "X1_test")
    plt.scatter(X2_test[:,0], X2_test[:,1], color = "magenta", label = "X2_test")

    plt.legend(loc = "lower right")
    plt.show()

    ## DICTIONARY OF DATA
    data = {}
    data['X'] = np.vstack((X0_train, X1_train))
    data['LX'] = np.vstack((np.zeros((flag.train_samples, 1)), np.ones( (flag.train_samples, 1) )) )
    data['Y_0_1'] = np.vstack((X0_test, X1_test))
    data['Y_0_1_2'] = np.vstack((X0_test, X1_test, X2_test))
    data['LY_0_1'] = np.vstack((np.zeros((flag.test_samples, 1)), np.ones((flag.test_samples, 1))))
    data['LY_0_1_2'] = np.vstack((np.zeros((flag.test_samples, 1)), np.ones((flag.test_samples, 1)), np.ones((flag.test_samples, 1)) * 2))

    ## Save the dictionary into a pickle file
    with open(flag.data_name, 'wb') as fid: ## wb : write binary
        pickle.dump(data, fid)


## TERMINAL
## python .\practical_4_deneme.py --create_data

if exr0:
    with open(flag.data_name, "rb") as fid:
        data = pickle.load(fid)
        print(data.keys()) ## dict_keys(['X', 'LX', 'Y_0_1', 'Y_0_1_2', 'LY_0_1', 'LY_0_1_2'])

    obj = KMeans(flag.ncluster, imax=100)
    obj.fit(data["X"])
    predict = obj.predict(data["Y_0_1_2"])
    plt.figure()

    for i in range(flag.nclusters ):
        plt.scatter(data['Y_0_1_2'][predict==i,0], data['Y_0_1_2'][predict==i,1], label=f'centroid{i}' )
    plt.legend()
    plt.show()
    print("shape", data["X"].shape)
    print( completeness_score(predict, data['LY_0_1_2'].reshape( -1, )))
    print( accuracy_score(predict, data['LY_0_1_2'].reshape( -1, )))



