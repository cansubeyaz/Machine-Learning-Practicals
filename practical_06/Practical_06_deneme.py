import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

from skimage.io import imread, imsave

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve as prc

from libs.outlier import Z_score_mod, Z_score
from libs.metrics import pr_curve
from libs.Gaussian import MultiGMM

np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='Extracting command line arguments', add_help=True)
parser.add_argument('--exr', action='store', required=True)
parser.add_argument('--split', action='store', type=float, default=0.3)
parser.add_argument('--verbosity', action='store_true')
flags = parser.parse_args()

# preamble
exr0 = True if flags.exr == 'exr0' else False
exr1 = True if flags.exr == 'exr1' else False
exr2 = True if flags.exr == 'exr2' else False

if exr0:
    with open(r'C:\Users\Lenovo\Desktop\PAML_2024\practical_06\mvg_data2.pkl', 'rb') as fid:
        data = pickle.load(fid)

    # Assuming data is a dictionary with at least one key containing the data array
    key = list(data.keys())[0]  # Get the first key
    data_array = data[key]

    print(type(data))
    print('original shape', data_array.shape)
    print('first')
    print(data_array.mean(axis=0))
    print(np.cov(data_array.T))

    # get rid of the nans
    loc = np.unique(np.where(np.isnan(data_array)))
    print(loc)
    data_array = np.delete(data_array, loc, 0)
    print('second')
    print(data_array.mean(axis=0))
    print(np.cov(data_array.T))

    # get rid of the infs
    loc = np.unique(np.where(np.isinf(data_array)))
    print(loc)
    data_array = np.delete(data_array, loc, 0)
    print('third')
    print(data_array.mean(axis=0))
    print(np.cov(data_array.T))

    xz = Z_score(data_array)
    print('z score')
    print(xz.shape)
    print(xz.mean(axis=0))
    print(np.cov(xz.T))

    xzm = Z_score_mod(data_array)
    print('z mod')
    print(xzm.shape)
    print(xzm.mean(axis=0))
    print(np.cov(xzm.T))

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].scatter(data_array[:, 0], data_array[:, 1])
    ax[0].set_title('Original')
    ax[1].scatter(xz[:, 0], xz[:, 1])
    ax[1].set_title('Z score')
    ax[2].scatter(xzm[:, 0], xzm[:, 1])
    ax[2].set_title('Z score mod')
    plt.show()

if exr1:
    # get the data to use
    samples = 1000
    X0 = np.random.normal(0.7, 0.3, size=(samples, 1))
    X0 = np.clip(X0, 0., 1.)
    X1 = np.random.normal(0.3, 0.1, size=(samples, 1))
    X1 = np.clip(X1, 0., 1.)
    L0 = np.ones((samples, 1))
    L1 = np.zeros((samples, 1))
    X = np.vstack((X0, X1))
    L = np.vstack((L0, L1)).astype(bool)

    # plt.figure()
    # plt.hist( X0, bins=50, range=(0.,1.) )
    # plt.hist(X1, bins=50, range=(0., 1.))
    # plt.show()

    pr = pr_curve()
    print(pr.calculate_statistic(L, X > 0.5))
    print(pr.calculate_statistic(L, X > 0.4))
    print(pr.calculate_statistic(L, X > 0.6))

    pr(L, X)

if exr2:
    colours = ['black', 'grey']
    train = {}
    test_first = True
    loc = r'C:\Users\Lenovo\Desktop\PAML_2024\data\colour_snippets'
    for i, c in enumerate(sorted(os.listdir(loc))):
        if c in colours:
            first = True
            for f in sorted(glob.glob(os.path.join(loc, c, '*.png'))):
                rgb = imread(f).reshape((-1, 3))
                if first:
                    v = rgb
                    first = False
                else:
                    v = np.vstack((v, rgb))
            train[c], e = train_test_split(v, test_size=flags.split)
            if test_first:
                test = e
                labels = np.zeros((e.shape[0],))
                test_first = False
            else:
                test = np.vstack((test, e))
                labels = np.hstack((labels, np.ones((e.shape[0],))))
    print(train.keys())
    print(test.shape, labels.shape)
    print(np.unique(labels))

    gmm = MultiGMM(5)
    gmm.fit(train)

    _, scores = gmm.predict(test)
    print(scores.shape)
    scores = scores[:, 1].reshape(-1, 1)

    p, r, t = prc(labels, scores)
    f1 = 2 * p * r / (p + r + 0.000000001)
    am = np.argmax(f1)
    plt.figure()
    plt.plot(r, p)
    plt.plot(r[am], p[am], 'r*')
    plt.title('Precision recall curve - Precision Recall: F1-score of {}'.format(f1[am]))
    plt.show()
    print(f'Precision {p[am]}')
    print(f'Recall {r[am]}')
    print(f'F1 {f1[am]}')
    print(f'threshold {t[am]}')

## python.\Practical_06_deneme.py - -exr exr2

