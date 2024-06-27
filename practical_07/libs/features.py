import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog

from practic04.practical_04.libs.clustering import KMeans


def extract_hog(file, o, ppc, cpb):
    img = rgb2gray(imread(file))
    feat = hog(img, orientations=o, pixels_per_cell=ppc, cells_per_block=cpb, visualize=False, feature_vector=True)
    return feat.reshape(-1, o)


def extract_train_hog_feature_vector(files, labels, o, ppc, cpb):
    feats = None  # Initialize feats outside of the loop
    labs = []

    for f, i in zip(files, labels):
        fe = extract_hog(f, o, ppc, cpb)
        if feats is None:
            feats = fe
        else:
            feats = np.vstack((feats, fe))

        labs += [i] * fe.shape[0]

    return feats, np.array(labs)


class BoVW:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit(self, X):
        self.kmeans = KMeans(self.num_clusters)
        self.kmeans.fit(X)

    def predict(self, X):
        fv = self.kmeans.predict(X)
        h, _ = np.histogram(fv, bins=self.num_clusters)
        return h
