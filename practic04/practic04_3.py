import pickle
import argparse
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import completeness_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Clustering")
    parser.add_argument("--data_name", type=str, default="data.pkl", help="Name of the pickle file to load data")
    parser.add_argument("--n_clusters", type=int, default=2, help="Number of clusters for spectral clustering")
    args = parser.parse_args()

    # Load data
    with open(args.data_name, 'rb') as fid:
        data = pickle.load(fid)

    # Create spectral clustering object
    spectral_clustering = SpectralClustering(n_clusters=args.n_clusters, assign_labels="discretize", random_state=0)

    # Fit and predict
    labels_pred = spectral_clustering.fit_predict(data['X'])

    # Calculate centroids based on the output from fit_predict and data['X']
    centroids = np.array([np.mean(data['X'][labels_pred == i], axis=0) for i in range(args.n_clusters)])

    # Ensure centroids are calculated correctly
    print("Centroids:", centroids)

    # Ensure labels_pred has the same number of samples as data['LY_0_1']
    labels_pred = labels_pred[:len(data['LY_0_1'])]

    # Calculate completeness score
    completeness = completeness_score(data['LY_0_1'], labels_pred)
    print("Completeness Score:", completeness)

#  A completeness score close to 0 indicates poor clustering performance, while a score close to 1 indicates perfect clustering.