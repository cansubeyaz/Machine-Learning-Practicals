import argparse
import numpy as np
import pickle
from sklearn.cluster import SpectralClustering
from sklearn.metrics import completeness_score, accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Clustering")
    parser.add_argument("--data_name", type=str, default="data.pkl", help="Name of the pickle file to load data")
    parser.add_argument("--n_clusters", type=int, default=2, help="Number of clusters")
    args = parser.parse_args()

    # Load data
    with open(args.data_name, 'rb') as fid:
        data = pickle.load(fid)

    # Check if the number of samples in data['LY_0_1'] matches the number of samples in data['X']
    if len(data['LY_0_1']) != len(data['X']):
        raise ValueError("Number of samples in true labels (LY_0_1) does not match the number of samples in X")

    # Create SpectralClustering object
    spectral_clustering = SpectralClustering(n_clusters=args.n_clusters, assign_labels="discretize", random_state=42)

    # Fit and predict clusters for data 'X'
    labels_spectral = spectral_clustering.fit_predict(data['X'])

    # Calculate centroids based on the output from fit_predict and data 'X'
    centroids_spectral = np.array([np.mean(data['X'][labels_spectral == i], axis=0) for i in range(args.n_clusters)])

    # Calculate the centroid closest to each data 'Y_0_1' point
    closest_centroids = [np.argmin(np.linalg.norm(data_point - centroids_spectral, axis=1)) for data_point in data['Y_0_1']]

    # Calculate completeness score and accuracy score
    completeness_spectral = completeness_score(data['LY_0_1'], labels_spectral[:len(data['LY_0_1'])])
    accuracy_spectral = accuracy_score(data['LY_0_1'], closest_centroids)

    print("Completeness Score for Spectral Clustering:", completeness_spectral)
    print("Accuracy Score for Spectral Clustering:", accuracy_spectral)
