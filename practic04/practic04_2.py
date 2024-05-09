import argparse
import numpy as np
import pickle
from sklearn.metrics import completeness_score

#the structure of a KMeans class with methods for fitting the model to data (fit) and predicting cluster labels for new data (predict).he class also contains helper methods for initializing cluster centers, assigning data points to clusters, updating cluster centers, and checking for convergence.
class KMeansClustering:
    def __init__(self, n_clusters, imax=100): #imax = max iteration, n_clusters: The number of clusters (or centroids)
        self.n_clusters = n_clusters
        self.imax = imax
        self.centers = None

    def fit(self, data): #fitting the model to data
        # Randomly initialize cluster centers
        self.centers = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.imax):
            # Assign each data point to the nearest cluster
            labels = self.predict(data) #predicting cluster labels for new data

            # Update cluster centers based on the mean of the data points assigned to each cluster
            for i in range(self.n_clusters):
                self.centers[i] = np.mean(data[labels == i], axis=0)

    def predict(self, data):
        # Calculate distances between data points and cluster centers
        distances = np.linalg.norm(data[:, np.newaxis] - self.centers, axis=2)

        # Assign data points to the nearest cluster
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMeans Clustering")
    parser.add_argument("--data_name", type=str, default="data.pkl", help="Name of the pickle file to load data")
    args = parser.parse_args()

    # Load data
    with open(args.data_name, 'rb') as fid:
        data = pickle.load(fid)

    # Create KMeans object with 2 clusters
    kmeans_2 = KMeansClustering(n_clusters=2)
    kmeans_2.fit(data['X'])

    # Predict and evaluate completeness score for data 'Y_0_1'
    labels_pred_2 = kmeans_2.predict(data['Y_0_1'])
    completeness_2 = completeness_score(data['LY_0_1'], labels_pred_2)
    print("Completeness Score for 2 clusters:", completeness_2)

    # Create KMeans object with 3 clusters
    kmeans_3 = KMeansClustering(n_clusters=3)
    kmeans_3.fit(data['X'])

    # Predict and evaluate completeness score for data 'Y_0_1_2'
    labels_pred_3 = kmeans_3.predict(data['Y_0_1_2'])
    completeness_3 = completeness_score(data['LY_0_1_2'], labels_pred_3)
    print("Completeness Score for 3 clusters:", completeness_3)


'''
Instead, completeness score measures the completeness of the clusters, which is a more suitable metric for evaluating clustering algorithms in an unsupervised setting. Completeness score measures the extent to which all data points that are members of a given true class are assigned to the same cluster. It ranges from 0 to 1, with higher values indicating better clustering performance.
'''