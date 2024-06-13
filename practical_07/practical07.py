import argparse
import os
from skimage.feature import hog
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def extract_hog_features(image_path, orientations, pixels_per_cell, cells_per_block):
    image = io.imread(image_path, as_gray=True)
    hog_features = hog(image, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block), feature_vector=True)
    return hog_features.reshape(-1, orientations)


def prepare_dataset(dataset_location, testing_split, labels):
    train_files, eval_files, train_labels, eval_labels = [], [], [], []
    for label in labels:
        file_paths = [f"{dataset_location}/{label}/{file}" for file in os.listdir(f"{dataset_location}/{label}")]
        train, eval = train_test_split(file_paths, test_size=testing_split, random_state=42)
        train_files.extend(train)
        eval_files.extend(eval)
        train_labels.extend([label] * len(train))
        eval_labels.extend([label] * len(eval))
    return train_files, eval_files, train_labels, eval_labels


def extract_hog_feature_matrix(files, labels, orientations, pixels_per_cell, cells_per_block):
    hog_features, hog_labels = [], []
    for file, label in zip(files, labels):
        features = extract_hog_features(file, orientations, pixels_per_cell, cells_per_block)
        hog_features.append(features)
        hog_labels.append(label)
    return np.vstack(hog_features), np.array(hog_labels)


class BoVW:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X):
        self.kmeans.fit(X)

    def predict(self, X):
        visual_words = self.kmeans.predict(X)
        histograms = np.zeros((len(X), self.n_clusters))
        for i, word in enumerate(visual_words):
            histograms[i, word] += 1
        return histograms


def main(args):
    # Prepare dataset
    train_files, eval_files, train_labels, eval_labels = prepare_dataset(args.dataset_location, args.testing_split,
                                                                         args.labels.split(','))

    # Extract HOG features
    train_features, train_labels = extract_hog_feature_matrix(train_files, train_labels, args.orientations,
                                                              args.pixels_per_cell, args.cells_per_block)

    # Train BoVW model
    bovw = BoVW(n_clusters=args.n_clusters)
    bovw.fit(train_features)

    # Predict and visualize histograms
    eval_features, eval_labels = extract_hog_feature_matrix(eval_files, eval_labels, args.orientations,
                                                            args.pixels_per_cell, args.cells_per_block)
    eval_histograms = bovw.predict(eval_features)

    # Plot histograms (example)
    for hist, label in zip(eval_histograms, eval_labels):
        plt.figure()
        plt.title(f"Histogram for {label}")
        plt.bar(range(args.n_clusters), hist)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOG and BoVW example")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--testing_split', type=float, default=0.2, help='Fraction of data to be used for evaluation')
    parser.add_argument('--labels', type=str, required=True,
                        help='Comma-separated list of labels (subdirectory names) in the dataset')
    parser.add_argument('--orientations', type=int, default=9, help='Number of gradient orientations for HOG')
    parser.add_argument('--pixels_per_cell', type=int, default=8, help='Size of a cell in pixels for HOG')
    parser.add_argument('--cells_per_block', type=int, default=1, help='Number of cells in each block for HOG')
    parser.add_argument('--n_clusters', type=int, default=50, help='Number of clusters (visual words) for BoVW')

    args = parser.parse_args()
    main(args)
