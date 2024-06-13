import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from skimage import io


class KNN:
    def __init__(self, K):
        self.K = K

    def fit(self, X, y):
        self.X = X
        self.y = y

    def euclid(self, x):
        return np.sqrt(np.sum((self.X - x) ** 2, axis=1))

    def predict(self, D):
        cls = np.zeros(D.shape[0])
        for i, sample in enumerate(D):
            distances = self.euclid(sample)
            k_indices = np.argsort(distances)[:self.K]
            k_nearest_labels = self.y[k_indices]
            cls[i] = mode(k_nearest_labels).mode[0]
        return cls


def load_and_preprocess_data(dataset_location, classes):
    samples, labels = [], []
    for label, color in enumerate(classes):
        file_paths = [os.path.join(dataset_location, color, file) for file in
                      os.listdir(os.path.join(dataset_location, color))]
        for file_path in file_paths:
            image = io.imread(file_path) / 255.0
            image = (image - 0.5) / 0.5
            reshaped_image = image.reshape(-1, 3)
            samples.append(reshaped_image)
            labels.append(np.full(reshaped_image.shape[0], label))
    samples = np.vstack(samples)
    labels = np.concatenate(labels)
    return samples, labels


def main(args):
    # Load and preprocess data
    samples, labels = load_and_preprocess_data(args.dataset_location, args.classes)

    # Split data into training and evaluation sets
    X_train, X_eval, y_train, y_eval = train_test_split(samples, labels, test_size=0.3, random_state=42)

    # Create KNN object and fit the model
    knn = KNN(K=args.K)
    knn.fit(X_train, y_train)

    # Predict on evaluation data
    y_pred = knn.predict(X_eval)

    # Print confusion matrix
    cm = confusion_matrix(y_eval, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(args.classes))
    plt.xticks(tick_marks, args.classes, rotation=45)
    plt.yticks(tick_marks, args.classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=r"C:\Users\Lenovo\Desktop\PAML_2024\data\colour_snippets")
    parser.add_argument('--dataset_location', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--classes', '-c', nargs='+', required=True, help='List of colors to use in the dataset')
    parser.add_argument('--K', type=int, default=3, help='Number of neighbors for KNN')

    args = parser.parse_args()
    main(args)

    ## python .\knn_classifier.py --dataset_location C:\Users\Lenovo\Desktop\PAML_2024\data\colour_snippets -c black white --K 3

