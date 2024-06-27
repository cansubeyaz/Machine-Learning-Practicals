import argparse
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score
from logistic_regression import LogisticRegression
import cv2
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Extracting command line arguments', add_help=True)
    parser.add_argument('--exr', action='store', required=True)
    parser.add_argument('--dataset', action='store',
                        default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\colour_snippets')
    parser.add_argument('--classes', '-c', nargs="+", default=[])
    parser.add_argument('--split', action='store', type=float, default=0.3)
    parser.add_argument('--imax', action='store', type=int, default=10)
    parser.add_argument('--logreg', action='store', default='logreg_model.pkl')
    parser.add_argument('--image', action='store',
                        default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\practical_09\sweetpepper.png')
    return parser.parse_args()


def read_files_to_matrix(files, labels):
    feature_matrix = []
    feature_labels = []

    for file in files:
        image = cv2.imread(file)
        if image is None:
            print(f"Warning: Unable to read image file {file}. Skipping.")
            continue

        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                feature_matrix.append(np.concatenate((image[i, j], lab_image[i, j])))
                feature_labels.append(labels[file])

    return np.array(feature_matrix), np.array(feature_labels)


def exr0(flags):
    dataset_path = flags.dataset
    split_ratio = flags.split

    files = []
    labels = {}
    label_idx = 0

    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path):
                    files.append(file_path)
                    labels[file_path] = label_idx
            label_idx += 1

    feature_matrix, feature_labels = read_files_to_matrix(files, labels)

    split_index = int(len(feature_matrix) * split_ratio)
    train_features, eval_features = feature_matrix[:split_index], feature_matrix[split_index:]
    train_labels, eval_labels = feature_labels[:split_index], feature_labels[split_index:]

    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)

    train_features = (train_features - mean) / std
    eval_features = (eval_features - mean) / std

    # Train the model on all 6 features
    model = LogisticRegression(max_iterations=flags.imax)
    model.fit(train_features, train_labels)

    _, train_preds = model.predict(train_features)

    print("Model Accuracy:", accuracy_score(train_labels, train_preds))
    print("Confusion Matrix:\n", confusion_matrix(train_labels, train_preds))

    precision, recall, thresholds = precision_recall_curve(train_labels, train_preds)

    if len(thresholds) > 0:
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_thresh = thresholds[np.argmax(f1_scores[:-1])]
    else:
        best_thresh = 0.5  # Default threshold if none are found

    save_model(flags.logreg, model, mean, std, best_thresh)


def save_model(filename, model, mean, std, thresh):
    data = {
        'model': model,
        'mu': mean,
        'std': std,
        'thresh': thresh
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

        ## python main.py --exr exr0 --imax 10


def exr1(flags):
    with open(flags.logreg, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    mean = data['mu']
    std = data['std']
    threshold = data['thresh']

    image = cv2.imread(flags.image)
    if image is None:
        print(f"Error: Unable to read image file {flags.image}")
        return

    rows, cols, _ = image.shape
    reshaped_image = image.reshape(-1, 3)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    reshaped_lab_image = lab_image.reshape(-1, 3)

    # Normalize RGB and Lab channels separately
    normalized_rgb = (reshaped_image - mean[:3]) / std[:3]
    normalized_lab = (reshaped_lab_image - mean[3:]) / std[3:]

    # Concatenate normalized RGB and Lab features
    normalized_image = np.concatenate((normalized_rgb, normalized_lab), axis=1)

    probs, classifications = model.predict(normalized_image)

    segmented_image = np.array(classifications).reshape(rows, cols)
    probability_image = np.array([1 if p <= threshold else 0 for p in probs]).reshape(rows, cols)

    cv2.imwrite('segmented_image.png', segmented_image * 255)
    cv2.imwrite('probability_image.png', probability_image * 255)


if __name__ == "__main__":
    flags = parse_args()
    if flags.exr == 'exr0':
        exr0(flags)
    elif flags.exr == 'exr1':
        exr1(flags)

        ##  python main.py --exr exr1
