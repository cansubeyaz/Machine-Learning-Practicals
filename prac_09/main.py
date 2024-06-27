import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from utils import save_model, load_model
from skimage import color
from PIL import Image
import argparse
import pickle

def read_files_to_matrix(dataset, labels):
    # Dummy implementation for illustration
    # You need to implement reading files, converting to Lab, and extracting features.
    # For now, let's use some synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    y = np.random.randint(0, 2, 100)  # 100 binary labels
    return X, y

def normalize_features(X_train, X_eval):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_eval = (X_eval - mean) / std
    return X_train, X_eval, mean, std

def extract_rgb_features(X):
    return X  # Assuming X is already in RGB format

def extract_lab_features(X):
    return color.rgb2lab(X.reshape(-1, 1, 3)).reshape(-1, 3)

def plot_precision_recall_curve(precision, recall, label):
    plt.plot(recall, precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

def exr0(flags):
    # Reading and splitting the dataset
    X, y = read_files_to_matrix(flags.dataset, flags.classes)
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=flags.split, random_state=42)

    # Feature extraction
    X_train_rgb, X_eval_rgb, mean_rgb, std_rgb = normalize_features(X_train, X_eval)
    X_train_lab, X_eval_lab, mean_lab, std_lab = normalize_features(extract_lab_features(X_train), extract_lab_features(X_eval))

    # Creating and training logistic regression models
    logreg_rgb = LogisticRegression(max_iterations=flags.imax)
    logreg_lab = LogisticRegression(max_iterations=flags.imax)

    logreg_rgb.fit(X_train_rgb, y_train)
    logreg_lab.fit(X_train_lab, y_train)

    # Making predictions
    y_pred_prob_rgb, y_pred_class_rgb = logreg_rgb.predict(X_eval_rgb)
    y_pred_prob_lab, y_pred_class_lab = logreg_lab.predict(X_eval_lab)

    # Calculate metrics for RGB
    precision_rgb, recall_rgb, thresholds_rgb = precision_recall_curve(y_eval, y_pred_prob_rgb)
    threshold_rgb = thresholds_rgb[np.argmax(precision_rgb + recall_rgb - 1)]  # Best threshold

    # Save the RGB model
    save_model(logreg_rgb, flags.logreg.replace('.pkl', '_rgb.pkl'), mean_rgb, std_rgb, threshold_rgb)

    # Output metrics for RGB
    print("RGB Model Accuracy:", accuracy_score(y_eval, y_pred_class_rgb))
    print("RGB Confusion Matrix:\n", confusion_matrix(y_eval, y_pred_class_rgb))

    # Plot precision-recall curve for RGB
    plot_precision_recall_curve(precision_rgb, recall_rgb, 'RGB')

    # Calculate metrics for Lab
    precision_lab, recall_lab, thresholds_lab = precision_recall_curve(y_eval, y_pred_prob_lab)
    threshold_lab = thresholds_lab[np.argmax(precision_lab + recall_lab - 1)]  # Best threshold

    # Save the Lab model
    save_model(logreg_lab, flags.logreg.replace('.pkl', '_lab.pkl'), mean_lab, std_lab, threshold_lab)

    # Output metrics for Lab
    print("Lab Model Accuracy:", accuracy_score(y_eval, y_pred_class_lab))
    print("Lab Confusion Matrix:\n", confusion_matrix(y_eval, y_pred_class_lab))

    # Plot precision-recall curve for Lab
    plot_precision_recall_curve(precision_lab, recall_lab, 'Lab')

    plt.show()

def extract_features_from_image(image_array):
    # Dummy implementation for illustration
    # You need to implement actual feature extraction logic.
    return image_array.reshape(-1, 3)  # Flatten the image to (N, 3) shape

def exr1(flags):
    # Load the logistic regression model
    data = load_model(flags.logreg)
    model = data['model']
    mean = data['mu']
    std = data['std']
    threshold = data['thresh']

    # Load and preprocess the image
    image = Image.open(flags.image)
    image_array = np.array(image)
    rows, cols, _ = image_array.shape

    # Extract features from the image
    features = extract_features_from_image(image_array)
    features = (features - mean) / std

    # Predict the segmentation
    y_pred_prob, y_pred_class = model.predict(features)

    # Reshape and display segmentation result
    segmentation_image_class = np.array(y_pred_class).reshape(rows, cols)
    plt.imshow(segmentation_image_class, cmap='gray')
    plt.title('Segmented Image (Class)')
    plt.show()

    # Save the class segmentation image
    class_image = Image.fromarray((segmentation_image_class * 255).astype(np.uint8))
    class_image.save('segmentation_class.png')

    # Using the pixel probabilities and threshold
    y_pred_thresh = [1 if p > threshold else 0 for p in y_pred_prob]
    segmentation_image_thresh = np.array(y_pred_thresh).reshape(rows, cols)
    plt.imshow(segmentation_image_thresh, cmap='gray')
    plt.title('Segmented Image (Threshold)')
    plt.show()

    # Save the threshold segmentation image
    thresh_image = Image.fromarray((segmentation_image_thresh * 255).astype(np.uint8))
    thresh_image.save('segmentation_threshold.png')

def save_model(model, filename, mean, std, threshold):
    dit = {
        'model': model,
        'mu': mean,
        'std': std,
        'thresh': threshold
    }
    with open(filename, 'wb') as fid:
        pickle.dump(dit, fid)

def parse_args():
    parser = argparse.ArgumentParser(description='Extracting command line arguments', add_help=True)
    parser.add_argument('--exr', action='store', required=True)
    parser.add_argument('--dataset', action='store', default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\data\colour_snippets')
    parser.add_argument('--classes', '-c', nargs='+', default=[])
    parser.add_argument('--split', action='store', type=float, default=0.3)
    parser.add_argument('--imax', action='store', type=int, default=10)
    parser.add_argument('--logreg', action='store', default='logreg_model.pkl')
    parser.add_argument('--image', action='store', default=r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\prac_09\sweetpepper.png')
    return parser.parse_args()

if __name__ == "__main__":
    flags = parse_args()
    if flags.exr == 'exr0':
        exr0(flags)
    elif flags.exr == 'exr1':
        exr1(flags)
