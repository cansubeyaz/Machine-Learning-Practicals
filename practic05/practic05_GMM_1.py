import numpy as np
import pickle

class MVG:
    def __init__(self):
        self.mean = None
        self.covariance = None
        self.constants_sum = None

    def train(self, X):
        self.mean = np.mean(X, axis=0)
        self.covariance = np.cov(X, rowvar=False, bias=True)
        self._precalculations()

    def _precalculations(self):
        self.inverse_covariance = np.linalg.inv(self.covariance)
        n = len(self.mean)
        self.constant1 = -(n / 2) * np.log(2 * np.pi)
        self.constant2 = -0.5 * np.linalg.slogdet(self.covariance)[1]
        self.constants_sum = self.constant1 + self.constant2

    def log_likelihood(self, X):
        residuals = X - self.mean
        exponent_term = -0.5 * np.sum(residuals @ self.inverse_covariance * residuals, axis=1)
        log_likelihoods = self.constants_sum + exponent_term
        return log_likelihoods

# Load data
try:
    with open(r'C:\Users\Lenovo\machine_learning\practic05\mvg_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    LX = data['LX']
    Y01 = data['Y01']
    LY01 = data['LY01']

    X0 = X[LX == 0]
    X1 = X[LX == 1]

    # Train MVG objects
    mvg0 = MVG()
    mvg0.train(X0)

    mvg1 = MVG()
    mvg1.train(X1)

    # Calculate log likelihoods
    log_likelihoods = np.column_stack((mvg0.log_likelihood(Y01), mvg1.log_likelihood(Y01)))

    # Classify using argmax
    predictions = np.argmax(log_likelihoods, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predictions == LY01)
    print("Accuracy:", accuracy)

    # Calculate confusion matrix
    confusion_matrix = np.zeros((2, 2))
    for true_label, predicted_label in zip(LY01, predictions):
        confusion_matrix[true_label][predicted_label] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)

except FileNotFoundError:
    print("Error: Data file not found.")
except KeyError as e:
    print("Error: Key '{}' not found in the data dictionary.".format(e))
except Exception as e:
    print("An error occurred:", e)
