import numpy as np
import matplotlib.pyplot as plt

# Generate the data
samples = 1000
X0 = np.random.normal(0.7, 0.3, size=(samples, 1))
X0 = np.clip(X0, 0., 1.)
X1 = np.random.normal(0.3, 0.1, size=(samples, 1))
X1 = np.clip(X1, 0., 1.)
L0 = np.ones((samples, 1))
L1 = np.zeros((samples, 1))
X = np.vstack((X0, X1))
L = np.vstack((L0, L1)).astype(bool)

# Plot the histogram
plt.hist(X, bins=50, range=(0., 1.))
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Generated Data')
plt.show()


class pr_metric:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def calculate_statistics(self, labels, predictions):
        TP = np.sum((labels == True) & (predictions == True))
        FP = np.sum((labels == False) & (predictions == True))
        MD = np.sum((labels == True) & (predictions == False))

        precision = TP / (TP + FP + self.epsilon)
        recall = TP / (TP + MD + self.epsilon)
        f1_score = 2 * precision * recall / (precision + recall + self.epsilon)

        return precision, recall, f1_score

    def __call__(self, labels, scores):
        unique_scores = np.unique(np.sort(scores))
        P, R, F1 = np.zeros_like(unique_scores), np.zeros_like(unique_scores), np.zeros_like(unique_scores)

        for i, s in enumerate(unique_scores):
            predictions = scores > s
            P[i], R[i], F1[i] = self.calculate_statistics(labels, predictions)

        best_index = np.argmax(F1)
        best_precision, best_recall, best_f1_score, best_threshold = P[best_index], R[best_index], F1[best_index], \
        unique_scores[best_index]

        # Plot precision-recall curve
        plt.plot(R, P, label="Precision-Recall curve")
        plt.scatter([R[best_index]], [P[best_index]], color='red', zorder=5,
                    label=f'Best F1 score (Threshold: {best_threshold:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

        print(f'Best Precision: {best_precision}')
        print(f'Best Recall: {best_recall}')
        print(f'Best F1-Score: {best_f1_score}')
        print(f'Best Threshold: {best_threshold}')

# Instantiate the pr_metric class
pr = pr_metric()

# Generate scores based on our data
scores = X.flatten()

# Call the class with labels and scores
pr(L.flatten(), scores)
