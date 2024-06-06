from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assuming you have created and trained your MultiGMM object

# Get scores from MultiGMM predict
classification, scores = multi_gmm.predict(test_data)

# Use only the second column of scores
scores = scores[:, 1].reshape(-1, 1)

# Get precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(test_labels, scores)

# Calculate F1-score
f1_score = 2 * precision * recall / (precision + recall)

# Find the index of the best F1-score
best_index = np.argmax(f1_score)

# Plot precision-recall curve
plt.plot(recall, precision, label="Precision-Recall curve")
plt.scatter(recall[best_index], precision[best_index], color='red', zorder=5, label=f'Best F1 score (Threshold: {thresholds[best_index]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

print(f'Best Precision: {precision[best_index]}')
print(f'Best Recall: {recall[best_index]}')
print(f'Best F1-Score: {f1_score[best_index]}')
print(f'Best Threshold: {thresholds[best_index]}')
