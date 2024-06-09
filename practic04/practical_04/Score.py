from sklearn.metrics import accuracy_score, completeness_score

# True labels
true_labels = [0, 0, 1, 1, 2, 2]

# Predicted labels from a clustering algorithm
predicted_labels = [1, 1, 0, 0, 2, 2]  # Assume the clustering labels are permuted

# Calculate accuracy score
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy Score: {accuracy}")

# Calculate completeness score
completeness = completeness_score(true_labels, predicted_labels)
print(f"Completeness Score: {completeness}")

"""
Use accuracy score for classification tasks where label consistency is guaranteed.
Use completeness score for clustering tasks to evaluate how well the algorithm groups data points of the same class together, regardless of label permutations.
"""