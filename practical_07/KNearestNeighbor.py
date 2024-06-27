from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import matplotlib.pyplot as plt

# Plot the predicted labels against one of the features
feature_index = 0  # Choose the index of the feature to plot against the predicted labels

plt.scatter(X_test[:, feature_index], y_test, color='black', label='True labels')
plt.scatter(X_test[:, feature_index], y_pred, color='red', label='Predicted labels')
plt.xlabel(iris.feature_names[feature_index])
plt.ylabel('Class')
plt.title('Comparison of True and Predicted Labels')
plt.legend()
plt.show()