import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import SpectralClustering
# Load the data (assuming data is in the same format as the previous examples)
import pickle

from sklearn.preprocessing import StandardScaler

with open(r'C:\Users\cansu\OneDrive\Desktop\PAML_2024\practical_05\mvg_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract X and LX
X = data['X'] ## rgb train
LX = data['LX'] ## true label

# Standardize the data -  scaling the data has improved the performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
LX_scaled = scaler.fit_transform(LX)
print("shape X :", X_scaled.shape) ## (200,2)
print("shape LX :",LX_scaled.shape) ## (200,1)

# Fit the GMM
gmm = GaussianMixture(n_components=2, random_state=0).fit(X_scaled)

# Predict probabilities and classes
probabilities = gmm.predict_proba(X_scaled)
predicted_classes = gmm.predict(X_scaled)

# Evaluate the model
accuracy = accuracy_score(LX_scaled, predicted_classes)
conf_matrix = confusion_matrix(LX_scaled, predicted_classes)

print("GMM Accuracy:", accuracy)
print("GMM Confusion Matrix:\n", conf_matrix)

# Print the means and covariances
print("GMM Means:\n", gmm.means_)
print("GMM Covariances:\n", gmm.covariances_)

###  SPECTRAL CLUSTERING

spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)
labels = spectral.fit_predict(X_scaled)

# Evaluate the model
accuracy = accuracy_score(LX_scaled, labels)
conf_matrix = confusion_matrix(LX_scaled, labels)

print("Spectral Clustering Accuracy:", accuracy)
print("Spectral Clustering Confusion Matrix:\n", conf_matrix)

