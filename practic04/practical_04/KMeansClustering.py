import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4,5,10,4,3,11,14,6,10,12]
y = [21,9,24,17,16,25,24,22,21,21]

plt.scatter(x,y, label = "Data Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D Data Points")
plt.legend(loc="lower right")
plt.show()

data = list(zip(x, y))
print(data)

inertias = []

for i in range(1,11): ## only have 10 data points, so the maximum number of clusters is 10. So for each value K in range(1,11)
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker = "o")
plt.title("Elbow Method") ##to select the good k number
plt.xlabel("Number of clusters") ## where the interia becomes more linear) is at K=2. according to plot 2 is good chose for k value
plt.ylabel("Inertia")
plt.show()

kmeans1 = KMeans(n_clusters=2)
kmeans1.fit(data)
labels = kmeans1.labels_

plt.scatter(x,y, c = labels, label = "data points")
centroids = kmeans1.cluster_centers_

# Create subplots
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

# Plot the initial data points
ax[0].scatter(x, y, label='Data Points', color='blue')
ax[0].set_title("2D Data Points")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].legend(loc="lower right")

# Plot the clustered data points
scatter = ax[1].scatter(x, y, c=labels, cmap='viridis', label='Data Points')

for i, centroid in enumerate(centroids):
    ax[1].scatter(*centroid, c='red', marker='x', s=100, label=f'Centroid {i+1}')
ax[1].set_title("KMeans Clustering")
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")

# Ensure the legend shows unique labels
handles, labels = ax[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys(), loc="lower right")

plt.tight_layout()
plt.show()
