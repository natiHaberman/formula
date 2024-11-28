import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the point cloud data from CSV
file_path = 'Frame_702_Refl_1.csv'
data = pd.read_csv(file_path)

# Extract only the 'x', 'y', and 'z' columns
point_cloud = data[["x", "y", "z"]]


# Calculate distances
distances = np.sqrt(point_cloud["x"]**2 + point_cloud["y"]**2 + point_cloud["z"]**2)
threshold = distances.mean() + 3 * distances.std()  # 3 standard deviations

# Filter points
filtered_cloud = point_cloud[distances < threshold]


# Ground point removal
ground_threshold = 0.1  # Needs adjustment
no_ground_cloud = filtered_cloud[filtered_cloud["z"] > ground_threshold]


# Prepare data for clustering
data = no_ground_cloud[["x", "y", "z"]].values
dbscan = DBSCAN(eps=0.5, min_samples=10)  # Parameter adjustment needed
labels = dbscan.fit_predict(data)

# Add cluster labels to the DataFrame
no_ground_cloud["cluster"] = labels


# Plot clusters in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(no_ground_cloud["x"], no_ground_cloud["y"], no_ground_cloud["z"], c=no_ground_cloud["cluster"], cmap="viridis")
plt.colorbar(scatter, label="Cluster")
plt.show()