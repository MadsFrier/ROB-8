import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

np.random.seed(1)
num_points = 300
cluster_params = [
 {"mean": np.array([0, 0, 0]), "cov": np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])},
 {"mean": np.array([4, 4, 4]), "cov": np.array([[1, -0.8, -0.8], [-0.8, 1, -0.8], [-0.8, -0.8, 1]])},
 {"mean": np.array([-3, -4, -5]), "cov": np.array([[1, -0.8, -0.8], [-0.8, 1, -0.8], [-0.8, -0.8, 1]])}
]

clusters = []
for param in cluster_params:
 cluster = np.random.multivariate_normal(param["mean"], param["cov"], num_points // 3)
 clusters.append(cluster)
points = np.vstack(clusters)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Synthetic Point Cloud')
plt.show()

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
eps = 1.2 # Distance threshold for points in a cluster
min_points = 10 # Minimum number of points per cluster
dbscan_labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

print("Cluster labels (with -1 indicating noise): ")
print(f"Labels: {dbscan_labels}")

colors = plt.get_cmap("tab10")(dbscan_labels)
colors[dbscan_labels == -1] = [0.5, 0.5, 0.5, 1]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()