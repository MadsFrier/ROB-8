import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd_path = "/workspaces/ROB-8/docker/src/content/meeti_db/rgb_pcd/rgb_map.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
z_threshold = 0.7
z_t = -1.32

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

mask = points[:,2] < z_threshold

points = points[mask]
colors = colors[mask]

mask = points[:,2] > z_t

points = points[mask]
colors = colors[mask]

points[:,2] = 0

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

#mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#    size=0.6, origin=[-2, -2, -2])

o3d.visualization.draw_geometries([pcd], zoom=0.27999999999999958,
                                  front=[ 0.081165611950153704, 0.022528435461310507, 0.99644599102631892 ],
                                  lookat=[ -0.5374871461257219, 2.4917603057239077, -0.0075029322434033048 ],
                                  up=[ 0.78370178016044789, 0.6162495248044797, -0.077769164529381679 ])

o3d.io.write_point_cloud('/workspaces/ROB-8/docker/src/content/meeti_db/rgb_pcd/rgb_2d_map.pcd', pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)