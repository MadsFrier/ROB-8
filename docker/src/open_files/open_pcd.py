import open3d as o3d
import numpy as np



pcd_path = "/workspaces/ROB-8/docker/src/content/meeti_db/rgb_pcd/rgb_map.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)


o3d.visualization.draw_geometries([pcd])
