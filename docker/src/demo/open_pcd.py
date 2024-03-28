import open3d as o3d

pcd_path = "/workspaces/ROB-8/docker/src/content/demo_data/seg_pcd/5LpN3gDmAk7_132.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
o3d.visualization.draw_geometries([pcd])