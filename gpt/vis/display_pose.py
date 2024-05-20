import open3d as o3d
import numpy as np
import json
import time

def vis_init(full_scrn=False, pcd_path="/Users/madsf/Documents/github/ROB-8/gpt/Landmark/rgb_2d_map.pcd"):
    pcd = o3d.io.read_point_cloud(pcd_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=0.01)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(mesh_cylinder)

    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.05, cylinder_height=0.2, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
    mesh_arrow.compute_vertex_normals()
    mesh_arrow.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(mesh_arrow)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame().translate((1, -3.5, 0))
    vis.add_geometry(mesh_frame)
    
    T_origin_cylinder = transform_robot_euler(mesh_arrow, 0, 0, 0, 0, 0)
    T_origin_arrow = transform_robot_euler(mesh_arrow, 0, 0, 0, np.pi/2, 0)

    mesh_cylinder.transform(T_origin_cylinder)
    mesh_arrow.transform(T_origin_arrow)    

    if full_scrn:
        vis.toggle_full_screen()
        
    vis.set_view_status('''{
        "class_name" : "ViewTrajectory",
        "interval" : 29,
        "is_loop" : true,
        "trajectory" : 
        [
            {
                "boundingbox_max" : [ 0.29999999999999999, 0.29999999999999999, 2.0 ],
                "boundingbox_min" : [ -0.29999999999999999, -0.29999999999999999, -2.0 ],
                "field_of_view" : 60.0,
                "front" : [ -0.014904310150318859, 0.026503511320607263, 0.99953760580911699 ],
                "lookat" : [ -0.6615656249598999, 2.6645809815335051, -0.20109738382577624 ],
                "up" : [ 0.77949876669366092, 0.62638391953166639, -0.0049857873208546503 ],
                "zoom" : 0.84000000000000008
            }
        ],
        "version_major" : 1,
        "version_minor" : 0
    }''') 
        
    return vis, pcd, mesh_cylinder, mesh_arrow

def transform_robot_euler(mesh_arrow, x, y, e1, e2, e3):
    T = np.eye(4)
    T[:3, :3] = mesh_arrow.get_rotation_matrix_from_xyz((e1, e2, e3))
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = 0.01

    return T

def update_robot(vis, pcd, mesh_cylinder, mesh_arrow, x, y, yaw):     
    T = transform_robot_euler(mesh_arrow, x, y, 0, 0, yaw)
    vis.update_geometry(pcd)
    
    mesh_cylinder.transform(T)
    mesh_arrow.transform(T)
    
    vis.update_geometry(mesh_cylinder)
    vis.update_geometry(mesh_arrow)
    
    vis.poll_events()
    vis.update_renderer()
    
    T_inv = np.linalg.inv(T)
    mesh_cylinder.transform(T_inv)
    mesh_arrow.transform(T_inv)  

# file_path = 'C:/Users/Christian/Documents/GitHub/ROB-8/gpt/ChatGPT/landmarks.json'
# with open(file_path, 'r') as file:
#     landmark_dict = json.load(file)
    
# pos_list = []

# for lm in landmark_dict:
#     pos = landmark_dict.get(lm)
#     pos_list.append(pos[:2])
    
# #-------------------------#

# vis, pcd, mesh_cylinder, mesh_arrow = vis_init(pcd_path='C:/Users/Christian/Documents/GitHub/ROB-8/gpt/Landmark/rgb_2d_map.pcd')

# while True:
#     for i in pos_list:
#         update_robot(vis, pcd, mesh_cylinder, mesh_arrow,  i[0], i[1], 0)
#         time.sleep(1)