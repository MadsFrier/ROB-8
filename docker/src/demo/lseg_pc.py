from image_generation import *
import open3d as o3d
import cv2

if __name__ == "__main__":
    
    # choose directory of images to load in
    data_dir = "/workspaces/ROB-8/docker/src/content/demo_data/"
    
    # choose rgb image to load in
    rgb_name = "5LpN3gDmAk7_1.png"
    
    # choose depth image to load in
    depth_name = "5LpN3gDmAk7_1.png"
    
    # show rgb and depth images
    create_image(data_dir, rgb_name, depth_name, show=True)
    
    color = o3d.io.read_image(data_dir + 'rgb/' + rgb_name)
    depth = o3d.io.read_image(data_dir + 'depth/' + depth_name)
            
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    
    o3d.visualization.draw_geometries([pcd])