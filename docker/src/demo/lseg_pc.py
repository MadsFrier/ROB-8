import open3d as o3d
import cv2
import matplotlib.pyplot as plt

def show_images(rgbd_image):
    plt.subplot(1, 2, 1)
    plt.title('RGB')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth')
    plt.imshow(rgbd_image.depth)
    plt.show()  

if __name__ == "__main__":
    
    # choose directory of images to load in
    data_dir = "/workspaces/ROB-8/docker/src/content/demo_data/"
    
    # choose rgb image to load in
    rgb_name = "5LpN3gDmAk7_130.png"
    
    # choose depth image to load in
    depth_name = "5LpN3gDmAk7_130.png"
    
    # load color and depth images
    color = o3d.io.read_image(data_dir + 'rgb/' + rgb_name)
    depth = o3d.io.read_image(data_dir + 'depth/' + depth_name)
    
    # create rgbd
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    
    # check resolution and channels
    print(rgbd)
        
     # show rgb and depth images
    show_images(rgbd)
    
    # create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    
    # show point cloud
    o3d.visualization.draw_geometries([pcd])