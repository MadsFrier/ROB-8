import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import sqlite3 as sql
from PIL import Image
import io
import cv2

def show_rgbd_o3d(rgbd_image):
    plt.subplot(1, 2, 1)
    plt.title('RGB')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth')
    plt.imshow(rgbd_image.depth)
    plt.show()

# Directory of data
data_dir = "/workspaces/ROB-8/docker/src/content/" # This is for users inside of docker only

# choose database
dataset = 'meetwb_db/'
data_directory = data_dir + dataset

rgb_pcd_folder = 'rgb_pcd/'
seg_pcd_folder = 'seg_pcd/'
rgb_folder = 'rgb/'
seg_folder = 'seg/'
pose_folder = 'pose/'
depth_folder = 'depth/'

# Choose database
db_name = "meetwb.db"

# choose  file name
file_name = "meetwb_"

# choose index to log
chosen_i = 0

conn = sql.connect(data_directory + db_name)

data = conn.execute("SELECT image FROM Data")
images = data.fetchall()

for i, image in enumerate(images):
    
    if i == chosen_i:
        print('Processing rgb', i)
        
        image_blob = image[0]
            
        # Create a PIL Image object from the image blob
        rgb_img = np.array(Image.open(io.BytesIO(image_blob)))
        plt.imsave(data_dir + 'retrieved_data/rgb/rgb.jpg', rgb_img)
        break

data = conn.execute("SELECT depth FROM Data")
depths = data.fetchall()

for i, depth in enumerate(depths):
    
    if i == chosen_i:
        print('Processing depth', i)
        
        depth_blob = depth[0]
            
        # Create a PIL Image object from the image blob
        depth_img = np.array(Image.open(io.BytesIO(depth_blob)))
        plt.imsave(data_dir + 'retrieved_data/depth/depth.jpg', depth_img)
        break
    
data = conn.execute("SELECT pose FROM Node")
poses = data.fetchall()

for i, pose in enumerate(poses):
    
    if i == chosen_i:
        print('Processing pose', i)
        
        print(pose[0])
            
        break

conn.close()

conn = sql.connect(data_directory + db_name[:-3] + '_seg.db')

data = conn.execute("SELECT image FROM Data")
segs = data.fetchall()

for i, seg in enumerate(segs):
    
    if i == chosen_i:
        print('Processing seg', i)
        
        seg_blob = seg[0]
            
        # Create a PIL Image object from the image blob
        seg_img = np.array(Image.open(io.BytesIO(seg_blob)))
        plt.imsave(data_dir + 'retrieved_data/seg/seg.jpg', seg_img)
        break
    
conn.close()

#pcds
bgr_o3d_img = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
seg_o3d_img = o3d.geometry.Image(seg_img)
depth_o3d_img = o3d.geometry.Image(depth_img)   
        
# create rgbd
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(bgr_o3d_img, depth_o3d_img, convert_rgb_to_intensity=False, depth_trunc=11.0)
segd = o3d.geometry.RGBDImage.create_from_color_and_depth(seg_o3d_img, depth_o3d_img, convert_rgb_to_intensity=False, depth_trunc=11.0)

# show rgb/seg and depth images
#show_rgbd_o3d(rgbd)
#show_rgbd_o3d(segd)

# create point cloud
rgb_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
seg_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(segd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# flip the orientation, so it looks upright, not upside-down
rgb_pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
seg_pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

# show point cloud
o3d.visualization.draw_geometries([rgb_pcd])
o3d.visualization.draw_geometries([seg_pcd])

# save point cloud
#o3d.io.write_point_cloud(data_dir + rgb_pcd_folder + 'rgb_pcd.pcd', rgb_pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)
#o3d.io.write_point_cloud(data_dir + rgb_pcd_folder + 'seg_pcd.pcd', seg_pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)

