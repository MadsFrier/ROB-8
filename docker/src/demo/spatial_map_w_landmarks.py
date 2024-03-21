'''
Full pipeline for spatial map with landmarks
1. Load rgb, depth, pose
2. Align depth and segmentation with rgb
3. Create rgbd image with rgb and segmentation seperated
4. Create point cloud for each image
5. Combine point clouds with multiway registration to create coherent spatial map with landmarks
6. Save spatial map with landmarks
'''

#imports

import os
import matplotlib.pyplot as plt
import open3d as o3d

# functions

def show_rgb_depth(rgbd_image):
    plt.subplot(1, 2, 1)
    plt.title('RGB')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth')
    plt.imshow(rgbd_image.depth)
    plt.show()  

# 1. Load rgb, depth, pose

# Directory of data
data_directory = "/home/mads/github/ROB-8/docker/src/content/demo_data/" # This is for users outside of docker only
#data_directory = "/workspaces/ROB-8/docker/src/content/demo_data/" # This is for users inside of docker only

# specific folders for data
depth_folder = 'depth/'
rgb_folder = 'rgb/'
pose_folder = 'pose/'

# file name of the data
file_name = "5LpN3gDmAk7_"

# get list of files in the directory
lst = os.listdir(data_directory+rgb_folder)
lst.sort()

# check if file name is in each element in the list and extract index
lst_checked = []
for i in lst:
    if file_name in i:
        lst_checked.append(i[12:-4])

# loop though each index and create and save point cloud
for i in lst_checked:

    # 2. Align depth and segmentation with rgb

    # ignore for now

    # 3. Create rgbd image with rgb and segmentation seperated

    # load color and depth image
    color = o3d.io.read_image(data_directory + rgb_folder + file_name + str(i) + '.png')
    depth = o3d.io.read_image(data_directory + depth_folder + file_name + str(i) + '.png')
    
    # create rgbd
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    
    show_rgb_depth(rgbd)

    # 4. Create point cloud for each image



# 5. Combine point clouds with multiway registration to create coherent spatial map with landmarks



# 6. Save spatial map with landmarks
