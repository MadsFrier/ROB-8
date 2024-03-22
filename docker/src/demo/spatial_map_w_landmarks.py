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
import numpy as np
import cv2
import pyrealsense2 as rs

from lseg_demo import lseg_image
from depth_npy2png import depth_npy2png


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

    #segmentation
    # choose prompt
    prompt = 'other, floor, ceiling, cabinet, counter, chair, painting, oven, window, wall, sofa, rug'

    # segment image using lseg
    lseg_img = np.array(lseg_image(data_directory+rgb_folder, [i], prompt, show=True), dtype=np.uint16)

    # resize segmented image to match rgb
    lseg_img = cv2.resize(lseg_img, dsize=(1080, 720), interpolation=cv2.INTER_CUBIC)

    # save segmentation
    #img_name = img_name[:-4] + ".npy"
    #np.save("/home/mads/github/ROB-8/docker/src/content/demo_data/semantic" + img_name, lseg_img)
    #np.save("/workspaces/ROB-8/docker/src/content/demo_data/semantic/" + img_name, lseg_img)


    #depth
    # convert depth image to .png and show
    depth_npy2png(data_directory+depth_folder, [i], show=True)


   

    # 3. Create rgbd image with rgb and segmentation seperated

    # load color and depth image (right now we assume that images are aligned and hacve the same res)
    color = o3d.io.read_image(data_directory + rgb_folder + file_name + str(i) + '.png')
    depth = o3d.io.read_image(data_directory + depth_folder + file_name + str(i) + '.png')
    
    # create rgbd
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    # segd = o3d.geometry.RGBDImage.create_from_color_and_depth(seg, depth, convert_rgb_to_intensity=False)
    
    # show rgb and depth images
    # show_rgb_depth(rgbd)

    # 4. Create point cloud for each image




# 5. Combine point clouds with multiway registration to create coherent spatial map with landmarks



# 6. Save spatial map with landmarks
