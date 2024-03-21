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




# 1. Load rgb, depth, pose

data_directory = "/home/gayath/project/ROB-8/docker/src/content/demo_data/"
depth_folder = 'depth/'
rgb_folder = 'rgb/'
pose_folder = 'pose/'

file_name = "5LpN3gDmAk7_"

for i in range(len(data_directory+rgb_folder)):
    print(i)

# 2. Align depth and segmentation with rgb

# ignore for now

# 3. Create rgbd image with rgb and segmentation seperated



# 4. Create point cloud for each image



# 5. Combine point clouds with multiway registration to create coherent spatial map with landmarks



# 6. Save spatial map with landmarks
