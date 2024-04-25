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

from demo.lseg_demo import load_lseg, run_lseg, torch_clear_cache

# functions

def load_npy(npy_filepath):
    with open(npy_filepath, 'rb') as f:
        npy = np.load(f)
    return npy

def load_poses(file_path):
    x, y, z, q0, q1, q2, q3 = [], [], [], [], [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            numbers = [float(i) for i in line.split()]
            x.append(numbers[0])
            y.append(numbers[1])
            z.append(numbers[2])
            q0.append(numbers[3])
            q1.append(numbers[4])
            q2.append(numbers[5])
            q3.append(numbers[6])
    return x, y, z, q0, q1, q2, q3

def show_rgbg_o3d(rgbd_image):
    plt.subplot(1, 2, 1)
    plt.title('RGB')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth')
    plt.imshow(rgbd_image.depth)
    plt.show()  

def load_point_clouds(start_file, num_files, data_directory, folder, file_name, voxel_size=0.0):
    pcds = []
    for i in range(start_file, num_files):
        pcd = o3d.io.read_point_cloud(data_directory + folder + file_name + str(i) + '.pcd')
        pcd.estimate_normals()
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds

def pairwise_registration(source, target, max_correspondence_distance_fine, max_correspondence_distance_coarse):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_fine, max_correspondence_distance_coarse)
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def flatten(lst):
    return [x for xs in lst for x in xs]

def pose_to_mat(x, y, z, q0, q1, q2, q3):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    # 3x3 rotation matrix
    trans_matrix = np.array([[r00, r01, r02, x],
                           [r10, r11, r12, y],
                           [r20, r21, r22, z],
                           [0, 0, 0, 1]])
    return trans_matrix
if __name__ == "__main__":

    # Directory of data
    data_directory = "/workspaces/ROB-8/docker/src/content/" # This is for users inside of docker only
    
    # choose dataset
    dataset = 'group_data/'
    data_directory = data_directory + dataset

    # specific folders for data
    depth_folder = 'depth/'
    rgb_folder = 'rgb/'
    pose_folder = 'pose/'
    rgb_pcd_folder = 'rgb_pcd/'
    seg_pcd_folder = 'seg_pcd/'
    map_folder = 'maps/'
    sem_folder = 'semantic/'

    # file name of the data
    file_name = "group_"
    
    # choose file format
    rgb_format = '.jpg' # has to be jpg, otherwise open3d will not create the pcd
    depth_format = '.png'
        
    # set voxel grid size
    voxel_size = 0.001
    
    # allow for lseg to run
    allow_lseg = False
    
    # Choose to create rgb or seg pcd or both
    pcds = [rgb_pcd_folder] # [seg_pcd_folder, rgb_pcd_folder]
    
    # choose prompt
    prompt = 'other, floor, ceiling, table, cabinet, lamp, chair, curtain, window'
    
    # colors associated with prompt
    colors = [
    [0, 0, 0],       # Black
    [255, 0, 0],     # Red
    [0, 255, 0],     # Lime
    [0, 0, 255],     # Blue
    [255, 255, 0],   # Yellow
    [0, 255, 255],   # Cyan
    [255, 0, 255],   # Magenta
    [192, 192, 192], # Silver
    [128, 128, 128], # Gray
    [128, 0, 0],     # Maroon
    [128, 128, 0],   # Olive
    [0, 128, 0],     # Green
    [128, 0, 128],   # Purple
    [0, 128, 128],   # Teal
    [0, 0, 128],     # Navy
    [255, 165, 0],   # Orange
    [255, 215, 0],   # Gold
    [255, 255, 255], # White
    [255, 105, 180], # Hot Pink
    [75, 0, 130],    # Indigo
    [255, 192, 203], # Pink
    [0, 255, 127],   # Spring Green
    [0, 206, 209],   # Dark Turquoise
    [148, 0, 211],   # Dark Violet
    [244, 164, 96]   # Sandy Brown
    ]

    # check if there is enough colors for prompt
    if len(colors) < len(prompt.split(',')):
        print('Not enough colors for the amount of prompts, fix that')
        exit()

    # get list of files in the directory
    lst = os.listdir(data_directory+rgb_folder)
    
    # check if file name is in each element in the list and extract index
    lst_checked = []
    for i in lst:
        if file_name in i:
            lst_checked.append(i)
            
    # sort the list of files
    int_lst = [int(sub[len(file_name):-4]) for sub in lst_checked]
    lst_checked = sorted(int_lst)
            
    ### !!! FOR TESTING ONLY !!! ###
    
    lst_checked = lst_checked[0:3]
        
    ################################
            
    print(len(lst_checked), " images loaded")
    
    # loop though each index and create and save point cloud
    for i in lst_checked:
        print("Processing image", i, 'of', lst_checked[0] + len(lst_checked))
        
        # load rgb, depth
        rgb_img = cv2.imread(data_directory + rgb_folder + file_name + str(i) + rgb_format)
        if depth_format == '.npy':
            depth_img = load_npy(data_directory + depth_folder + file_name + str(i) + depth_format).astype(np.uint16)
            float_test = load_npy(data_directory + depth_folder + file_name + str(i) + depth_format)[0][0]
            if isinstance(float_test, np.floating):
                depth_img = load_npy(data_directory + depth_folder + file_name + str(i) + depth_format)
                depth_img = (depth_img * 1000).astype(np.uint16)
            
        else:
            depth_img = cv2.imread(data_directory + depth_folder + file_name + str(i) + depth_format, cv2.IMREAD_ANYDEPTH)

            
        if allow_lseg:            
            # load lseg model and segment rgb image            
            model, labels = load_lseg(prompt)
            
            palette = flatten(colors[0:len(labels)])
            
            lseg_img, mask_img, patches = run_lseg(rgb_img, model, labels, palette, show=False)
            lseg_img = np.array(lseg_img, dtype=np.uint8)
            mask_img = np.array(mask_img)

            #clear cache to clear GPU memory for next iteration
            torch_clear_cache(print_cache=True)
            
            # resize segmented image to match rgb and convert to rgb to add color corresponding to labels
            lseg_img = cv2.resize(lseg_img, dsize=(rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask_img = cv2.resize(mask_img, dsize=(rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            lseg_img = cv2.cvtColor(lseg_img, cv2.COLOR_GRAY2RGB)

            #for i in range(len(labels)):
            #    lseg_img = np.where(lseg_img == [i, i, i], colors[i], lseg_img) 
                         
            #lseg_img = cv2.cvtColor(lseg_img, cv2.COLOR_RGB2BGR)
            
            # save segmentation
            plt.imsave(data_directory + sem_folder + file_name + str(i) + ".png", mask_img)

        # load color , seg, and depth image (right now we assume that images are aligned and have the same res)
        
        bgr_o3d_img = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        depth_o3d_img = o3d.geometry.Image(depth_img)   
             
        # create rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(bgr_o3d_img, depth_o3d_img, convert_rgb_to_intensity=False, depth_trunc=11.0)
        
        # show rgb and depth images
        # show_rgbg_o3d(rgbd)
        # show_rgbd_o3d(segd)
        
        # create point cloud
        rgb_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # flip the orientation, so it looks upright, not upside-down
        rgb_pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        
        # show point cloud
        #o3d.visualization.draw_geometries([rgb_pcd])
        
        # save point cloud
        o3d.io.write_point_cloud(data_directory + rgb_pcd_folder + file_name + str(i) + '.pcd', rgb_pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)

        if allow_lseg:
            lseg_o3d_img = o3d.geometry.Image(lseg_img) 
            mask_o3d_img = o3d.geometry.Image(cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
            segd = o3d.geometry.RGBDImage.create_from_color_and_depth(mask_o3d_img, depth_o3d_img, convert_rgb_to_intensity=False, depth_trunc=11.0)
            seg_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(segd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
            seg_pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            #o3d.visualization.draw_geometries([seg_pcd])
            o3d.io.write_point_cloud(data_directory + seg_pcd_folder + file_name + str(i) + '.pcd', seg_pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)

    for i in pcds:
        print('Creating', i, 'point cloud')
        # 5. Combine point clouds with multiway registration to create coherent spatial map with landmarks
        print('loading point clouds...')
        pcds_down = load_point_clouds(lst_checked[0], lst_checked[0]+len(lst_checked), data_directory, str(i), file_name, voxel_size)
        #o3d.visualization.draw_geometries(pcds_down)
        
        print('Visualising combined point clouds...')

        pcds = load_point_clouds(lst_checked[0], lst_checked[0]+len(lst_checked), data_directory, str(i), file_name, voxel_size)
        pcd_combined = o3d.geometry.PointCloud()
        
        x, y, z, q1, q2, q3, q4 = load_poses(data_directory + pose_folder + file_name + '.txt')
        for point_id in range(len(pcds)):
            trans_mat = pose_to_mat(x[point_id], y[point_id], z[point_id], q1[point_id], q2[point_id], q3[point_id], q4[point_id])
            print(trans_mat)
            pcds[point_id].transform(trans_mat)
            pcds[point_id].transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            pcd_combined += pcds[point_id]
        pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
        o3d.io.write_point_cloud(data_directory + map_folder + 'map_' + str(i)[:-5] + '.pcd', pcd_combined_down, format='auto', write_ascii=False, compressed=False, print_progress=False)
        print('Point cloud saved')
        o3d.visualization.draw_geometries([pcd_combined_down])