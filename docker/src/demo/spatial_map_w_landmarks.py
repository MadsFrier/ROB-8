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
import sys

from lseg_demo import lseg_image
from depth_npy2png import depth_npy2png


# functions

def load_npy(npy_filepath):
    with open(npy_filepath, 'rb') as f:
        npy = np.load(f)
    return npy

def show_rgbg_o3d(rgbd_image):
    plt.subplot(1, 2, 1)
    plt.title('RGB')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth')
    plt.imshow(rgbd_image.depth)
    plt.show()  

def load_point_clouds(voxel_size=0.0):
    pcds = []
    #emo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in range(0, 6):
        pcd = o3d.io.read_point_cloud("/workspaces/ROB-8/docker/src/content/rs_data/rgb_pcd/rs_"+ str(path) + ".pcd")
        pcd.estimate_normals()
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
        #o3d.visualization.draw_geometries([pcd_down])
    return pcds


def pairwise_registration(source, target, max_correspondence_distance_fine, max_correspondence_distance_coarse):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        #o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        #o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
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
            print('pcd: ', target_id)
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_fine, max_correspondence_distance_coarse)
            print("Build o3d.pipelines.registration.PoseGraph")
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

if __name__ == "__main__":

    # 1. Load rgb, depth, pose

    # Directory of data
    #data_directory = "/home/mads/github/ROB-8/docker/src/content/rs_data/" # This is for users outside of docker only
    data_directory = "/workspaces/ROB-8/docker/src/content/rs_data/" # This is for users inside of docker only

    # specific folders for data
    depth_folder = 'depth/'
    rgb_folder = 'rgb/'
    pose_folder = 'pose/'
    rgb_pcd_folder = 'rgb_pcd/'

    # file name of the data
    file_name = "rs_"

    # get list of files in the directory
    lst = os.listdir(data_directory+rgb_folder)
    lst.sort()
        
    # check if file name is in each element in the list and extract index
    lst_checked = []
    for i in lst:
        if file_name in i:
            lst_checked.append(i[len(file_name):-4])

    # loop though each index and create and save point cloud
    for i in lst_checked:
        
        print('Image: ', i)
        
        #segmentation
        # choose prompt
        #prompt = 'other, floor, ceiling, cabinet, counter, chair, painting, oven, window, wall, sofa, rug'

        # segment image using lseg
        #lseg_img = np.array(lseg_image(data_directory+rgb_folder, [i], prompt, show=True), dtype=np.uint16)

        # resize segmented image to match rgb
        #lseg_img = cv2.resize(lseg_img, dsize=(1080, 720), interpolation=cv2.INTER_CUBIC)

        # save segmentation
        #img_name = img_name[:-4] + ".npy"
        #np.save("/home/mads/github/ROB-8/docker/src/content/demo_data/semantic" + img_name, lseg_img)
        #np.save("/workspaces/ROB-8/docker/src/content/demo_data/semantic/" + img_name, lseg_img)

        # 3. Create rgbd image with rgb and segmentation seperated

        # load color and depth image (right now we assume that images are aligned and have the same res)
        color = o3d.io.read_image(data_directory + rgb_folder + file_name + str(i) + '.png')
        
        depth_npy = load_npy(data_directory + depth_folder + file_name + str(i) + '.npy')
                        
        depth = o3d.geometry.Image(depth_npy.astype(np.uint16))   
             
        # create rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False, depth_trunc=11.0)
        # segd = o3d.geometry.RGBDImage.create_from_color_and_depth(seg, depth, convert_rgb_to_intensity=False)
        
        # show rgb and depth images
        #show_rgbg_o3d(rgbd)

        # 4. Create point cloud for each image
        
        # create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # flip the orientation, so it looks upright, not upside-down
        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        
        # show point cloud
        #o3d.visualization.draw_geometries([pcd])
        
        # save point cloud
        o3d.io.write_point_cloud(data_directory + rgb_pcd_folder + file_name + str(i) + '.pcd', pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)

    # 5. Combine point clouds with multiway registration to create coherent spatial map with landmarks

    voxel_size = 0.02
    pcds_down = load_point_clouds(voxel_size)
    o3d.visualization.draw_geometries(pcds_down)
    
    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)


    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
        
    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw_geometries(pcds_down)

    pcds = load_point_clouds(voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down])



    # 6. Save spatial map with landmarks
